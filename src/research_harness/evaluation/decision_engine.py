"""The decision engine: deterministic gates between research phases.

Everything here is ordinary code. The sequence implements the spec's
recommended ranking flow:

    hard-constraint filter
      -> absolute rubric scores (MCDA)
      -> pairwise competition
      -> disagreement analysis
      -> weight sensitivity
      -> Top-K retention
      -> (re-rank after small-scale validation)

Two things it deliberately refuses to do:

* promote a candidate that FAILs a hard constraint, however high it scores;
* claim global optimality. Every ranking carries the scope caveat that it is
  "better among the candidates considered, under current evidence and weights".
"""

from __future__ import annotations

from ..core.schemas import (
    ConstraintStatus,
    ExperimentCandidate,
    ExperimentResult,
    MetricDirection,
    MetricVerdict,
    PairwiseReport,
    Proposal,
    ProposalEvaluation,
    Provenance,
    Ranking,
    RankEntry,
    ResearchContract,
    StabilityReport,
    StopDecision,
    MaturityLevel,
    ValidationResult,
)
from .constraints import filter_eligible
from .stability import order_of


def build_ranking(
    *,
    run_id: str,
    rubric_version: str,
    evaluations: list[ProposalEvaluation],
    top_k: int = 2,
    pairwise: PairwiseReport | None = None,
    stability: StabilityReport | None = None,
) -> Ranking:
    """Produce the ranking. Constraint-failing candidates are excluded outright."""
    _, excluded = filter_eligible(evaluations)
    ordered_ids = order_of(evaluations)
    by_id = {e.proposal_id: e for e in evaluations}
    points = (pairwise.win_points if pairwise else {}) or {}

    entries: list[RankEntry] = []
    for rank, pid in enumerate(ordered_ids, start=1):
        e = by_id[pid]
        entries.append(
            RankEntry(
                proposal_id=pid,
                rank=rank,
                point_score=e.point_score,
                score_lower=e.score_lower,
                score_upper=e.score_upper,
                constraint_status=e.constraint_status,
                eligible=e.eligible,
                pairwise_points=points.get(pid, 0.0),
                evidence_sufficiency=e.evidence_sufficiency,
            )
        )

    # Top-K is drawn only from eligible candidates, in ranked order.
    selected = [pid for pid in ordered_ids if by_id[pid].eligible][:top_k]

    return Ranking(
        run_id=run_id,
        rubric_version=rubric_version,
        entries=entries,
        excluded_for_constraints=excluded,
        top_k=selected,
        stability=stability,
        pairwise=pairwise,
    )


def select_winner(ranking: Ranking) -> str | None:
    """Highest-ranked eligible candidate, or None if everything is excluded."""
    for e in sorted(ranking.entries, key=lambda x: x.rank):
        if e.eligible:
            return e.proposal_id
    return None


# --------------------------------------------------------------------------
# Value of information
# --------------------------------------------------------------------------


def rank_experiments(
    *,
    proposals: list[Proposal],
    evaluations: list[ProposalEvaluation],
    ranking: Ranking,
) -> list[ExperimentCandidate]:
    """Order candidate experiments by predicted decision impact per unit cost.

        priority = predicted_decision_impact / estimated_cost

    ``predicted_decision_impact`` is a *structured prediction*, flagged as such
    on every record (``impact_is_predicted=True``). It is not a measured or
    calibrated quantity, and it does not replace research judgement -- it only
    orders a shortlist for a human or the planner to choose from.

    Impact heuristic: an assumption is worth more to resolve when it is held by a
    top-ranked candidate, when that candidate is under-evidenced, and when the
    score gap to its nearest rival is small (i.e. the decision is genuinely open).
    """
    by_id = {e.proposal_id: e for e in evaluations}
    prop_by_id = {p.proposal_id: p for p in proposals}
    eligible = [e for e in sorted(ranking.entries, key=lambda x: x.rank) if e.eligible]

    # How contested is the decision? Small top-two gap -> experiments matter more.
    if len(eligible) >= 2:
        gap = abs(eligible[0].point_score - eligible[1].point_score)
        contested = max(0.0, 1.0 - gap / 25.0)  # 25 points apart -> settled
    else:
        contested = 0.3

    candidates: list[ExperimentCandidate] = []
    for entry in eligible:
        ev = by_id[entry.proposal_id]
        prop = prop_by_id.get(entry.proposal_id)
        if prop is None:
            continue
        rank_factor = 1.0 / entry.rank
        gap_factor = 1.0 - ev.evidence_sufficiency  # more missing evidence -> more value

        assumptions = list(dict.fromkeys(ev.unresolved_assumptions)) or []
        for assumption in assumptions:
            # UNKNOWN hard constraints dominate: they can disqualify outright.
            blocks_constraint = any(
                assumption == a.unverified_assumption
                for a in ev.constraint_assessments
                if a.status is ConstraintStatus.UNKNOWN
            )
            impact = 0.25 * rank_factor + 0.3 * gap_factor + 0.25 * contested
            if blocks_constraint:
                impact += 0.3
            impact = min(1.0, round(impact, 4))
            cost = prop.poc.estimated_cost_units
            candidates.append(
                ExperimentCandidate(
                    label=(
                        f"Resolve for {entry.proposal_id}: {assumption[:110]}"
                        + ("..." if len(assumption) > 110 else "")
                    ),
                    targets_assumption=assumption,
                    proposal_ids=[entry.proposal_id],
                    predicted_decision_impact=impact,
                    estimated_cost_units=cost,
                    priority=round(impact / cost, 4),
                    impact_is_predicted=True,
                )
            )

    return sorted(candidates, key=lambda c: (-c.priority, c.label))


# --------------------------------------------------------------------------
# Validation level
# --------------------------------------------------------------------------


def verdicts_for(
    contract: ResearchContract, experiment: ExperimentResult
) -> list[MetricVerdict]:
    """Compare observations to contract thresholds. Pure arithmetic.

    Metrics with no observation come back ``met=None`` (unmeasured), never False,
    because "we did not measure it" and "it failed" are different findings.
    """
    obs = {o.metric_key: o for o in experiment.observations}
    out: list[MetricVerdict] = []
    for m in contract.success_metrics:
        o = obs.get(m.key)
        if o is None:
            out.append(
                MetricVerdict(
                    metric_key=m.key,
                    target_threshold=m.target_threshold,
                    direction=m.direction,
                    observed=None,
                    met=None,
                    provenance=Provenance.AGENT_INFERENCE,
                )
            )
            continue
        met = m.is_met(o.value) if m.target_threshold is not None else None
        out.append(
            MetricVerdict(
                metric_key=m.key,
                target_threshold=m.target_threshold,
                direction=m.direction,
                observed=o.value,
                met=met,
                provenance=o.provenance,
            )
        )
    return out


def determine_maturity_level(
    *,
    contract: ResearchContract,
    experiment: ExperimentResult | None,
    verdicts: list[MetricVerdict],
    implementation_conforms: bool | None,
    constraint_status: ConstraintStatus,
    has_real_evidence: bool,
    has_test_plan: bool,
) -> tuple[MaturityLevel, str]:
    """Decide the honest validation level. Returns ``(level, reason)``.

    The central guard: **simulated or unexecuted experiments can never reach L4
    or L5.** A run whose only "results" are synthetic is INCONCLUSIVE, with the
    reason stated plainly, no matter how good the numbers look.
    """
    if constraint_status is ConstraintStatus.FAIL:
        return (
            MaturityLevel.REJECTED,
            "A hard constraint is violated; the candidate is not viable regardless "
            "of its other scores.",
        )

    # Checked before the "not executed" branch below: an experiment that was
    # *attempted and blocked* on real-world input is a different finding from one
    # that was never run, and must not be reported as merely "designed".
    if experiment is not None and experiment.requires_human_input:
        return (
            MaturityLevel.INCONCLUSIVE,
            "Validation requires real-world input (users, devices, or physical "
            "measurement) that this harness cannot obtain autonomously. A concrete "
            "external validation plan is attached; the goal is neither confirmed "
            "nor refuted.",
        )

    if experiment is None or not experiment.executed:
        base = (
            MaturityLevel.R2_DESIGNED
            if has_test_plan
            else (MaturityLevel.R1_RESEARCHED if has_real_evidence else MaturityLevel.INCONCLUSIVE)
        )
        if base is MaturityLevel.INCONCLUSIVE:
            return (
                base,
                "No experiment was executed and no non-simulated evidence was "
                "gathered; nothing has been verified.",
            )
        return (
            base,
            "A complete design and test plan exist, but no experiment has been "
            "executed, so no empirical claim is made.",
        )

    if experiment.is_simulated:
        return (
            MaturityLevel.R3_PROTOTYPED,
            "A proof-of-concept ran, but on SIMULATED data from a mock adapter. "
            "Mock results exercise the pipeline and must not be read as empirical "
            "evidence about the real goal, so maturity stops at R3_PROTOTYPED.",
        )

    # Real execution from here on.
    if implementation_conforms is False:
        return (
            MaturityLevel.R3_PROTOTYPED,
            "The implementation was executed but does not conform to the selected "
            "proposal's specification; goal attainment cannot be attributed.",
        )

    # Three distinct states, deliberately not collapsed into "unmeasured":
    #   decided    -- observed AND a threshold exists, so met is True/False
    #   undecidable-- observed but the contract set no threshold to judge it by
    #   unobserved -- nothing was measured at all
    # Reporting an undecidable metric as "not measured" blames the experiment
    # for a gap in the contract, which sends the reader to fix the wrong thing.
    decided = [v for v in verdicts if v.met is not None]
    undecidable = [v.metric_key for v in verdicts if v.met is None and v.observed is not None]
    unobserved = [v.metric_key for v in verdicts if v.observed is None]

    if not decided:
        if undecidable:
            return (
                MaturityLevel.R3_PROTOTYPED,
                f"The experiment ran and measured {', '.join(undecidable)}, but the "
                "contract sets no numeric threshold for "
                f"{'them' if len(undecidable) > 1 else 'it'}, so goal attainment "
                "cannot be decided. This is a gap in the contract, not in the "
                "experiment: supply a target and re-judge.",
            )
        return (
            MaturityLevel.R3_PROTOTYPED,
            "The experiment ran, but none of the contract's success metrics were "
            "observed at all, so goal attainment is unknown.",
        )
    unmeasured = [v.metric_key for v in verdicts if v.met is None]
    all_met = all(v.met for v in decided)
    measured = decided

    if all_met and not unmeasured:
        if constraint_status is ConstraintStatus.UNKNOWN:
            return (
                MaturityLevel.R4_VALIDATED,
                "All measured metrics met their contract thresholds under real "
                "execution, but at least one hard constraint remains unverified, so "
                "full goal achievement is not asserted.",
            )
        return (
            MaturityLevel.R5_GOAL_ACHIEVED,
            "Every success metric in the research contract was independently "
            "measured under real execution and met its threshold, with all hard "
            "constraints satisfied.",
        )
    if all_met and unmeasured:
        return (
            MaturityLevel.R4_VALIDATED,
            "All measured metrics passed under real execution, but "
            f"{len(unmeasured)} contract metric(s) were never measured "
            f"({', '.join(unmeasured)}); the original goal is not fully demonstrated.",
        )
    failed = [v.metric_key for v in measured if not v.met]
    return (
        MaturityLevel.R4_VALIDATED,
        "The implementation was independently validated as running correctly, but "
        f"metric(s) {', '.join(failed)} did not reach the contract threshold -- the "
        "research goal is not met.",
    )


def build_validation_result(
    *,
    contract: ResearchContract,
    proposal_id: str,
    experiment: ExperimentResult,
    implementation_conforms: bool | None,
    implementation_notes: str,
    failed_test_ids: list[str],
    constraint_status: ConstraintStatus,
    has_real_evidence: bool,
    has_test_plan: bool,
    external_plan: list[str] | None = None,
    evidence_gaps: list[str] | None = None,
) -> ValidationResult:
    verdicts = verdicts_for(contract, experiment)
    level, reason = determine_maturity_level(
        contract=contract,
        experiment=experiment,
        verdicts=verdicts,
        implementation_conforms=implementation_conforms,
        constraint_status=constraint_status,
        has_real_evidence=has_real_evidence,
        has_test_plan=has_test_plan,
    )

    # goal_satisfied is True only at L5 -- derived from the contract and real
    # measurement, never from a proposal's own definition of done.
    if level is MaturityLevel.R5_GOAL_ACHIEVED:
        goal_satisfied: bool | None = True
    elif level in (MaturityLevel.INCONCLUSIVE, MaturityLevel.R1_RESEARCHED,
                   MaturityLevel.R2_DESIGNED, MaturityLevel.R3_PROTOTYPED):
        goal_satisfied = None  # genuinely unknown
    elif level is MaturityLevel.REJECTED:
        goal_satisfied = False
    else:  # L4
        measured = [v for v in verdicts if v.met is not None]
        goal_satisfied = False if any(not v.met for v in measured) else None

    gaps = list(evidence_gaps or [])
    for v in verdicts:
        if v.met is None and v.observed is not None:
            gaps.append(
                f"Metric '{v.metric_key}' was measured ({v.observed}) but the "
                "contract sets no threshold, so it cannot decide goal attainment."
            )
        elif v.met is None:
            gaps.append(f"Metric '{v.metric_key}' was never measured.")
        elif v.provenance is Provenance.SIMULATED:
            gaps.append(
                f"Metric '{v.metric_key}' value came from SIMULATED data, not measurement."
            )

    return ValidationResult(
        run_id=contract.run_id,
        proposal_id=proposal_id,
        experiment_id=experiment.experiment_id,
        implementation_conforms=implementation_conforms,
        implementation_notes=implementation_notes,
        failed_test_ids=failed_test_ids,
        metric_verdicts=verdicts,
        goal_satisfied=goal_satisfied,
        maturity_level=level,
        evidence_gaps=list(dict.fromkeys(gaps)),
        blocking_reason=reason,
        external_validation_plan=list(external_plan or []),
    )


# --------------------------------------------------------------------------
# Stop policy
# --------------------------------------------------------------------------


def should_stop(
    *,
    contract: ResearchContract,
    iteration: int,
    elapsed_seconds: float,
    tokens_used: int,
    cost_usd: float,
    consecutive_no_gain: int,
    repair_attempts: int,
    goal_satisfied: bool | None,
    candidates_remaining: int,
) -> StopDecision:
    """Deterministic termination check. There is no unbounded loop anywhere."""
    b = contract.budget

    if goal_satisfied is True:
        return StopDecision(
            should_stop=True,
            reason="goal_satisfied",
            detail="The original research goal was independently validated as met.",
        )
    if candidates_remaining <= 0:
        return StopDecision(
            should_stop=True,
            reason="no_viable_candidates",
            detail="Every candidate has been rejected or exhausted.",
        )
    if iteration >= b.max_iterations:
        return StopDecision(
            should_stop=True,
            reason="max_iterations",
            detail=f"Reached the configured iteration budget ({b.max_iterations}).",
        )
    if b.max_wall_clock_seconds is not None and elapsed_seconds >= b.max_wall_clock_seconds:
        return StopDecision(
            should_stop=True,
            reason="wall_clock_exhausted",
            detail=f"Elapsed {elapsed_seconds:.1f}s >= budget {b.max_wall_clock_seconds}s.",
        )
    if b.max_tokens is not None and tokens_used >= b.max_tokens:
        return StopDecision(
            should_stop=True,
            reason="token_budget_exhausted",
            detail=f"Used {tokens_used} >= budget {b.max_tokens} tokens.",
        )
    if b.max_cost_usd is not None and cost_usd >= b.max_cost_usd:
        return StopDecision(
            should_stop=True,
            reason="cost_budget_exhausted",
            detail=f"Spent ${cost_usd:.4f} >= budget ${b.max_cost_usd:.4f}.",
        )
    if consecutive_no_gain >= b.max_consecutive_no_gain:
        return StopDecision(
            should_stop=True,
            reason="no_progress",
            detail=(
                f"{consecutive_no_gain} consecutive iteration(s) produced no measurable "
                "improvement; continuing would burn budget without new information."
            ),
        )
    if repair_attempts > b.max_repair_attempts:
        return StopDecision(
            should_stop=True,
            reason="repair_attempts_exhausted",
            detail=(
                f"Implementation repair attempted {repair_attempts} time(s), over the "
                f"limit of {b.max_repair_attempts}; the failure is not a simple "
                "implementation bug."
            ),
        )
    return StopDecision(should_stop=False, reason="continue", detail="Budget remains.")


def scope_qualified_claim(ranking: Ranking, winner: str | None) -> str:
    """Phrase the recommendation with its real scope. No global-optimum claims."""
    if winner is None:
        return (
            "No candidate is recommended: every candidate considered either violates "
            "a hard constraint or lacks the evidence to be preferred."
        )
    n = len(ranking.entries)
    caveat = ""
    if ranking.stability and not ranking.stability.top1_stable:
        caveat = (
            " This preference is NOT stable under small weight changes "
            f"({', '.join(ranking.stability.unstable_dimensions)}), so it should be "
            "treated as provisional pending more evidence."
        )
    return (
        f"{winner} is the preferred candidate among the {n} considered, under the "
        f"current evidence, the contract's goals, and rubric {ranking.rubric_version}. "
        "This is not a claim of global optimality." + caveat
    )
