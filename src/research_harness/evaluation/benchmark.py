"""Benchmarking the scoring system itself (spec section 15).

The problem with evaluating a judge is where ground truth comes from. The spec
rules out the easy answers: the judge labelling its own truth, multi-agent
votes as truth, model preference read as real-world probability, or outcomes
leaking back into the prior baseline.

**Our ground truth is construction, not labelling.** If candidate A is at least
as good as B on *every* rubric dimension and strictly better on some, then A
dominates B, and preferring B is wrong for any non-negative weighting. No human
judgement and no outcome data is needed -- the answer follows from the
definition. That gives real, if narrow, ground truth.

What this does and does not measure:

* **Does** detect a judge that is position-biased, insensitive to the attributes
  its rubric claims to score, inconsistent under relabelling, or willing to
  prefer a constraint-violating candidate.
* **Does not** measure whether the judge has good taste on genuinely contested
  trade-offs -- the cases where real expertise matters. Dominance cases are the
  easy ones. A judge that scores 1.0 here is not therefore good; it has merely
  failed to be obviously broken.

Calibration (do high scores predict real success?) is not here: it needs real
outcome labels. See :mod:`research_harness.evaluation.calibration`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.schemas import (
    Base,
    ConstraintStatus,
    Evidence,
    HardConstraint,
    MetricDirection,
    PoCSketch,
    Proposal,
    Provenance,
    ResearchContract,
    Rubric,
    Stance,
    SuccessMetric,
)
from . import constraints as constraint_mod
from .decision_engine import build_ranking, select_winner
from .mcda import evaluate_proposal
from .pairwise import run_pairwise
from .rubric import build_rubric
from .stability import kendall_tau, order_of, relabel_invariance, weight_sensitivity

BENCHMARK_RUN_ID = "benchmark"


# --------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------


@dataclass
class BenchmarkCase:
    """One benchmark scenario with objectively-derived expectations."""

    name: str
    rationale: str
    contract: ResearchContract
    evidence: list[Evidence]
    proposals: list[Proposal]
    #: Pairs ``(better, worse)`` where the first dominates the second.
    dominance_pairs: list[tuple[str, str]] = field(default_factory=list)
    #: Reference ordering, best first, where dominance makes it unambiguous.
    reference_ranking: list[str] = field(default_factory=list)
    #: Candidates that must be excluded for violating a hard constraint.
    expected_excluded: list[str] = field(default_factory=list)
    #: Candidate that must be selected, when dominance determines it.
    expected_winner: str | None = None


def _contract(
    *, cost_limit: float | None = 4.0, metrics: list[SuccessMetric] | None = None
) -> ResearchContract:
    constraints: list[HardConstraint] = []
    if cost_limit is not None:
        constraints.append(
            HardConstraint(
                key="within_cost_budget",
                description=f"declared effort must not exceed {cost_limit} cost units",
                category="budget",
                limit_metric="declared_cost_units",
                limit_value=cost_limit,
                limit_direction=MetricDirection.MINIMIZE,
            )
        )
    return ResearchContract(
        run_id=BENCHMARK_RUN_ID,
        original_idea="Benchmark: improve throughput within a fixed effort budget",
        problem_statement="Synthetic benchmark contract with constructed ground truth.",
        goal_type="technical_performance",
        success_metrics=metrics
        or [
            SuccessMetric(
                key="throughput",
                description="requests served per second",
                unit="rps",
                target_threshold=100.0,
                direction=MetricDirection.MAXIMIZE,
                measurement_method="load test, 5 repeats",
            )
        ],
        hard_constraints=constraints,
        available_resources=["existing runtime", "existing test harness"],
    )


def _evidence() -> list[Evidence]:
    return [
        Evidence(
            evidence_id="E001",
            claim="Benchmark evidence: the approach improves throughput in comparable settings.",
            source_type="paper",
            source_uri=None,
            provenance=Provenance.SOURCE,
            stance=Stance.SUPPORTS,
            quality=5,
            verification_status="cross_checked",
        ),
        Evidence(
            evidence_id="E002",
            claim="Benchmark evidence: a weak, tangential report of the same effect.",
            source_type="web",
            source_uri=None,
            provenance=Provenance.SOURCE,
            stance=Stance.UNCERTAIN,
            quality=1,
            verification_status="unverified",
        ),
    ]


def _proposal(
    pid: str,
    *,
    archetype: str,
    cost: float,
    deps: list[str],
    failure_modes: list[str],
    assumptions: list[str],
    evidence_ids: list[str],
    falsifier: str,
    poc_cost: float,
    goal_mapping: str,
    differentiators: list[str],
    mechanism: str,
) -> Proposal:
    return Proposal(
        proposal_id=pid,
        run_id=BENCHMARK_RUN_ID,
        title=f"Benchmark candidate {pid}",
        strategy_archetype=archetype,
        core_mechanism=mechanism,
        goal_mapping=goal_mapping,
        dependencies=deps,
        premises=["benchmark premise"],
        key_steps=["measure baseline", "apply mechanism", "re-measure"],
        resource_needs=["existing runtime"],
        differentiators=differentiators,
        failure_modes=failure_modes,
        critical_assumptions=assumptions,
        poc=PoCSketch(
            description="minimal benchmark trial",
            expected_result="a throughput measurement",
            falsification_condition=falsifier,
            estimated_cost_units=poc_cost,
        ),
        evidence_ids=evidence_ids,
        declared_cost_units=cost,
        generated_by="benchmark-constructor",
    )


def dominance_case() -> BenchmarkCase:
    """STRONG dominates WEAK on every attribute the rubric scores.

    Both share an archetype, so archetype-driven dimensions (expected benefit,
    novelty) tie rather than cutting across the comparison. STRONG then has:
    fewer unmet dependencies, lower cost, fewer failure modes, fewer unverified
    assumptions, a stated falsifier, a cheaper PoC, high-quality cross-checked
    evidence, and explicit metric mapping. Preferring WEAK is wrong under any
    non-negative weighting.
    """
    contract = _contract()
    strong = _proposal(
        "STRONG",
        archetype="balanced",
        cost=1.0,
        deps=["existing runtime"],  # covered by available_resources
        failure_modes=["one bounded, detectable failure mode"],
        assumptions=[],
        evidence_ids=["E001"],
        falsifier="throughput does not exceed the baseline beyond variance",
        poc_cost=1.0,
        goal_mapping="directly targets the contract metric throughput in rps",
        differentiators=["smallest blast radius", "cheapest to verify"],
        mechanism="apply a well-understood batching change behind the existing interface",
    )
    weak = _proposal(
        "WEAK",
        archetype="balanced",
        cost=3.0,
        deps=["an unavailable external queue", "a second unavailable service"],
        failure_modes=["mode a", "mode b", "mode c", "mode d"],
        assumptions=["unverified a", "unverified b", "unverified c"],
        evidence_ids=["E002"],
        falsifier="",  # unfalsifiable as stated
        poc_cost=5.0,
        goal_mapping="may help overall performance somehow",
        differentiators=[],
        mechanism="route traffic through an additional unavailable queueing layer",
    )
    return BenchmarkCase(
        name="dominance",
        rationale=(
            "STRONG is at least as good as WEAK on every rubric dimension and "
            "strictly better on most. Any non-negative weighting must rank it first."
        ),
        contract=contract,
        evidence=_evidence(),
        proposals=[strong, weak],
        dominance_pairs=[("STRONG", "WEAK")],
        reference_ranking=["STRONG", "WEAK"],
        expected_winner="STRONG",
    )


def constraint_case() -> BenchmarkCase:
    """An otherwise-excellent candidate that breaches the cost ceiling.

    OVERBUDGET is built to score well on everything except cost, and exceeds the
    machine-checkable limit. It must be excluded regardless of its score, and
    the modest compliant candidate must win.
    """
    contract = _contract(cost_limit=4.0)
    overbudget = _proposal(
        "OVERBUDGET",
        archetype="exploratory",
        cost=9.0,  # breaches the 4.0 limit
        deps=["existing runtime"],
        failure_modes=[],
        assumptions=[],
        evidence_ids=["E001"],
        falsifier="throughput does not exceed the baseline",
        poc_cost=1.0,
        goal_mapping="directly targets the contract metric throughput in rps",
        differentiators=["large headroom", "novel mechanism"],
        mechanism="restructure the pipeline around a different primitive entirely",
    )
    compliant = _proposal(
        "COMPLIANT",
        archetype="conservative",
        cost=2.0,
        deps=["existing runtime"],
        failure_modes=["one mode", "another mode"],
        assumptions=["one unverified assumption"],
        evidence_ids=["E002"],
        falsifier="throughput does not exceed the baseline",
        poc_cost=2.0,
        goal_mapping="targets throughput in rps",
        differentiators=["proven"],
        mechanism="tune the existing configuration within current structure",
    )
    return BenchmarkCase(
        name="constraint_non_compensation",
        rationale=(
            "OVERBUDGET violates a machine-checkable hard constraint. It must be "
            "ineligible however well it scores; COMPLIANT must be selected."
        ),
        contract=contract,
        evidence=_evidence(),
        proposals=[overbudget, compliant],
        expected_excluded=["OVERBUDGET"],
        expected_winner="COMPLIANT",
    )


def evidence_case() -> BenchmarkCase:
    """Identical candidates except for the evidence behind them.

    CITED and UNCITED are structurally the same; only CITED is backed by a real,
    cross-checked source. A judge that scores evidence strength must prefer
    CITED, and must mark the uncited one UNKNOWN rather than guessing a number.
    """
    contract = _contract()
    shared = dict(
        archetype="balanced",
        cost=2.0,
        deps=["existing runtime"],
        failure_modes=["one mode"],
        assumptions=[],
        falsifier="throughput does not exceed the baseline",
        poc_cost=1.0,
        goal_mapping="directly targets the contract metric throughput in rps",
        differentiators=["clear boundary"],
        mechanism="apply a batching change behind the existing interface",
    )
    return BenchmarkCase(
        name="evidence_grounding",
        rationale=(
            "The two candidates differ only in whether a real source backs them. "
            "The uncited one must score UNKNOWN on evidence strength, not a guess."
        ),
        contract=contract,
        evidence=_evidence(),
        proposals=[
            _proposal("CITED", evidence_ids=["E001"], **shared),
            _proposal("UNCITED", evidence_ids=[], **shared),
        ],
        dominance_pairs=[("CITED", "UNCITED")],
        reference_ranking=["CITED", "UNCITED"],
    )


def default_cases() -> list[BenchmarkCase]:
    return [dominance_case(), constraint_case(), evidence_case()]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


class CaseResult(Base):
    name: str
    rationale: str
    produced_ranking: list[str]
    reference_ranking: list[str]
    rank_correlation: float | None = None
    pairwise_agreement: float | None = None
    pairwise_decided: int = 0
    pairwise_total: int = 0
    top_k_recall: float | None = None
    position_bias_rate: float = 0.0
    order_consistency: float = 1.0
    relabel_invariant: bool = True
    top1_stable: bool = True
    min_kendall_tau: float = 1.0
    evidence_grounding: float = 0.0
    unknown_rate: float = 0.0
    constraint_respected: bool = True
    goal_fidelity: bool = True
    selected: str | None = None
    notes: list[str] = []


class BenchmarkReport(Base):
    judge_id: str
    cases: list[CaseResult] = []
    pairwise_agreement: float | None = None
    mean_rank_correlation: float | None = None
    top_k_recall: float | None = None
    position_bias_rate: float = 0.0
    order_consistency: float = 1.0
    evidence_grounding: float = 0.0
    constraint_respect: float = 1.0
    goal_fidelity: float = 1.0
    relabel_invariant: bool = True
    passed: bool = True
    failures: list[str] = []
    scope_note: str = (
        "Ground truth here is dominance by construction, not expert labelling. "
        "These cases detect a broken judge; they do not establish that a judge "
        "handles genuinely contested trade-offs well. Calibration against real "
        "outcomes is a separate question (see the calibration module)."
    )


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def run_case(
    *, judge, case: BenchmarkCase, rubric: Rubric | None = None, top_k: int = 1
) -> CaseResult:
    r = rubric or build_rubric(case.contract, version="benchmark-v1")
    notes: list[str] = []

    evaluations = []
    for proposal in case.proposals:
        scoring = judge.score_rubric(
            goal=case.contract, rubric=r, proposal=proposal, evidence=case.evidence
        )
        assessments = constraint_mod.ensure_complete(
            case.contract, scoring.constraint_assessments
        )
        evaluations.append(
            evaluate_proposal(
                run_id=BENCHMARK_RUN_ID,
                proposal_id=proposal.proposal_id,
                rubric=r,
                dimension_scores=scoring.dimension_scores,
                constraint_status=constraint_mod.combine(assessments),
                constraint_assessments=assessments,
                unresolved_assumptions=scoring.unresolved_assumptions,
            )
        )

    pairwise = run_pairwise(
        judge=judge,
        contract=case.contract,
        rubric=r,
        proposals=case.proposals,
        evidence=case.evidence,
    )
    stability = weight_sensitivity(r, evaluations)
    invariant, invariance_detail = relabel_invariance(r, evaluations)
    if not invariant:
        notes.append(invariance_detail)

    ranking = build_ranking(
        run_id=BENCHMARK_RUN_ID,
        rubric_version=r.version,
        evaluations=evaluations,
        top_k=max(top_k, 1),
        pairwise=pairwise,
        stability=stability,
    )
    produced = order_of(evaluations)
    winner = select_winner(ranking)

    # -- pairwise agreement on dominance pairs ---------------------------
    agreement: float | None = None
    decided = 0
    if case.dominance_pairs:
        correct = 0
        for better, worse in case.dominance_pairs:
            outcome = next(
                (
                    o
                    for o in pairwise.outcomes
                    if {o.proposal_x, o.proposal_y} == {better, worse}
                ),
                None,
            )
            if outcome is None or outcome.agreed_winner is None:
                if outcome is not None and not outcome.consistent:
                    notes.append(
                        f"judge contradicted itself on the dominance pair "
                        f"{better} vs {worse}; no preference signal"
                    )
                continue
            decided += 1
            if outcome.agreed_winner == better:
                correct += 1
            else:
                notes.append(
                    f"judge preferred {worse} over {better}, which dominates it -- "
                    "this is wrong under any non-negative weighting"
                )
        agreement = (correct / decided) if decided else None

    # -- rank correlation and Top-K recall -------------------------------
    correlation = (
        kendall_tau(case.reference_ranking, produced) if case.reference_ranking else None
    )
    recall: float | None = None
    if case.reference_ranking:
        expected_top = [
            p
            for p in case.reference_ranking[: max(top_k, 1)]
            if p not in case.expected_excluded
        ]
        if expected_top:
            recall = sum(1 for p in expected_top if p in ranking.top_k) / len(expected_top)

    # -- evidence grounding ----------------------------------------------
    real_ids = {e.evidence_id for e in case.evidence if e.is_real}
    scored = 0
    grounded = 0
    unknown = 0
    total = 0
    for ev in evaluations:
        for s in ev.dimension_scores:
            total += 1
            if s.status == "UNKNOWN":
                unknown += 1
                continue
            scored += 1
            if set(s.evidence_ids) & real_ids:
                grounded += 1
    grounding = grounded / scored if scored else 0.0
    unknown_rate = unknown / total if total else 0.0

    # -- constraint respect and goal fidelity ----------------------------
    excluded_ok = all(
        pid in ranking.excluded_for_constraints for pid in case.expected_excluded
    )
    if not excluded_ok:
        missed = [
            p for p in case.expected_excluded if p not in ranking.excluded_for_constraints
        ]
        notes.append(
            f"constraint-violating candidate(s) {missed} were not excluded -- a hard "
            "constraint was treated as compensable"
        )
    winner_ok = winner not in case.expected_excluded
    if not winner_ok:
        notes.append(f"a constraint-violating candidate ({winner}) was selected")

    fidelity = True
    if case.expected_winner is not None:
        fidelity = winner == case.expected_winner
        if not fidelity:
            notes.append(
                f"expected {case.expected_winner} to be selected (it dominates or is "
                f"the only compliant candidate) but got {winner}"
            )
    # Whatever is selected must at least be eligible.
    if winner is not None:
        selected_eval = next(e for e in evaluations if e.proposal_id == winner)
        if selected_eval.constraint_status is ConstraintStatus.FAIL:
            fidelity = False
            notes.append(f"selected candidate {winner} fails a hard constraint")

    return CaseResult(
        name=case.name,
        rationale=case.rationale,
        produced_ranking=produced,
        reference_ranking=case.reference_ranking,
        rank_correlation=correlation,
        pairwise_agreement=agreement,
        pairwise_decided=decided,
        pairwise_total=len(case.dominance_pairs),
        top_k_recall=recall,
        position_bias_rate=pairwise.position_bias_rate,
        order_consistency=pairwise.consistency_rate,
        relabel_invariant=invariant,
        top1_stable=stability.top1_stable,
        min_kendall_tau=stability.min_kendall_tau,
        evidence_grounding=round(grounding, 4),
        unknown_rate=round(unknown_rate, 4),
        constraint_respected=excluded_ok and winner_ok,
        goal_fidelity=fidelity,
        selected=winner,
        notes=notes,
    )


def run_benchmark(
    *, judge, cases: list[BenchmarkCase] | None = None, top_k: int = 1
) -> BenchmarkReport:
    """Run the benchmark and aggregate. ``passed`` is a hard gate.

    A run fails if the judge ever prefers a dominated candidate, lets a
    constraint violation through, breaks relabel invariance, or shows position
    bias. These are correctness failures, not matters of degree.
    """
    suite = cases if cases is not None else default_cases()
    results = [run_case(judge=judge, case=c, top_k=top_k) for c in suite]

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    agreements = [r.pairwise_agreement for r in results if r.pairwise_agreement is not None]
    correlations = [r.rank_correlation for r in results if r.rank_correlation is not None]
    recalls = [r.top_k_recall for r in results if r.top_k_recall is not None]

    failures: list[str] = []
    for r in results:
        if r.pairwise_agreement is not None and r.pairwise_agreement < 1.0:
            failures.append(
                f"[{r.name}] judge preferred a dominated candidate "
                f"(agreement {r.pairwise_agreement:.0%})"
            )
        if not r.constraint_respected:
            failures.append(f"[{r.name}] a hard constraint was not respected")
        if not r.goal_fidelity:
            failures.append(f"[{r.name}] the wrong candidate was selected")
        if not r.relabel_invariant:
            failures.append(f"[{r.name}] ranking changed under relabelling (engine bug)")
        if r.position_bias_rate > 0:
            failures.append(
                f"[{r.name}] position bias detected ({r.position_bias_rate:.0%} of pairs)"
            )

    return BenchmarkReport(
        judge_id=getattr(judge, "judge_id", "unknown"),
        cases=results,
        pairwise_agreement=mean(agreements),
        mean_rank_correlation=mean(correlations),
        top_k_recall=mean(recalls),
        position_bias_rate=round(
            sum(r.position_bias_rate for r in results) / len(results), 4
        )
        if results
        else 0.0,
        order_consistency=round(
            sum(r.order_consistency for r in results) / len(results), 4
        )
        if results
        else 1.0,
        evidence_grounding=round(
            sum(r.evidence_grounding for r in results) / len(results), 4
        )
        if results
        else 0.0,
        constraint_respect=round(
            sum(1 for r in results if r.constraint_respected) / len(results), 4
        )
        if results
        else 1.0,
        goal_fidelity=round(
            sum(1 for r in results if r.goal_fidelity) / len(results), 4
        )
        if results
        else 1.0,
        relabel_invariant=all(r.relabel_invariant for r in results),
        passed=not failures,
        failures=failures,
    )


def format_report(report: BenchmarkReport) -> str:
    """Human-readable benchmark summary."""

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0%}"

    lines = [
        f"Judge: {report.judge_id}",
        f"Result: {'PASS' if report.passed else 'FAIL'}",
        "",
        "Aggregate metrics",
        f"  pairwise agreement (dominance) : {pct(report.pairwise_agreement)}",
        f"  mean rank correlation (tau)    : "
        f"{'n/a' if report.mean_rank_correlation is None else report.mean_rank_correlation}",
        f"  top-K recall                   : {pct(report.top_k_recall)}",
        f"  position bias rate             : {pct(report.position_bias_rate)}",
        f"  order consistency              : {pct(report.order_consistency)}",
        f"  evidence grounding             : {pct(report.evidence_grounding)}",
        f"  constraint respect             : {pct(report.constraint_respect)}",
        f"  goal fidelity                  : {pct(report.goal_fidelity)}",
        f"  relabel invariant              : {report.relabel_invariant}",
        "",
        "Cases",
    ]
    for c in report.cases:
        lines += [
            f"  {c.name}",
            f"    why            : {c.rationale}",
            f"    produced order : {' > '.join(c.produced_ranking)}",
            f"    reference      : {' > '.join(c.reference_ranking) or '(none)'}",
            f"    selected       : {c.selected}",
            f"    agreement      : {pct(c.pairwise_agreement)} "
            f"({c.pairwise_decided}/{c.pairwise_total} pairs decided)",
            f"    tau / recall   : "
            f"{'n/a' if c.rank_correlation is None else c.rank_correlation} / "
            f"{pct(c.top_k_recall)}",
            f"    grounding      : {pct(c.evidence_grounding)} "
            f"(UNKNOWN rate {pct(c.unknown_rate)})",
            f"    stability      : top1_stable={c.top1_stable}, "
            f"min_tau={c.min_kendall_tau}, relabel_invariant={c.relabel_invariant}",
        ]
        for note in c.notes:
            lines.append(f"    note           : {note}")
        lines.append("")

    if report.failures:
        lines += ["Failures"] + [f"  - {f}" for f in report.failures] + [""]
    lines += ["Scope", f"  {report.scope_note}", ""]
    return "\n".join(lines)
