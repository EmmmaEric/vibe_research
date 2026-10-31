"""Judge adapters: the pluggable semantic-evaluation layer.

A judge supplies *semantic* judgements only -- per-dimension anchored scores with
justifications, per-constraint assessments, and pairwise preferences. It never
computes the aggregate score, never decides eligibility, and never states a
real-world success probability. Those are deterministic concerns owned by
:mod:`research_harness.evaluation.mcda`, :mod:`...constraints` and
:mod:`...decision_engine`.

Implementations:

* :class:`MockJudgeAdapter` -- offline, deterministic, reproducible. Scores are
  derived from declared proposal attributes by explicit heuristics, and every
  score is tagged ``score_source="simulated"``. Used by the default demo and all
  unit tests. Its output is a pipeline exercise, not research evidence.
* :class:`LLMJudgeAdapter` -- a general LLM via the provider abstraction,
  returning structured scores with checkable justifications.
* :class:`JevJudgeAdapter` -- deliberately NOT implemented against a guessed
  API. See the class docstring.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from pydantic import Field

from ..core.schemas import (
    Base,
    ConstraintAssessment,
    ConstraintStatus,
    DimensionScore,
    Evidence,
    HardConstraint,
    PairwiseJudgement,
    Proposal,
    ResearchContract,
    Rubric,
)


class RubricScoring(Base):
    """A judge's complete semantic verdict on one proposal."""

    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    constraint_assessments: list[ConstraintAssessment] = Field(default_factory=list)
    unresolved_assumptions: list[str] = Field(default_factory=list)
    recommended_next_test: str = ""
    judge_disagreement: str = "unknown"
    judge_id: str = "unknown"


@runtime_checkable
class JudgeAdapter(Protocol):
    """Interface every judge must satisfy."""

    judge_id: str

    def score_rubric(
        self,
        *,
        goal: ResearchContract,
        rubric: Rubric,
        proposal: Proposal,
        evidence: list[Evidence],
    ) -> RubricScoring: ...

    def compare_pair(
        self,
        *,
        goal: ResearchContract,
        rubric: Rubric,
        proposal_a: Proposal,
        proposal_b: Proposal,
        evidence: list[Evidence],
    ) -> PairwiseJudgement: ...


# --------------------------------------------------------------------------
# Deterministic offline judge
# --------------------------------------------------------------------------


def _clamp(v: float, lo: int = 0, hi: int = 5) -> int:
    """Round to the nearest anchor, halves upward, then clamp to ``[lo, hi]``.

    Deliberately not ``round()``: Python rounds halves to even, so ``round(4.5)``
    is 4 while ``round(3.5)`` is 4 as well. On a 0-5 anchored scale that makes
    the boundary between two anchors depend on the parity of the lower one,
    which is indefensible when the number is a judgement about quality. Half-up
    is what the anchor descriptions imply.
    """
    import math

    return int(max(lo, min(hi, math.floor(v + 0.5))))


def _mentions(haystack: list[str], needle: str) -> bool:
    low = needle.lower().strip()
    if not low:
        return False
    tokens = [t for t in low.replace("_", " ").split() if len(t) > 3]
    blob = " ".join(haystack).lower()
    return any(t in blob for t in tokens)


class MockJudgeAdapter:
    """Offline, reproducible judge driven by explicit heuristics.

    The heuristics read *declared* attributes of a proposal (how many
    dependencies it names, how many failure modes it admits, how much it says it
    costs, how much evidence it cites, whether its PoC states a falsification
    condition). This makes the resulting scores vary meaningfully between
    genuinely different proposals -- so the MCDA, pairwise and stability engines
    are exercised on real spread -- while remaining fully deterministic.

    It is NOT a research instrument. Everything it emits is tagged
    ``score_source="simulated"``.
    """

    judge_id = "mock-judge-v1(SIMULATED)"

    def __init__(self, *, always_prefer_first_slot: bool = False) -> None:
        # Test hook: forces the position-bias failure mode so the detector in
        # pairwise.py can be verified against a judge known to be biased.
        self.always_prefer_first_slot = always_prefer_first_slot

    # -- rubric ----------------------------------------------------------

    def score_rubric(
        self,
        *,
        goal: ResearchContract,
        rubric: Rubric,
        proposal: Proposal,
        evidence: list[Evidence],
    ) -> RubricScoring:
        ev_by_id = {e.evidence_id: e for e in evidence}
        cited = [ev_by_id[e] for e in proposal.evidence_ids if e in ev_by_id]
        real_cited = [e for e in cited if e.is_real]
        mean_quality = (
            sum(e.quality for e in real_cited) / len(real_cited) if real_cited else 0.0
        )

        scores: list[DimensionScore] = []

        for dim in rubric.dimensions:
            raw: int | None
            status = "SCORED"
            why = ""
            ev_ids = [e.evidence_id for e in cited]

            if dim.key == "goal_alignment":
                mapped = sum(
                    1
                    for m in goal.success_metrics
                    if _mentions([proposal.goal_mapping, proposal.core_mechanism], m.key)
                    or _mentions([proposal.goal_mapping], m.description)
                )
                total = max(len(goal.success_metrics), 1)
                raw = _clamp(2 + 3 * mapped / total)
                why = f"Maps {mapped}/{total} declared success metric(s) explicitly."

            elif dim.key == "feasibility":
                hard_deps = len(proposal.dependencies)
                unmet = sum(
                    1
                    for d in proposal.dependencies
                    if not _mentions(goal.available_resources, d)
                )
                raw = _clamp(5 - 0.8 * unmet - 0.2 * hard_deps)
                why = (
                    f"{hard_deps} dependency(ies), {unmet} of which are not covered by "
                    "declared available resources."
                )

            elif dim.key == "evidence_strength":
                if not real_cited:
                    raw, status = None, "UNKNOWN"
                    why = (
                        "No non-simulated evidence cited; refusing to assign a number "
                        "for absent evidence."
                    )
                else:
                    cross = sum(
                        1 for e in real_cited if e.verification_status == "cross_checked"
                    )
                    raw = _clamp(mean_quality * 0.7 + cross)
                    why = (
                        f"{len(real_cited)} real source(s), mean quality "
                        f"{mean_quality:.1f}/5, {cross} cross-checked."
                    )

            elif dim.key == "expected_benefit":
                refuting = sum(
                    1 for e in cited if e.stance.value == "refutes"
                )
                archetype_upside = {
                    "conservative": 2.5,
                    "balanced": 3.5,
                    "exploratory": 4.5,
                }.get(proposal.strategy_archetype, 3.0)
                raw = _clamp(archetype_upside - refuting)
                why = (
                    f"Archetype '{proposal.strategy_archetype}' upside prior "
                    f"{archetype_upside}, minus {refuting} refuting source(s)."
                )

            elif dim.key == "cost_time":
                # Favourability: cheaper is better. Banded on declared cost units.
                c = proposal.declared_cost_units
                raw = _clamp(5 - c)
                why = f"Declared cost {c} unit(s) (5 = negligible, 0 = prohibitive)."

            elif dim.key == "risk_robustness":
                modes = len(proposal.failure_modes)
                assumptions = len(proposal.critical_assumptions)
                raw = _clamp(5 - 0.6 * modes - 0.5 * assumptions)
                why = (
                    f"{modes} declared failure mode(s) and {assumptions} critical "
                    "unverified assumption(s)."
                )

            elif dim.key == "testability":
                has_falsifier = bool(proposal.poc.falsification_condition.strip())
                cheap = proposal.poc.estimated_cost_units <= 2
                raw = _clamp(2 + (2 if has_falsifier else 0) + (1 if cheap else 0))
                why = (
                    f"Falsification condition {'stated' if has_falsifier else 'absent'}; "
                    f"PoC cost {proposal.poc.estimated_cost_units} unit(s)."
                )

            elif dim.key == "novelty":
                raw = _clamp(
                    {
                        "conservative": 1.5,
                        "balanced": 3.0,
                        "exploratory": 4.5,
                    }.get(proposal.strategy_archetype, 2.5)
                    + 0.3 * len(proposal.differentiators)
                )
                why = (
                    f"Archetype '{proposal.strategy_archetype}' with "
                    f"{len(proposal.differentiators)} declared differentiator(s)."
                )

            else:
                raw, status = None, "UNKNOWN"
                why = f"No heuristic for custom dimension '{dim.key}'."

            scores.append(
                DimensionScore(
                    dimension_key=dim.key,
                    raw=raw,
                    status=status,  # type: ignore[arg-type]
                    justification=f"[SIMULATED] {why}",
                    evidence_ids=ev_ids if status == "SCORED" else [],
                    score_source="simulated",
                )
            )

        assessments = [
            self._assess_constraint(c, proposal) for c in goal.hard_constraints
        ]
        unresolved = list(proposal.critical_assumptions)

        next_test = (
            proposal.poc.description
            if proposal.critical_assumptions
            else "No unresolved assumption identified; proceed to goal acceptance test."
        )

        return RubricScoring(
            dimension_scores=scores,
            constraint_assessments=assessments,
            unresolved_assumptions=unresolved,
            recommended_next_test=next_test,
            judge_disagreement="unknown",
            judge_id=self.judge_id,
        )

    @staticmethod
    def _assess_constraint(
        c: HardConstraint, proposal: Proposal
    ) -> ConstraintAssessment:
        """Decide one constraint.

        Machine-checkable constraints (those with ``limit_value``) are decided by
        arithmetic. Everything else is PASS only when the proposal explicitly
        engages with the constraint, otherwise UNKNOWN -- never an implicit pass.
        """
        if c.limit_value is not None and c.limit_metric == "declared_cost_units":
            ok = c.check_value(proposal.declared_cost_units)
            return ConstraintAssessment(
                constraint_key=c.key,
                status=ConstraintStatus.PASS if ok else ConstraintStatus.FAIL,
                rationale=(
                    f"Declared cost {proposal.declared_cost_units} vs limit "
                    f"{c.limit_value} ({c.limit_direction.value}): "
                    f"{'within' if ok else 'EXCEEDS'} budget. Checked arithmetically."
                ),
            )

        engaged = _mentions(
            proposal.dependencies
            + proposal.premises
            + proposal.resource_needs
            + [proposal.core_mechanism, proposal.goal_mapping],
            c.description,
        )
        if engaged:
            return ConstraintAssessment(
                constraint_key=c.key,
                status=ConstraintStatus.PASS,
                rationale=(
                    "[SIMULATED] Proposal explicitly engages with this constraint. "
                    "Textual engagement is weak evidence of satisfaction."
                ),
            )
        return ConstraintAssessment(
            constraint_key=c.key,
            status=ConstraintStatus.UNKNOWN,
            rationale="[SIMULATED] Proposal does not address this constraint.",
            unverified_assumption=f"That the proposal satisfies '{c.key}': {c.description}",
        )

    # -- pairwise --------------------------------------------------------

    def compare_pair(
        self,
        *,
        goal: ResearchContract,
        rubric: Rubric,
        proposal_a: Proposal,
        proposal_b: Proposal,
        evidence: list[Evidence],
    ) -> PairwiseJudgement:
        if self.always_prefer_first_slot:
            return PairwiseJudgement(
                first_proposal_id=proposal_a.proposal_id,
                second_proposal_id=proposal_b.proposal_id,
                verdict="A",
                rationale="[SIMULATED] Deliberately position-biased judge (test hook).",
                judge_id=self.judge_id + "+biased",
            )

        qa = self._card_quality(goal, rubric, proposal_a, evidence)
        qb = self._card_quality(goal, rubric, proposal_b, evidence)

        # Abstain when neither candidate has any real evidence behind it.
        real_ids = {e.evidence_id for e in evidence if e.is_real}
        if not (set(proposal_a.evidence_ids) & real_ids) and not (
            set(proposal_b.evidence_ids) & real_ids
        ):
            verdict = "insufficient_evidence"
            rationale = (
                "[SIMULATED] Neither candidate cites non-simulated evidence; "
                "declining to express a preference."
            )
        elif abs(qa - qb) < 1e-6:
            verdict = "tie"
            rationale = f"[SIMULATED] Equal derived quality ({qa:.3f})."
        else:
            verdict = "A" if qa > qb else "B"
            rationale = (
                f"[SIMULATED] Derived quality {qa:.3f} vs {qb:.3f} from anchored "
                "dimension heuristics over the anonymised cards."
            )

        return PairwiseJudgement(
            first_proposal_id=proposal_a.proposal_id,
            second_proposal_id=proposal_b.proposal_id,
            verdict=verdict,  # type: ignore[arg-type]
            rationale=rationale,
            judge_id=self.judge_id,
        )

    def _card_quality(
        self,
        goal: ResearchContract,
        rubric: Rubric,
        proposal: Proposal,
        evidence: list[Evidence],
    ) -> float:
        """Content-derived quality proxy.

        Deliberately computed from the same rubric heuristics as
        :meth:`score_rubric` so the pairwise signal is *coherent* with the
        absolute scores rather than an independent random draw -- and, critically,
        is independent of slot position, which is what makes the position-bias
        detector informative.
        """
        scoring = self.score_rubric(
            goal=goal, rubric=rubric, proposal=proposal, evidence=evidence
        )
        weights = rubric.normalized_weights()
        total = 0.0
        used = 0.0
        for s in scoring.dimension_scores:
            if s.status == "SCORED" and s.raw is not None:
                w = weights[s.dimension_key]
                total += w * (s.raw / 5.0)
                used += w
        return total / used if used else 0.0


# --------------------------------------------------------------------------
# LLM judge
# --------------------------------------------------------------------------


class _LLMDimScore(Base):
    dimension_key: str
    raw: int | None = None
    unknown: bool = False
    justification: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class _LLMConstraint(Base):
    constraint_key: str
    status: str  # PASS | FAIL | UNKNOWN
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    unverified_assumption: str | None = None


class _LLMScoringResponse(Base):
    dimension_scores: list[_LLMDimScore] = Field(default_factory=list)
    constraint_assessments: list[_LLMConstraint] = Field(default_factory=list)
    unresolved_assumptions: list[str] = Field(default_factory=list)
    recommended_next_test: str = ""


class _LLMPairResponse(Base):
    verdict: str  # A | B | tie | insufficient_evidence
    rationale: str = ""


_SCORING_SYSTEM = """\
You are an Evaluation Judge in a research harness. You assign anchored rubric \
scores and assess hard constraints. You do NOT compute weighted totals, decide \
which proposal wins overall, or state any real-world probability of success -- \
deterministic code does that.

Rules you must follow:
- Score each dimension on its 0-5 anchors. Quote the anchor you matched.
- If the evidence needed for a dimension is absent, set unknown=true and leave \
raw null. Never invent a number to fill a gap.
- Cite evidence_ids you actually relied on. Never invent an id or a source.
- For hard constraints: PASS only with evidence of satisfaction, FAIL only with \
evidence of violation, otherwise UNKNOWN with the assumption that needs checking.
- A high score elsewhere never excuses a constraint violation.
"""

_PAIR_SYSTEM = """\
You are an Evaluation Judge comparing two anonymised research proposal cards \
against a goal and rubric. Reply with exactly one verdict:
  "A" -- card A is better for this goal
  "B" -- card B is better for this goal
  "tie" -- genuinely equivalent on the rubric
  "insufficient_evidence" -- you cannot tell from what is given

Judge only the substance against the rubric and the goal. Ignore length, \
polish, ordering and which card came first. You do not know who wrote either \
card; do not speculate. Prefer "insufficient_evidence" over a coin flip.
"""


class LLMJudgeAdapter:
    """Judge backed by a general LLM through the provider abstraction."""

    def __init__(self, provider, *, prompt_version: str = "v1") -> None:
        self.provider = provider
        self.prompt_version = prompt_version
        self.judge_id = f"llm-judge-{provider.provider_id}-{prompt_version}"

    def score_rubric(
        self,
        *,
        goal: ResearchContract,
        rubric: Rubric,
        proposal: Proposal,
        evidence: list[Evidence],
    ) -> RubricScoring:
        prompt = _render_scoring_prompt(goal, rubric, proposal, evidence)
        resp = self.provider.complete_structured(
            system=_SCORING_SYSTEM,
            prompt=prompt,
            schema=_LLMScoringResponse,
            purpose="judge.score_rubric",
        )

        valid_evidence = {e.evidence_id for e in evidence}
        scores: list[DimensionScore] = []
        returned = {d.dimension_key: d for d in resp.dimension_scores}
        for dim in rubric.dimensions:
            d = returned.get(dim.key)
            if d is None or d.unknown or d.raw is None:
                scores.append(
                    DimensionScore(
                        dimension_key=dim.key,
                        raw=None,
                        status="UNKNOWN",
                        justification=(
                            d.justification
                            if d
                            else "Judge returned no score for this dimension."
                        ),
                        score_source="llm_judge",
                    )
                )
            else:
                # Drop any evidence id the judge invented.
                cited = [e for e in d.evidence_ids if e in valid_evidence]
                scores.append(
                    DimensionScore(
                        dimension_key=dim.key,
                        raw=max(0, min(5, d.raw)),
                        status="SCORED",
                        justification=d.justification,
                        evidence_ids=cited,
                        score_source="llm_judge",
                    )
                )

        assessments: list[ConstraintAssessment] = []
        declared = {c.key for c in goal.hard_constraints}
        for c in resp.constraint_assessments:
            if c.constraint_key not in declared:
                continue
            try:
                status = ConstraintStatus(c.status.strip().upper())
            except ValueError:
                status = ConstraintStatus.UNKNOWN
            assumption = c.unverified_assumption
            if status is ConstraintStatus.UNKNOWN and not assumption:
                assumption = f"Unspecified; judge could not verify '{c.constraint_key}'."
            assessments.append(
                ConstraintAssessment(
                    constraint_key=c.constraint_key,
                    status=status,
                    rationale=c.rationale,
                    evidence_ids=[e for e in c.evidence_ids if e in valid_evidence],
                    unverified_assumption=assumption,
                )
            )

        return RubricScoring(
            dimension_scores=scores,
            constraint_assessments=assessments,
            unresolved_assumptions=resp.unresolved_assumptions,
            recommended_next_test=resp.recommended_next_test,
            judge_disagreement="unknown",
            judge_id=self.judge_id,
        )

    def compare_pair(
        self,
        *,
        goal: ResearchContract,
        rubric: Rubric,
        proposal_a: Proposal,
        proposal_b: Proposal,
        evidence: list[Evidence],
    ) -> PairwiseJudgement:
        prompt = (
            f"GOAL (original words of the requester):\n{goal.original_idea}\n\n"
            f"PROBLEM STATEMENT:\n{goal.problem_statement}\n\n"
            f"RUBRIC DIMENSIONS:\n"
            + "\n".join(f"- {d.name}: {d.definition}" for d in rubric.dimensions)
            + f"\n\n=== CARD A ===\n{proposal_a.card()}\n"
            f"\n=== CARD B ===\n{proposal_b.card()}\n"
        )
        resp = self.provider.complete_structured(
            system=_PAIR_SYSTEM,
            prompt=prompt,
            schema=_LLMPairResponse,
            purpose="judge.compare_pair",
        )
        verdict = resp.verdict.strip()
        if verdict not in ("A", "B", "tie", "insufficient_evidence"):
            verdict = "insufficient_evidence"
        return PairwiseJudgement(
            first_proposal_id=proposal_a.proposal_id,
            second_proposal_id=proposal_b.proposal_id,
            verdict=verdict,  # type: ignore[arg-type]
            rationale=resp.rationale,
            judge_id=self.judge_id,
            prompt_version=self.prompt_version,
        )


def _render_scoring_prompt(
    goal: ResearchContract,
    rubric: Rubric,
    proposal: Proposal,
    evidence: list[Evidence],
) -> str:
    dim_block = []
    for d in rubric.dimensions:
        anchors = "\n".join(f"      {lvl}: {txt}" for lvl, txt in sorted(d.anchors.items()))
        dim_block.append(
            f"- key: {d.key}\n  name: {d.name}\n  definition: {d.definition}\n"
            f"  boundary: {d.boundary_note}\n  anchors:\n{anchors}"
        )

    ev_block = (
        "\n".join(
            f"- {e.evidence_id} [{e.source_type}/{e.stance.value}/"
            f"{e.verification_status}/provenance={e.provenance.value}] {e.claim}"
            + (f" (limits: {'; '.join(e.limitations)})" if e.limitations else "")
            for e in evidence
        )
        or "(no evidence available)"
    )

    metrics = (
        "\n".join(
            f"- {m.key}: {m.description} [{m.unit}] target "
            f"{m.direction.value} {m.target_threshold}"
            for m in goal.success_metrics
        )
        or "(none declared)"
    )
    cons = (
        "\n".join(f"- {c.key} ({c.category}): {c.description}" for c in goal.hard_constraints)
        or "(none declared)"
    )

    return (
        f"ORIGINAL IDEA (verbatim, authoritative):\n{goal.original_idea}\n\n"
        f"PROBLEM STATEMENT:\n{goal.problem_statement}\n\n"
        f"SUCCESS METRICS:\n{metrics}\n\n"
        f"HARD CONSTRAINTS (non-compensatory):\n{cons}\n\n"
        f"AVAILABLE RESOURCES:\n{', '.join(goal.available_resources) or '(none declared)'}\n\n"
        f"RUBRIC:\n" + "\n".join(dim_block) + "\n\n"
        f"EVIDENCE POOL (cite only these ids):\n{ev_block}\n\n"
        f"PROPOSAL UNDER EVALUATION:\n{proposal.card(max_items=6)}\n"
    )


# --------------------------------------------------------------------------
# Jev / OpenJev
# --------------------------------------------------------------------------


class JevUnavailable(RuntimeError):
    pass


class JevJudgeAdapter:
    """Placeholder for a Jev / OpenJev backed judge -- intentionally inert.

    This adapter is **not implemented**, and that is a deliberate correctness
    decision rather than an omission: writing it would mean guessing at a
    third-party request/response shape, and a judge wired to an invented API
    would fail at run time or, worse, silently mis-parse scores.

    To implement it properly:

    1. Read the chosen repository's current README, published interface and
       licence. Confirm what it actually returns -- plausibly only a choice, a
       rubric score and/or a preference probability.
    2. Map exactly those outputs onto :class:`RubricScoring` /
       :class:`PairwiseJudgement`. Leave dimensions it cannot speak to as
       ``UNKNOWN``; do not synthesise values to fill the schema.
    3. Delegate justification and source-checking to the other agents -- a
       preference model is not a citation checker.
    4. Keep the fallback: the core pipeline must run with Jev absent.

    Note on interpretation: a preference probability from such a model describes
    that model's judgement distribution over the options it was shown. It is not
    the real-world probability that the proposal will succeed, and must not be
    reported as one without calibration against real outcome labels.
    """

    judge_id = "jev-judge(unavailable)"
    available = False

    def __init__(self, *_: object, **__: object) -> None:
        raise JevUnavailable(
            "JevJudgeAdapter is not implemented: its upstream API has not been "
            "verified from source in this environment, and guessing it would be "
            "worse than not having it. Use --judge mock or --judge llm. See the "
            "class docstring for the steps to implement it correctly."
        )

    def score_rubric(self, **_: object) -> RubricScoring:  # pragma: no cover
        raise JevUnavailable("not implemented")

    def compare_pair(self, **_: object) -> PairwiseJudgement:  # pragma: no cover
        raise JevUnavailable("not implemented")
