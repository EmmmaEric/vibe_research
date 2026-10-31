"""Deterministic MCDA aggregation.

No LLM touches these numbers. Given dimension scores and a rubric, the weighted
score is a pure function -- which is what makes re-ranking on weight change, and
the stability analysis, meaningful.

Handling of missing evidence: a dimension marked ``UNKNOWN`` carries no number.
We do NOT impute one. Instead we report three figures:

* ``point_score``  -- weighted mean over SCORED dimensions, weights renormalised
* ``score_lower``  -- UNKNOWN dimensions counted as 0 (worst case)
* ``score_upper``  -- UNKNOWN dimensions counted as 5 (best case)

plus ``evidence_sufficiency``, the share of total weight that is actually backed
by a score. A wide ``[lower, upper]`` band is the honest signal that the
candidate is under-evidenced, and is preferred over a fabricated "confidence
penalty" multiplier.
"""

from __future__ import annotations

from ..core.schemas import (
    ConstraintStatus,
    DimensionScore,
    ProposalEvaluation,
    Rubric,
)

MAX_RAW = 5.0


def _normalise(raw: float, higher_is_better: bool) -> float:
    """Map a 0..5 anchored score to 0..1."""
    frac = raw / MAX_RAW
    return frac if higher_is_better else 1.0 - frac


def aggregate(
    rubric: Rubric,
    scores: list[DimensionScore],
) -> tuple[float, float, float, float]:
    """Return ``(point_score, score_lower, score_upper, evidence_sufficiency)``.

    Scores are on a 0..100 display scale. Raises if a rubric dimension has no
    corresponding score, so a silently dropped dimension cannot inflate a result.
    """
    weights = rubric.normalized_weights()
    by_key = {s.dimension_key: s for s in scores}

    missing = set(weights) - set(by_key)
    if missing:
        raise ValueError(
            f"no score supplied for rubric dimension(s) {sorted(missing)}; "
            "mark them UNKNOWN explicitly rather than omitting them"
        )
    extra = set(by_key) - set(weights)
    if extra:
        raise ValueError(f"score(s) for unknown dimension(s) {sorted(extra)}")

    scored_weight = 0.0
    weighted_sum = 0.0
    lower_sum = 0.0
    upper_sum = 0.0

    for key, w in weights.items():
        dim = rubric.by_key(key)
        s = by_key[key]
        if s.status == "SCORED":
            assert s.raw is not None  # guaranteed by DimensionScore validation
            norm = _normalise(float(s.raw), dim.higher_is_better)
            weighted_sum += w * norm
            lower_sum += w * norm
            upper_sum += w * norm
            scored_weight += w
        else:
            # UNKNOWN: excluded from the point estimate, widens the band.
            lower_sum += w * _normalise(0.0, dim.higher_is_better)
            upper_sum += w * _normalise(MAX_RAW, dim.higher_is_better)

    point = (weighted_sum / scored_weight * 100.0) if scored_weight > 0 else 0.0
    return (
        round(point, 4),
        round(lower_sum * 100.0, 4),
        round(upper_sum * 100.0, 4),
        round(scored_weight, 6),
    )


def evaluate_proposal(
    *,
    run_id: str,
    proposal_id: str,
    rubric: Rubric,
    dimension_scores: list[DimensionScore],
    constraint_status: ConstraintStatus,
    constraint_assessments: list,
    judge_disagreement: str = "unknown",
    unresolved_assumptions: list[str] | None = None,
    recommended_next_test: str = "",
) -> ProposalEvaluation:
    """Assemble a full evaluation. Pure function of its inputs."""
    point, lo, hi, suff = aggregate(rubric, dimension_scores)

    violated = [
        a.constraint_key
        for a in constraint_assessments
        if a.status is ConstraintStatus.FAIL
    ]
    unverified = [
        a.constraint_key
        for a in constraint_assessments
        if a.status is ConstraintStatus.UNKNOWN
    ]
    # Non-compensatory: a FAIL makes the candidate ineligible no matter the score.
    eligible = constraint_status is not ConstraintStatus.FAIL

    evidence_ids: list[str] = []
    for s in dimension_scores:
        for e in s.evidence_ids:
            if e not in evidence_ids:
                evidence_ids.append(e)
    for a in constraint_assessments:
        for e in a.evidence_ids:
            if e not in evidence_ids:
                evidence_ids.append(e)

    assumptions = list(unresolved_assumptions or [])
    for a in constraint_assessments:
        if a.status is ConstraintStatus.UNKNOWN and a.unverified_assumption:
            if a.unverified_assumption not in assumptions:
                assumptions.append(a.unverified_assumption)

    return ProposalEvaluation(
        proposal_id=proposal_id,
        run_id=run_id,
        rubric_version=rubric.version,
        constraint_status=constraint_status,
        constraint_assessments=list(constraint_assessments),
        violated_constraints=violated,
        unverified_constraints=unverified,
        dimension_scores=list(dimension_scores),
        weights=rubric.normalized_weights(),
        point_score=point,
        score_lower=lo,
        score_upper=hi,
        evidence_sufficiency=suff,
        evidence_ids=evidence_ids,
        judge_disagreement=judge_disagreement,
        unresolved_assumptions=assumptions,
        recommended_next_test=recommended_next_test,
        eligible=eligible,
    )


def rescore(rubric: Rubric, evaluation: ProposalEvaluation) -> ProposalEvaluation:
    """Recompute an existing evaluation against a different rubric version.

    Used when the user changes weights: the semantic dimension scores are reused
    (they do not depend on weights), only the aggregation changes. The returned
    object records the new ``rubric_version``.
    """
    return evaluate_proposal(
        run_id=evaluation.run_id,
        proposal_id=evaluation.proposal_id,
        rubric=rubric,
        dimension_scores=evaluation.dimension_scores,
        constraint_status=evaluation.constraint_status,
        constraint_assessments=evaluation.constraint_assessments,
        judge_disagreement=evaluation.judge_disagreement,
        unresolved_assumptions=evaluation.unresolved_assumptions,
        recommended_next_test=evaluation.recommended_next_test,
    )
