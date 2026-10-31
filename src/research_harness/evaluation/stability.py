"""Ranking stability analysis.

Two deterministic checks the MVP must provide:

* **Weight sensitivity** -- perturb each dimension's weight by +/-delta,
  renormalise, re-rank, and measure how far the ranking moved (Kendall tau) and
  whether the top choice flipped. A top-1 that flips under a 20% weight nudge is
  not a robust recommendation, and the report says so.

* **Relabel invariance** -- re-run aggregation with proposal ids renamed and the
  input order shuffled. Because aggregation is a pure function of scores and
  weights, the ranking must be identical; any difference is a bug in the engine,
  not a property of the proposals.
"""

from __future__ import annotations

from ..core.schemas import (
    ProposalEvaluation,
    Rubric,
    SensitivityFinding,
    StabilityReport,
)
from .mcda import rescore

DEFAULT_DELTAS = (-0.2, 0.2)


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall rank correlation between two orderings of the same items.

    Returns 1.0 for identical orderings, -1.0 for exact reversal. Items present
    in only one list are ignored.
    """
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return 1.0
    rank_b = {item: i for i, item in enumerate(b)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]
            # a says x before y; does b agree?
            if rank_b[x] < rank_b[y]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return round((concordant - discordant) / total, 4) if total else 1.0


def order_of(evaluations: list[ProposalEvaluation]) -> list[str]:
    """Deterministic ranking order.

    Key, in order: eligible first, then ``score_lower`` descending, then
    ``point_score`` descending, then ``proposal_id``.

    **Why ``score_lower`` and not ``point_score``.** The point estimate
    renormalises weights over the dimensions that were actually scored, which
    makes it incomparable *between* candidates with different evidence coverage:
    a candidate with no evidence at all has that dimension dropped, and if the
    scored value would have been below its own average, dropping it *raises* its
    point score. That rewards having no evidence, which is perverse -- and it
    was a live bug here, caught by the ``evidence_grounding`` benchmark case.

    ``score_lower`` bounds every UNKNOWN dimension at its worst value, so a gap
    in the evidence can never improve a candidate's standing. It is also
    order-equivalent to ``point_score`` whenever two candidates have the same
    dimensions scored, so this costs nothing in the common case. Under-evidenced
    candidates are ranked low but *retained*, which is what the spec asks for:
    keep them, and prioritise verifying them.

    ``point_score`` remains the headline number in reports, alongside the band
    and ``evidence_sufficiency``; it is just not the comparison key.

    The ``proposal_id`` tiebreak keeps the order total, so a stability result is
    never an artefact of dict iteration order.
    """
    return [
        e.proposal_id
        for e in sorted(
            evaluations,
            key=lambda e: (not e.eligible, -e.score_lower, -e.point_score, e.proposal_id),
        )
    ]


def weight_sensitivity(
    rubric: Rubric,
    evaluations: list[ProposalEvaluation],
    *,
    deltas: tuple[float, ...] = DEFAULT_DELTAS,
) -> StabilityReport:
    """Perturb each weight in turn and report ranking movement."""
    base_order = order_of(evaluations)
    base_top = base_order[0] if base_order else None
    findings: list[SensitivityFinding] = []

    for dim in rubric.dimensions:
        for delta in deltas:
            new_weight = max(dim.weight * (1.0 + delta), 0.0)
            perturbed = rubric.with_weights(
                {dim.key: new_weight},
                new_version=f"{rubric.version}+sens({dim.key}{delta:+.0%})",
            )
            re_evaluated = [rescore(perturbed, e) for e in evaluations]
            new_order = order_of(re_evaluated)
            new_top = new_order[0] if new_order else None
            findings.append(
                SensitivityFinding(
                    dimension_key=dim.key,
                    delta=delta,
                    new_top_proposal=new_top,
                    kendall_tau_vs_base=kendall_tau(base_order, new_order),
                    top1_changed=(new_top != base_top),
                )
            )

    unstable = sorted({f.dimension_key for f in findings if f.top1_changed})
    min_tau = min((f.kendall_tau_vs_base for f in findings), default=1.0)
    top1_stable = not unstable

    if top1_stable and min_tau == 1.0:
        interp = (
            "Ranking is insensitive to +/-20% weight changes on any single dimension."
        )
    elif top1_stable:
        interp = (
            f"Top choice held under all +/-20% single-weight perturbations, but lower "
            f"ranks moved (min Kendall tau {min_tau}). Differences between mid-ranked "
            "candidates are not meaningful at this evidence level."
        )
    else:
        interp = (
            f"Top choice is NOT stable: a +/-20% change to {', '.join(unstable)} flips "
            "it. Prefer gathering evidence on the contested dimension(s) over "
            "committing to a single winner."
        )

    return StabilityReport(
        run_id=evaluations[0].run_id if evaluations else "",
        rubric_version=rubric.version,
        base_ranking=base_order,
        findings=findings,
        top1_stable=top1_stable,
        min_kendall_tau=min_tau,
        unstable_dimensions=unstable,
        interpretation=interp,
    )


def relabel_invariance(
    rubric: Rubric, evaluations: list[ProposalEvaluation]
) -> tuple[bool, str]:
    """Check the engine ignores identity and input order.

    Renames every proposal, reverses the input list, re-aggregates, then maps the
    resulting order back through the renaming. Returns ``(ok, detail)``.
    """
    base_order = order_of(evaluations)
    alias = {e.proposal_id: f"__alias_{i}" for i, e in enumerate(evaluations)}
    inverse = {v: k for k, v in alias.items()}

    disguised = []
    for e in reversed(evaluations):
        data = e.model_dump()
        data["proposal_id"] = alias[e.proposal_id]
        disguised.append(ProposalEvaluation.model_validate(data))

    re_evaluated = [rescore(rubric, e) for e in disguised]
    recovered = [inverse[pid] for pid in order_of(re_evaluated)]

    if recovered == base_order:
        return True, "Ranking unchanged under proposal relabelling and input reordering."
    return False, (
        f"Ranking changed under relabelling: base={base_order} "
        f"vs relabelled={recovered}. This indicates an engine bug "
        "(aggregation must be a pure function of scores and weights)."
    )
