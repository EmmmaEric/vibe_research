"""Pluggable ranking strategies and Pareto analysis (spec sections 8.3, 13).

Two things the v2 spec adds over a single weighted sum:

* **A strategy protocol**, so the aggregation rule is replaceable without
  touching the orchestrator. :class:`WeightedMCDAStrategy` is the MVP default;
  AHP / TOPSIS / ELECTRE are deliberately *not* implemented — the spec says the
  MVP does not need them, and an unused ranking algorithm is dead weight. The
  protocol is the extension point.

* **Pareto non-dominance**, for goals where dimensions genuinely cannot trade
  off against each other. A weighted sum always permits a trade; when that is
  wrong, the honest answer is a *set* of incomparable candidates rather than a
  single winner. This reports that set; it does not silently replace the ranking.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.schemas import Base, ProposalEvaluation, Ranking, Rubric
from .decision_engine import build_ranking
from .stability import order_of, weight_sensitivity


@runtime_checkable
class RankingStrategy(Protocol):
    """How a set of scored candidates becomes an order."""

    strategy_id: str

    def rank(
        self,
        *,
        rubric: Rubric,
        evaluations: list[ProposalEvaluation],
        run_id: str,
        top_k: int,
    ) -> Ranking: ...


class WeightedMCDAStrategy:
    """Normalised weighted sum with constraint gating. The MVP default.

    Ranks on ``score_lower`` so a gap in the evidence can never improve a
    candidate's position (see ADR-014).
    """

    strategy_id = "weighted_mcda"

    def rank(
        self,
        *,
        rubric: Rubric,
        evaluations: list[ProposalEvaluation],
        run_id: str,
        top_k: int = 3,
    ) -> Ranking:
        return build_ranking(
            run_id=run_id,
            rubric_version=rubric.version,
            evaluations=evaluations,
            top_k=top_k,
            stability=weight_sensitivity(rubric, evaluations),
        )


# --------------------------------------------------------------------------
# Pareto
# --------------------------------------------------------------------------


class ParetoReport(Base):
    """Non-dominated candidates under no-trade-off assumptions."""

    frontier: list[str] = []
    dominated: dict[str, list[str]] = {}
    weighted_order: list[str] = []
    frontier_matches_top: bool = True
    interpretation: str = ""


def _comparable(a: ProposalEvaluation, b: ProposalEvaluation) -> list[str]:
    """Dimensions both candidates actually scored.

    An UNKNOWN dimension cannot participate in a dominance test: "we did not
    measure it" is not a value, and treating it as one would manufacture
    dominance out of missing evidence.
    """
    a_scored = {s.dimension_key for s in a.dimension_scores if s.status == "SCORED"}
    b_scored = {s.dimension_key for s in b.dimension_scores if s.status == "SCORED"}
    return sorted(a_scored & b_scored)


def dominates(a: ProposalEvaluation, b: ProposalEvaluation) -> bool:
    """True when ``a`` is >= ``b`` everywhere comparable and > somewhere.

    Requires at least two comparable dimensions: dominance established on a
    single shared dimension is not a multi-objective statement.
    """
    keys = _comparable(a, b)
    if len(keys) < 2:
        return False
    a_scores = {s.dimension_key: s.raw for s in a.dimension_scores}
    b_scores = {s.dimension_key: s.raw for s in b.dimension_scores}
    strictly_better = False
    for key in keys:
        av, bv = a_scores[key], b_scores[key]
        if av is None or bv is None:
            return False
        if av < bv:
            return False
        if av > bv:
            strictly_better = True
    return strictly_better


def pareto_frontier(evaluations: list[ProposalEvaluation]) -> ParetoReport:
    """Identify the non-dominated set among eligible candidates.

    Constraint-failing candidates are excluded first: a violation is not a
    trade-off to be weighed against other dimensions.
    """
    eligible = [e for e in evaluations if e.eligible]
    weighted = order_of(evaluations)

    if not eligible:
        return ParetoReport(
            frontier=[],
            weighted_order=weighted,
            interpretation=(
                "No eligible candidate: every one violates a hard constraint, so "
                "there is no frontier to report."
            ),
        )

    frontier: list[str] = []
    dominated: dict[str, list[str]] = {}
    for candidate in eligible:
        dominators = [
            other.proposal_id
            for other in eligible
            if other.proposal_id != candidate.proposal_id and dominates(other, candidate)
        ]
        if dominators:
            dominated[candidate.proposal_id] = sorted(dominators)
        else:
            frontier.append(candidate.proposal_id)

    frontier.sort(key=lambda pid: weighted.index(pid) if pid in weighted else 999)
    eligible_order = [p for p in weighted if p in {e.proposal_id for e in eligible}]
    matches_top = bool(frontier) and bool(eligible_order) and frontier[0] == eligible_order[0]

    if len(frontier) == 1:
        interpretation = (
            f"{frontier[0]} dominates every other eligible candidate on all "
            "comparable dimensions, so the choice does not depend on the weights."
        )
    elif len(frontier) == len(eligible):
        interpretation = (
            f"All {len(eligible)} eligible candidates are mutually non-dominated: "
            "each is better on something. Any single winner therefore reflects the "
            "chosen weights, not a dominance fact — if your dimensions genuinely "
            "cannot trade off, treat these as alternatives rather than a ranking."
        )
    else:
        interpretation = (
            f"{len(frontier)} of {len(eligible)} eligible candidates are "
            f"non-dominated ({', '.join(frontier)}); "
            f"{', '.join(sorted(dominated))} are dominated and can be dropped "
            "without choosing weights."
        )
    if not matches_top and frontier:
        interpretation += (
            f" The weighted ranking leads with {eligible_order[0]}, which is not "
            "first on the frontier — that difference is the weights doing the work."
        )

    return ParetoReport(
        frontier=frontier,
        dominated=dominated,
        weighted_order=weighted,
        frontier_matches_top=matches_top,
        interpretation=interpretation,
    )
