"""Next-best-action selection (spec section 9).

Scoring is not the goal. The point is to stop pouring effort into the wrong
candidate and to falsify big assumptions early. So this module turns a ranking
into an *action*: which experiment to run, which evidence gap to close, or which
human action to request.

Two properties the spec calls for explicitly:

* **Do not just test the leader.** The most informative experiment is often on
  the second-ranked candidate, or on an assumption shared by both. Candidates
  are drawn from across the eligible set, and the report says when the chosen
  experiment is not the top-ranked candidate's.
* **Predicted impact is labelled as predicted.** ``priority = impact / cost`` is
  an interpretable heuristic for ordering a shortlist, not a statistically
  optimal policy. Every record carries ``impact_is_predicted=True``.
"""

from __future__ import annotations

from ..core.schemas import (
    Base,
    ConstraintStatus,
    ExperimentCandidate,
    Proposal,
    ProposalEvaluation,
    Ranking,
)
from .decision_engine import rank_experiments


class HumanAction(Base):
    """Something only a person can do, with why it is needed."""

    label: str
    reason: str
    blocks_metric: str | None = None


class NextActionPlan(Base):
    """What to do next, and why — with the alternatives kept visible."""

    experiments: list[ExperimentCandidate] = []
    evidence_gaps: list[str] = []
    human_actions: list[HumanAction] = []
    chosen: ExperimentCandidate | None = None
    targets_top_ranked: bool | None = None
    rationale: str = ""
    caveat: str = (
        "Expected decision impact is a structured prediction, not a measured or "
        "calibrated quantity. It orders a shortlist; it does not replace research "
        "judgement, and `priority` is not a claim of statistical optimality."
    )


def _evidence_gaps(evaluations: list[ProposalEvaluation]) -> list[str]:
    """Named gaps, worst-evidenced candidates first."""
    gaps: list[str] = []
    for ev in sorted(evaluations, key=lambda e: e.evidence_sufficiency):
        unscored = [s.dimension_key for s in ev.dimension_scores if s.status == "UNKNOWN"]
        if unscored:
            gaps.append(
                f"{ev.proposal_id}: no usable evidence for "
                f"{', '.join(unscored)} ({ev.evidence_sufficiency:.0%} of rubric "
                "weight is backed by a score)."
            )
        for a in ev.constraint_assessments:
            if a.status is ConstraintStatus.UNKNOWN and a.unverified_assumption:
                gaps.append(
                    f"{ev.proposal_id}: hard constraint '{a.constraint_key}' is "
                    f"unverified — {a.unverified_assumption}"
                )
    return list(dict.fromkeys(gaps))


def _human_actions(
    proposals: list[Proposal], evaluations: list[ProposalEvaluation]
) -> list[HumanAction]:
    """Actions the harness cannot perform for itself."""
    out: list[HumanAction] = []
    by_id = {p.proposal_id: p for p in proposals}
    for ev in evaluations:
        proposal = by_id.get(ev.proposal_id)
        if proposal is None:
            continue
        for dep in proposal.dependencies:
            low = dep.lower()
            if any(
                k in low
                for k in ("device", "hardware", "user", "customer", "credential",
                          "access", "licence", "license", "approval")
            ):
                out.append(
                    HumanAction(
                        label=f"Obtain or confirm: {dep}",
                        reason=(
                            f"{ev.proposal_id} depends on it and the harness cannot "
                            "acquire it autonomously."
                        ),
                    )
                )
    return list({a.label: a for a in out}.values())


def select_next_action(
    *,
    proposals: list[Proposal],
    evaluations: list[ProposalEvaluation],
    ranking: Ranking,
) -> NextActionPlan:
    """Build the shortlist and pick the highest value-per-cost action."""
    candidates = rank_experiments(
        proposals=proposals, evaluations=evaluations, ranking=ranking
    )
    gaps = _evidence_gaps(evaluations)
    humans = _human_actions(proposals, evaluations)

    top_ranked = next(
        (
            e.proposal_id
            for e in sorted(ranking.entries, key=lambda x: x.rank)
            if e.eligible
        ),
        None,
    )

    if not candidates:
        rationale = (
            "No experiment would change the decision: no candidate has an "
            "unresolved assumption recorded."
        )
        if gaps:
            rationale += (
                f" {len(gaps)} evidence gap(s) remain and are reported instead."
            )
        if humans:
            rationale += (
                f" {len(humans)} action(s) require a person and are listed separately."
            )
        return NextActionPlan(
            experiments=[],
            evidence_gaps=gaps,
            human_actions=humans,
            chosen=None,
            targets_top_ranked=None,
            rationale=rationale,
        )

    chosen = candidates[0]
    targets_top = top_ranked in chosen.proposal_ids
    rationale = (
        f"Chose '{chosen.label}' — predicted decision impact "
        f"{chosen.predicted_decision_impact} over cost "
        f"{chosen.estimated_cost_units} gives the best priority "
        f"({chosen.priority}) of {len(candidates)} candidate action(s)."
    )
    if not targets_top:
        rationale += (
            f" Note: this does NOT test the top-ranked candidate ({top_ranked}). "
            "Resolving the contested assumption is worth more than confirming the "
            "current leader."
        )
    else:
        rationale += (
            f" It happens to target the top-ranked candidate ({top_ranked}); the "
            "shortlist was not restricted to it."
        )
    return NextActionPlan(
        experiments=candidates,
        evidence_gaps=gaps,
        human_actions=humans,
        chosen=chosen,
        targets_top_ranked=targets_top,
        rationale=rationale,
    )
