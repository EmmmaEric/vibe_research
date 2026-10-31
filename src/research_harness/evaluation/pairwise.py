"""Pairwise proposal competition with position-bias measurement.

Protocol for each unordered pair {X, Y}:

1. Judge sees (X as slot A, Y as slot B) -> forward verdict
2. Judge sees (Y as slot A, X as slot B) -> reverse verdict

Both calls use length-normalised, source-anonymised proposal *cards*
(:meth:`Proposal.card`) so the judge cannot score "which agent wrote it" or
reward verbosity. Verdicts name a *slot* (A/B/tie/insufficient_evidence); the
mapping back to proposal ids happens here, in deterministic code.

A pair is **consistent** when the two orders pick the same proposal (or both
abstain the same way). A pair exhibits **position bias** when the judge picked
slot A both times -- i.e. it preferred whatever came first, naming two different
proposals as winner.

Scoring: consistent winner takes 1.0; inconsistent or tied pairs split 0.5/0.5.
``insufficient_evidence`` awards nothing to anyone and is reported separately,
because "we cannot tell" is a legitimate outcome that should not be laundered
into a ranking signal.
"""

from __future__ import annotations

import itertools

from ..core.schemas import (
    PairOutcome,
    PairwiseJudgement,
    PairwiseReport,
    Proposal,
)
from .judge_adapter import JudgeAdapter


def _aggregate_pair(
    x: str, y: str, forward: PairwiseJudgement, reverse: PairwiseJudgement
) -> PairOutcome:
    """Fold two position-swapped judgements into one outcome.

    ``forward`` ran with x in slot A; ``reverse`` ran with y in slot A.
    """
    fwd_winner = forward.winner_id
    rev_winner = reverse.winner_id

    both_abstained = forward.verdict == reverse.verdict and fwd_winner is None
    consistent = (fwd_winner == rev_winner) or both_abstained

    # Position bias: each run picked its own slot A, naming different winners.
    favored_first = (
        forward.verdict == "A" and reverse.verdict == "A" and fwd_winner != rev_winner
    )

    if consistent and fwd_winner is not None:
        px, py = (1.0, 0.0) if fwd_winner == x else (0.0, 1.0)
        agreed = fwd_winner
    elif forward.verdict == "insufficient_evidence" and reverse.verdict == (
        "insufficient_evidence"
    ):
        px, py, agreed = 0.0, 0.0, None
    else:
        # Genuine tie, or the judge contradicted itself -> no information.
        px, py, agreed = 0.5, 0.5, None

    return PairOutcome(
        proposal_x=x,
        proposal_y=y,
        forward=forward,
        reverse=reverse,
        consistent=consistent,
        agreed_winner=agreed,
        favored_first_slot_both_times=favored_first,
        points_x=px,
        points_y=py,
    )


def run_pairwise(
    *,
    judge: JudgeAdapter,
    contract,
    rubric,
    proposals: list[Proposal],
    evidence: list,
    rubric_version: str | None = None,
) -> PairwiseReport:
    """Run the full A/B-swapped round robin over ``proposals``."""
    by_id = {p.proposal_id: p for p in proposals}
    ids = sorted(by_id)
    outcomes: list[PairOutcome] = []

    for x, y in itertools.combinations(ids, 2):
        forward = judge.compare_pair(
            goal=contract,
            rubric=rubric,
            proposal_a=by_id[x],
            proposal_b=by_id[y],
            evidence=evidence,
        )
        reverse = judge.compare_pair(
            goal=contract,
            rubric=rubric,
            proposal_a=by_id[y],
            proposal_b=by_id[x],
            evidence=evidence,
        )
        outcomes.append(_aggregate_pair(x, y, forward, reverse))

    points = {pid: 0.0 for pid in ids}
    for o in outcomes:
        points[o.proposal_x] += o.points_x
        points[o.proposal_y] += o.points_y

    n = len(outcomes)
    consistency = sum(1 for o in outcomes if o.consistent) / n if n else 0.0
    bias = sum(1 for o in outcomes if o.favored_first_slot_both_times) / n if n else 0.0
    insufficient = (
        sum(
            1
            for o in outcomes
            if o.forward.verdict == "insufficient_evidence"
            or o.reverse.verdict == "insufficient_evidence"
        )
        / n
        if n
        else 0.0
    )

    note_parts = [f"{n} pair(s), each judged twice with positions swapped."]
    if bias > 0:
        note_parts.append(
            f"Position bias detected in {bias:.0%} of pairs (judge preferred slot A "
            "regardless of content); treat the pairwise signal as weak."
        )
    if consistency < 1.0:
        note_parts.append(
            f"Order-consistency {consistency:.0%}: inconsistent pairs contributed no "
            "preference signal."
        )
    if insufficient > 0:
        note_parts.append(
            f"{insufficient:.0%} of pairs returned insufficient_evidence -- gather "
            "evidence rather than forcing a winner."
        )

    return PairwiseReport(
        run_id=contract.run_id,
        rubric_version=rubric_version or rubric.version,
        outcomes=outcomes,
        win_points=points,
        comparisons=n * 2,
        consistency_rate=round(consistency, 4),
        position_bias_rate=round(bias, 4),
        insufficient_evidence_rate=round(insufficient, 4),
        note=" ".join(note_parts),
    )
