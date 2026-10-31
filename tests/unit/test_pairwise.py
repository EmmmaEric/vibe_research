"""Pairwise competition: position-bias detection and abstention handling."""

from __future__ import annotations

from research_harness.core.schemas import PairwiseJudgement, Stance
from research_harness.evaluation.judge_adapter import MockJudgeAdapter
from research_harness.evaluation.pairwise import _aggregate_pair, run_pairwise
from research_harness.evaluation.rubric import build_rubric

from ..conftest import make_contract, make_evidence, make_proposal


def _j(first: str, second: str, verdict: str) -> PairwiseJudgement:
    return PairwiseJudgement(
        first_proposal_id=first, second_proposal_id=second, verdict=verdict
    )


# --------------------------------------------------------------------------
# Pair folding
# --------------------------------------------------------------------------


def test_consistent_winner_takes_the_full_point():
    # X wins from slot A, and still wins from slot B -> genuinely preferred.
    out = _aggregate_pair("X", "Y", _j("X", "Y", "A"), _j("Y", "X", "B"))
    assert out.consistent
    assert out.agreed_winner == "X"
    assert (out.points_x, out.points_y) == (1.0, 0.0)
    assert not out.favored_first_slot_both_times


def test_judge_that_always_picks_slot_a_is_flagged_as_biased():
    out = _aggregate_pair("X", "Y", _j("X", "Y", "A"), _j("Y", "X", "A"))
    assert not out.consistent
    assert out.agreed_winner is None
    assert out.favored_first_slot_both_times
    # Contradiction carries no preference signal.
    assert (out.points_x, out.points_y) == (0.5, 0.5)


def test_tie_splits_and_is_not_bias():
    out = _aggregate_pair("X", "Y", _j("X", "Y", "tie"), _j("Y", "X", "tie"))
    assert out.consistent
    assert out.agreed_winner is None
    assert not out.favored_first_slot_both_times
    assert (out.points_x, out.points_y) == (0.5, 0.5)


def test_mutual_abstention_awards_nothing_to_anyone():
    out = _aggregate_pair(
        "X", "Y",
        _j("X", "Y", "insufficient_evidence"),
        _j("Y", "X", "insufficient_evidence"),
    )
    assert out.consistent
    assert (out.points_x, out.points_y) == (0.0, 0.0)


# --------------------------------------------------------------------------
# Full round robin
# --------------------------------------------------------------------------


def test_every_pair_is_judged_twice_with_swapped_positions():
    contract = make_contract()
    rubric = build_rubric(contract)
    ev = [make_evidence("E001")]
    props = [
        make_proposal("PA", cost=1.0, evidence_ids=["E001"]),
        make_proposal("PB", cost=3.0, evidence_ids=["E001"]),
        make_proposal("PC", cost=5.0, evidence_ids=["E001"]),
    ]
    report = run_pairwise(
        judge=MockJudgeAdapter(), contract=contract, rubric=rubric,
        proposals=props, evidence=ev,
    )
    assert len(report.outcomes) == 3  # 3 choose 2
    assert report.comparisons == 6  # each judged both ways
    for o in report.outcomes:
        assert o.forward.first_proposal_id == o.proposal_x
        assert o.reverse.first_proposal_id == o.proposal_y


def test_unbiased_judge_is_order_consistent_with_no_bias():
    contract = make_contract()
    rubric = build_rubric(contract)
    ev = [make_evidence("E001")]
    props = [
        make_proposal("PA", cost=1.0, evidence_ids=["E001"]),
        make_proposal("PB", cost=4.0, evidence_ids=["E001"]),
    ]
    report = run_pairwise(
        judge=MockJudgeAdapter(), contract=contract, rubric=rubric,
        proposals=props, evidence=ev,
    )
    assert report.consistency_rate == 1.0
    assert report.position_bias_rate == 0.0


def test_biased_judge_is_caught_and_reported():
    """The detector must catch a judge known to favour whatever comes first."""
    contract = make_contract()
    rubric = build_rubric(contract)
    ev = [make_evidence("E001")]
    props = [
        make_proposal("PA", cost=1.0, evidence_ids=["E001"]),
        make_proposal("PB", cost=4.0, evidence_ids=["E001"]),
        make_proposal("PC", cost=2.0, evidence_ids=["E001"]),
    ]
    report = run_pairwise(
        judge=MockJudgeAdapter(always_prefer_first_slot=True),
        contract=contract, rubric=rubric, proposals=props, evidence=ev,
    )
    assert report.position_bias_rate == 1.0
    assert report.consistency_rate == 0.0
    assert "Position bias detected" in report.note
    # A biased judge produces a dead heat rather than a spurious winner.
    assert len(set(report.win_points.values())) == 1


def test_judge_abstains_when_no_candidate_cites_real_evidence():
    """"We cannot tell" must stay available rather than forcing a coin flip."""
    contract = make_contract()
    rubric = build_rubric(contract)
    sim_only = [make_evidence("E001", real=False)]
    props = [
        make_proposal("PA", cost=1.0, evidence_ids=["E001"]),
        make_proposal("PB", cost=4.0, evidence_ids=["E001"]),
    ]
    report = run_pairwise(
        judge=MockJudgeAdapter(), contract=contract, rubric=rubric,
        proposals=props, evidence=sim_only,
    )
    assert report.insufficient_evidence_rate == 1.0
    assert all(v == 0.0 for v in report.win_points.values())
    assert "insufficient_evidence" in report.note


# --------------------------------------------------------------------------
# Card anonymisation
# --------------------------------------------------------------------------


def test_proposal_card_hides_identity_and_authorship():
    p = make_proposal("PSECRET")
    p = p.model_copy(update={"generated_by": "designer-x"})
    card = p.card()
    assert "PSECRET" not in card
    assert "designer-x" not in card
    # But the substance a judge needs is present.
    assert "Mechanism:" in card and "Critical assumptions:" in card


def test_card_length_is_bounded_so_verbosity_cannot_win():
    verbose = make_proposal(
        "PA", steps=[f"step {i}" for i in range(40)],
        failure_modes=[f"mode {i}" for i in range(40)],
    )
    terse = make_proposal("PB", steps=["one"], failure_modes=["one"])
    # Truncation to max_items keeps the two cards comparable in size.
    assert len(verbose.card()) < 4 * len(terse.card())
    assert "step 30" not in verbose.card()
