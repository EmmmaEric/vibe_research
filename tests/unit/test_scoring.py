"""Scoring engine invariants.

These are the properties the whole harness rests on: the aggregate is a pure
function, missing evidence is never imputed, and a hard-constraint violation
cannot be bought off by a high score.
"""

from __future__ import annotations

import pytest

from research_harness.core.schemas import (
    ConstraintAssessment,
    ConstraintStatus,
    DimensionScore,
    HardConstraint,
    MetricDirection,
)
from research_harness.evaluation import constraints as cmod
from research_harness.evaluation.decision_engine import build_ranking, select_winner
from research_harness.evaluation.mcda import aggregate, evaluate_proposal, rescore
from research_harness.evaluation.rubric import build_rubric
from research_harness.evaluation.stability import (
    kendall_tau,
    order_of,
    relabel_invariance,
    weight_sensitivity,
)

from ..conftest import make_contract, passing, scores_for


@pytest.fixture
def r():
    return build_rubric(make_contract(), version="v1")


# --------------------------------------------------------------------------
# Normalisation and aggregation
# --------------------------------------------------------------------------


def test_all_fives_is_100_and_all_zeros_is_0(r):
    assert aggregate(r, scores_for(r, 5))[0] == pytest.approx(100.0)
    assert aggregate(r, scores_for(r, 0))[0] == pytest.approx(0.0)


def test_midpoint_scales_linearly(r):
    point, lo, hi, suff = aggregate(r, scores_for(r, 3))
    assert point == pytest.approx(60.0)
    assert lo == hi == pytest.approx(60.0)  # nothing unknown -> band collapses
    assert suff == pytest.approx(1.0)


def test_weights_are_normalised_so_scale_does_not_matter(r):
    doubled = r.with_weights({d.key: d.weight * 2 for d in r.dimensions}, "v1x2")
    assert aggregate(r, scores_for(r, 4))[0] == pytest.approx(
        aggregate(doubled, scores_for(doubled, 4))[0]
    )


def test_cost_dimension_is_favourability_not_raw_cost(r):
    """A high cost_time score must *help*, confirming the no-inversion design."""
    cheap = scores_for(r, 2)
    expensive = scores_for(r, 2)
    for s in cheap:
        if s.dimension_key == "cost_time":
            s.raw = 5
    for s in expensive:
        if s.dimension_key == "cost_time":
            s.raw = 0
    assert aggregate(r, cheap)[0] > aggregate(r, expensive)[0]


# --------------------------------------------------------------------------
# Missing evidence is bounded, never imputed
# --------------------------------------------------------------------------


def test_unknown_dimension_cannot_carry_a_number():
    with pytest.raises(ValueError, match="do not invent a number"):
        DimensionScore(dimension_key="x", raw=4, status="UNKNOWN")


def test_scored_dimension_requires_a_number():
    with pytest.raises(ValueError, match="requires a raw value"):
        DimensionScore(dimension_key="x", raw=None, status="SCORED")


def test_unknown_widens_the_band_and_lowers_sufficiency(r):
    full = aggregate(r, scores_for(r, 3))
    partial = aggregate(r, scores_for(r, 3, unknown=("evidence_strength",)))

    # Point estimate unchanged: remaining weights are renormalised, not padded.
    assert partial[0] == pytest.approx(full[0])
    # But the honest range widens in both directions.
    assert partial[1] < full[1]
    assert partial[2] > full[2]
    assert partial[3] < 1.0


def test_omitting_a_dimension_is_an_error_not_a_free_pass(r):
    partial = [s for s in scores_for(r, 4) if s.dimension_key != "feasibility"]
    with pytest.raises(ValueError, match="no score supplied"):
        aggregate(r, partial)


def test_unknown_dimension_key_is_rejected(r):
    extra = scores_for(r, 4) + [DimensionScore(dimension_key="invented", raw=5)]
    with pytest.raises(ValueError, match="unknown dimension"):
        aggregate(r, extra)


# --------------------------------------------------------------------------
# Hard constraints are non-compensatory  (MVP criterion)
# --------------------------------------------------------------------------


def test_constraint_fail_beats_a_perfect_score(r):
    """A candidate scoring 100 that violates a constraint must not be selected."""
    keys = [d.key for d in r.dimensions]
    perfect_but_illegal = evaluate_proposal(
        run_id="t",
        proposal_id="P_ILLEGAL",
        rubric=r,
        dimension_scores=scores_for(r, 5),
        constraint_status=ConstraintStatus.FAIL,
        constraint_assessments=[
            ConstraintAssessment(
                constraint_key="budget",
                status=ConstraintStatus.FAIL,
                rationale="over budget",
            )
        ],
    )
    mediocre_but_legal = evaluate_proposal(
        run_id="t",
        proposal_id="P_OK",
        rubric=r,
        dimension_scores=scores_for(r, 2),
        constraint_status=ConstraintStatus.PASS,
        constraint_assessments=passing(["budget"]),
    )

    assert perfect_but_illegal.point_score == pytest.approx(100.0)
    assert mediocre_but_legal.point_score == pytest.approx(40.0)
    assert perfect_but_illegal.eligible is False

    ranking = build_ranking(
        run_id="t",
        rubric_version="v1",
        evaluations=[perfect_but_illegal, mediocre_but_legal],
        top_k=2,
    )
    assert select_winner(ranking) == "P_OK"
    assert "P_ILLEGAL" not in ranking.top_k
    assert ranking.excluded_for_constraints == ["P_ILLEGAL"]
    # Ineligible candidates are ranked last regardless of score.
    assert order_of([perfect_but_illegal, mediocre_but_legal])[-1] == "P_ILLEGAL"


def test_no_winner_when_everything_violates(r):
    evals = [
        evaluate_proposal(
            run_id="t",
            proposal_id=pid,
            rubric=r,
            dimension_scores=scores_for(r, 5),
            constraint_status=ConstraintStatus.FAIL,
            constraint_assessments=[
                ConstraintAssessment(
                    constraint_key="safety", status=ConstraintStatus.FAIL
                )
            ],
        )
        for pid in ("PA", "PB")
    ]
    assert select_winner(build_ranking(
        run_id="t", rubric_version="v1", evaluations=evals
    )) is None


# --------------------------------------------------------------------------
# Constraint folding precedence
# --------------------------------------------------------------------------


def test_fail_dominates_unknown_dominates_pass():
    pass_ = ConstraintAssessment(constraint_key="a", status=ConstraintStatus.PASS)
    unk = ConstraintAssessment(
        constraint_key="b",
        status=ConstraintStatus.UNKNOWN,
        unverified_assumption="unchecked",
    )
    fail = ConstraintAssessment(constraint_key="c", status=ConstraintStatus.FAIL)

    assert cmod.combine([pass_]) is ConstraintStatus.PASS
    assert cmod.combine([pass_, unk]) is ConstraintStatus.UNKNOWN
    assert cmod.combine([pass_, unk, fail]) is ConstraintStatus.FAIL
    assert cmod.combine([]) is ConstraintStatus.PASS  # nothing declared


def test_unknown_must_name_the_assumption_to_verify():
    with pytest.raises(ValueError, match="must name the assumption"):
        ConstraintAssessment(constraint_key="a", status=ConstraintStatus.UNKNOWN)


def test_unassessed_constraint_becomes_unknown_not_pass():
    contract = make_contract(
        constraints=[
            HardConstraint(key="legal", description="must be lawful", category="legal")
        ]
    )
    completed = cmod.ensure_complete(contract, [])
    assert len(completed) == 1
    assert completed[0].status is ConstraintStatus.UNKNOWN
    assert completed[0].unverified_assumption


def test_assessments_for_undeclared_constraints_are_dropped():
    contract = make_contract(constraints=[])
    completed = cmod.ensure_complete(
        contract,
        [ConstraintAssessment(constraint_key="ghost", status=ConstraintStatus.PASS)],
    )
    assert completed == []


def test_machine_checkable_constraint_is_arithmetic():
    c = HardConstraint(
        key="budget",
        description="cost ceiling",
        category="budget",
        limit_metric="declared_cost_units",
        limit_value=4.0,
        limit_direction=MetricDirection.MINIMIZE,
    )
    assert c.check_value(3.9) is True
    assert c.check_value(4.0) is True
    assert c.check_value(4.1) is False


# --------------------------------------------------------------------------
# Re-ranking on weight change  (MVP criterion)
# --------------------------------------------------------------------------


def test_weight_change_reorders_and_preserves_the_old_version(r):
    """Candidates that differ per-dimension must reorder when weights change."""
    cheap_fragile = scores_for(r, 2)
    pricey_robust = scores_for(r, 2)
    for s in cheap_fragile:
        if s.dimension_key == "cost_time":
            s.raw = 5
    for s in pricey_robust:
        if s.dimension_key == "expected_benefit":
            s.raw = 5

    keys = ["budget"]
    a = evaluate_proposal(
        run_id="t", proposal_id="CHEAP", rubric=r, dimension_scores=cheap_fragile,
        constraint_status=ConstraintStatus.PASS, constraint_assessments=passing(keys),
    )
    b = evaluate_proposal(
        run_id="t", proposal_id="BENEFIT", rubric=r, dimension_scores=pricey_robust,
        constraint_status=ConstraintStatus.PASS, constraint_assessments=passing(keys),
    )

    cost_heavy = r.with_weights({"cost_time": 20.0}, "v2-cost")
    benefit_heavy = r.with_weights({"expected_benefit": 20.0}, "v2-benefit")

    assert order_of([rescore(cost_heavy, e) for e in (a, b)])[0] == "CHEAP"
    assert order_of([rescore(benefit_heavy, e) for e in (a, b)])[0] == "BENEFIT"

    # The original rubric object is untouched by with_weights.
    assert r.version == "v1"
    assert r.by_key("cost_time").weight != 20.0


def test_rescore_records_the_new_rubric_version(r):
    ev = evaluate_proposal(
        run_id="t", proposal_id="P", rubric=r, dimension_scores=scores_for(r, 3),
        constraint_status=ConstraintStatus.PASS, constraint_assessments=[],
    )
    v2 = r.with_weights({"cost_time": 5.0}, "v2")
    assert rescore(v2, ev).rubric_version == "v2"
    assert ev.rubric_version == "v1"  # original unchanged


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


def test_kendall_tau_bounds():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert kendall_tau(["a"], ["a"]) == 1.0


def test_ranking_is_invariant_under_relabelling(r):
    evals = []
    for i, pid in enumerate(("PA", "PB", "PC")):
        evals.append(
            evaluate_proposal(
                run_id="t", proposal_id=pid, rubric=r,
                dimension_scores=scores_for(r, 2 + i),
                constraint_status=ConstraintStatus.PASS, constraint_assessments=[],
            )
        )
    ok, detail = relabel_invariance(r, evals)
    assert ok, detail


def test_identical_candidates_are_stable_but_distinct_ones_are_detected(r):
    same = [
        evaluate_proposal(
            run_id="t", proposal_id=pid, rubric=r, dimension_scores=scores_for(r, 3),
            constraint_status=ConstraintStatus.PASS, constraint_assessments=[],
        )
        for pid in ("PA", "PB")
    ]
    report = weight_sensitivity(r, same)
    # Equal scores -> the id tiebreak keeps the order total and stable.
    assert report.top1_stable
    assert report.min_kendall_tau == 1.0
    assert report.findings  # every dimension x delta was actually probed
    assert len(report.findings) == len(r.dimensions) * 2


def test_sensitivity_detects_a_knife_edge(r):
    """A contested pair decided by one dimension must be reported as unstable."""
    a_scores = scores_for(r, 3)
    b_scores = scores_for(r, 3)
    for s in a_scores:
        if s.dimension_key == "cost_time":
            s.raw = 5
        if s.dimension_key == "expected_benefit":
            s.raw = 1
    for s in b_scores:
        if s.dimension_key == "cost_time":
            s.raw = 1
        if s.dimension_key == "expected_benefit":
            s.raw = 5

    evals = [
        evaluate_proposal(
            run_id="t", proposal_id=pid, rubric=r, dimension_scores=sc,
            constraint_status=ConstraintStatus.PASS, constraint_assessments=[],
        )
        for pid, sc in (("PA", a_scores), ("PB", b_scores))
    ]
    # Weight the two decisive dimensions almost equally so the lead is fragile.
    knife = r.with_weights({"cost_time": 10.0, "expected_benefit": 10.2}, "knife")
    evals = [rescore(knife, e) for e in evals]
    report = weight_sensitivity(knife, evals)
    assert not report.top1_stable
    assert "NOT stable" in report.interpretation
    assert set(report.unstable_dimensions) & {"cost_time", "expected_benefit"}
