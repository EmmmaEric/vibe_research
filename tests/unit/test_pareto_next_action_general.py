"""Pareto analysis, next-best-action selection, and the general agent.

Covers spec sections 8.3 (non-dominance), 9 (active experiment selection) and
4.1 (the single-agent default that reports signals rather than decisions).
"""

from __future__ import annotations

import pytest

from research_harness.agents.general import GeneralResearchAgent, ready_tasks
from research_harness.core.schemas import (
    ConstraintStatus,
    MetricDirection,
    RouteAction,
    Stance,
    SuccessMetric,
    TaskNode,
)
from research_harness.evaluation.mcda import evaluate_proposal
from research_harness.evaluation.next_action import select_next_action
from research_harness.evaluation.ranking import (
    WeightedMCDAStrategy,
    dominates,
    pareto_frontier,
)
from research_harness.evaluation.rubric import build_rubric
from research_harness.evaluation.decision_engine import build_ranking

from ..conftest import make_contract, make_evidence, make_proposal, passing, scores_for


@pytest.fixture
def r():
    return build_rubric(make_contract(), version="v1")


def ev(r, pid, raw, *, eligible=True, unknown=(), overrides=None, assumptions=None):
    scores = scores_for(r, raw, unknown=unknown)
    for s in scores:
        if overrides and s.dimension_key in overrides:
            s.raw = overrides[s.dimension_key]
    return evaluate_proposal(
        run_id="t",
        proposal_id=pid,
        rubric=r,
        dimension_scores=scores,
        constraint_status=(
            ConstraintStatus.PASS if eligible else ConstraintStatus.FAIL
        ),
        constraint_assessments=passing(["budget"]) if eligible else [],
        unresolved_assumptions=assumptions or [],
    )


# --------------------------------------------------------------------------
# Pareto  (spec 8.3)
# --------------------------------------------------------------------------


def test_a_uniformly_better_candidate_dominates(r):
    better, worse = ev(r, "BETTER", 5), ev(r, "WORSE", 2)
    assert dominates(better, worse)
    assert not dominates(worse, better)


def test_equal_candidates_do_not_dominate_each_other(r):
    a, b = ev(r, "A", 3), ev(r, "B", 3)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_a_trade_off_is_not_dominance(r):
    cheap = ev(r, "CHEAP", 3, overrides={"cost_time": 5, "expected_benefit": 1})
    rich = ev(r, "RICH", 3, overrides={"cost_time": 1, "expected_benefit": 5})
    assert not dominates(cheap, rich)
    assert not dominates(rich, cheap)

    report = pareto_frontier([cheap, rich])
    assert set(report.frontier) == {"CHEAP", "RICH"}
    assert "mutually non-dominated" in report.interpretation
    assert "reflects the chosen weights" in report.interpretation


def test_unknown_dimensions_cannot_establish_dominance(r):
    """Missing evidence is not a value, so it must not manufacture dominance."""
    a = ev(r, "A", 5, unknown=tuple(d.key for d in r.dimensions[1:]))
    b = ev(r, "B", 2, unknown=tuple(d.key for d in r.dimensions[1:]))
    # Only one dimension is comparable, which is not a multi-objective claim.
    assert not dominates(a, b)


def test_dominated_candidates_can_be_dropped_without_choosing_weights(r):
    best = ev(r, "BEST", 5)
    mid = ev(r, "MID", 3)
    worst = ev(r, "WORST", 1)
    report = pareto_frontier([best, mid, worst])
    assert report.frontier == ["BEST"]
    assert set(report.dominated) == {"MID", "WORST"}
    assert "does not depend on the weights" in report.interpretation


def test_constraint_violators_are_excluded_from_the_frontier(r):
    good = ev(r, "GOOD", 2)
    illegal = ev(r, "ILLEGAL", 5, eligible=False)
    report = pareto_frontier([good, illegal])
    assert report.frontier == ["GOOD"]
    assert "ILLEGAL" not in report.dominated


def test_frontier_is_empty_when_nothing_is_eligible(r):
    report = pareto_frontier([ev(r, "A", 5, eligible=False)])
    assert report.frontier == []
    assert "no frontier" in report.interpretation


def test_weighted_strategy_satisfies_the_protocol(r):
    strategy = WeightedMCDAStrategy()
    ranking = strategy.rank(
        rubric=r, evaluations=[ev(r, "A", 5), ev(r, "B", 2)], run_id="t", top_k=2
    )
    assert strategy.strategy_id == "weighted_mcda"
    assert [e.proposal_id for e in sorted(ranking.entries, key=lambda x: x.rank)] == [
        "A", "B"
    ]
    assert ranking.stability is not None


# --------------------------------------------------------------------------
# Next best action  (spec 9)
# --------------------------------------------------------------------------


def _plan(r, proposals, evaluations):
    ranking = build_ranking(
        run_id="t", rubric_version=r.version, evaluations=evaluations, top_k=2
    )
    return select_next_action(
        proposals=proposals, evaluations=evaluations, ranking=ranking
    ), ranking


def test_priority_is_impact_over_cost_and_always_marked_predicted(r):
    proposals = [
        make_proposal("PA", assumptions=["cheap unknown"], poc_cost=1.0),
        make_proposal("PB", assumptions=["pricey unknown"], poc_cost=8.0),
    ]
    evaluations = [
        ev(r, "PA", 3, unknown=("evidence_strength",), assumptions=["cheap unknown"]),
        ev(r, "PB", 3, unknown=("evidence_strength",), assumptions=["pricey unknown"]),
    ]
    plan, _ = _plan(r, proposals, evaluations)

    assert plan.experiments
    assert all(c.impact_is_predicted for c in plan.experiments)
    for c in plan.experiments:
        assert c.priority == pytest.approx(
            c.predicted_decision_impact / c.estimated_cost_units, abs=5e-5
        )
    assert plan.experiments == sorted(
        plan.experiments, key=lambda c: (-c.priority, c.label)
    )
    assert "not a measured or calibrated quantity" in plan.caveat


def test_the_shortlist_is_not_restricted_to_the_leader(r):
    """Spec 9: do not only test the highest-scoring candidate."""
    proposals = [
        make_proposal("LEADER", assumptions=["expensive to check"], poc_cost=9.0),
        make_proposal("SECOND", assumptions=["cheap to check"], poc_cost=1.0),
    ]
    evaluations = [
        ev(r, "LEADER", 4, unknown=("evidence_strength",),
           assumptions=["expensive to check"]),
        ev(r, "SECOND", 3, unknown=("evidence_strength",),
           assumptions=["cheap to check"]),
    ]
    plan, ranking = _plan(r, proposals, evaluations)

    leader = next(
        e.proposal_id for e in sorted(ranking.entries, key=lambda x: x.rank) if e.eligible
    )
    assert leader == "LEADER"
    assert plan.chosen.proposal_ids == ["SECOND"]
    assert plan.targets_top_ranked is False
    assert "does NOT test the top-ranked candidate" in plan.rationale


def test_evidence_gaps_are_named_when_no_experiment_helps(r):
    proposals = [make_proposal("PA")]
    evaluations = [ev(r, "PA", 3, unknown=("evidence_strength",))]
    plan, _ = _plan(r, proposals, evaluations)

    assert plan.experiments == []
    assert plan.chosen is None
    assert plan.targets_top_ranked is None
    assert any("evidence_strength" in g for g in plan.evidence_gaps)
    assert "No experiment would change the decision" in plan.rationale


def test_unverified_hard_constraints_appear_as_evidence_gaps(r):
    from research_harness.core.schemas import ConstraintAssessment

    evaluation = evaluate_proposal(
        run_id="t",
        proposal_id="PA",
        rubric=r,
        dimension_scores=scores_for(r, 3),
        constraint_status=ConstraintStatus.UNKNOWN,
        constraint_assessments=[
            ConstraintAssessment(
                constraint_key="device",
                status=ConstraintStatus.UNKNOWN,
                unverified_assumption="that the device exists",
            )
        ],
    )
    plan, _ = _plan(r, [make_proposal("PA")], [evaluation])
    assert any("device" in g and "unverified" in g for g in plan.evidence_gaps)


def test_human_actions_are_separated_from_experiments(r):
    proposals = [
        make_proposal(
            "PA",
            assumptions=["needs the rig"],
            steps=["measure"],
        ).model_copy(update={"dependencies": ["a physical device", "customer access"]})
    ]
    evaluations = [ev(r, "PA", 3, assumptions=["needs the rig"])]
    plan, _ = _plan(r, proposals, evaluations)

    labels = " ".join(a.label for a in plan.human_actions)
    assert "physical device" in labels
    assert "customer access" in labels
    for action in plan.human_actions:
        assert action.reason


# --------------------------------------------------------------------------
# General agent  (spec 4.1)
# --------------------------------------------------------------------------


def test_general_agent_reports_signals_and_never_an_order(provider):
    agent = GeneralResearchAgent(provider)
    report = agent.solve(
        contract=make_contract(),
        evidence=[make_evidence("E001"), make_evidence("E002", stance=Stance.REFUTES)],
    )
    assert report.findings
    assert report.hypotheses
    assert report.candidate_directions
    assert report.task_graph
    assert 0.0 <= report.complexity_score <= 1.0
    # Its action preference is advisory only.
    assert report.suggested_next_action in set(RouteAction) | {None}
    assert "[SIMULATED]" in report.findings


def test_general_agent_surfaces_a_blocking_unknown(provider):
    """A required metric with no threshold makes the goal undecidable."""
    contract = make_contract(
        metrics=[
            SuccessMetric(
                key="vibes", description="unmeasurable", unit="?",
                target_threshold=None, direction=MetricDirection.MAXIMIZE,
            )
        ]
    )
    report = GeneralResearchAgent(provider).solve(
        contract=contract, evidence=[make_evidence("E001")]
    )
    assert any("vibes" in u and "no numeric target" in u for u in report.critical_unknowns)
    assert report.information_sufficient is False


def test_general_agent_reports_no_evidence_as_a_critical_unknown(provider):
    report = GeneralResearchAgent(provider).solve(
        contract=make_contract(), evidence=[]
    )
    assert any("No non-simulated evidence" in u for u in report.critical_unknowns)


def test_absent_counter_evidence_is_flagged_not_treated_as_support(provider):
    report = GeneralResearchAgent(provider).solve(
        contract=make_contract(), evidence=[make_evidence("E001")]
    )
    joined = " ".join(report.counterexamples)
    assert "No disconfirming evidence was found" in joined
    assert "search is too narrow" in joined


def test_general_agent_discards_evidence_ids_it_did_not_have(provider):
    report = GeneralResearchAgent(provider).solve(
        contract=make_contract(), evidence=[make_evidence("E001")]
    )
    assert set(report.evidence_ids) <= {"E001"}


def test_task_graph_integration_node_depends_on_the_rest(provider):
    report = GeneralResearchAgent(provider).solve(
        contract=make_contract(), evidence=[make_evidence("E001")]
    )
    integration = next(t for t in report.task_graph if t.task_id == "TI")
    others = {t.task_id for t in report.task_graph} - {"TI"}
    assert set(integration.depends_on) == others, (
        "findings must be integrated against the one original goal"
    )


def test_ready_tasks_is_the_independent_frontier():
    graph = [
        TaskNode(task_id="A", question="a", status="done"),
        TaskNode(task_id="B", question="b"),
        TaskNode(task_id="C", question="c", depends_on=["A"]),
        TaskNode(task_id="D", question="d", depends_on=["B"]),
    ]
    ready = {t.task_id for t in ready_tasks(graph)}
    assert ready == {"B", "C"}, "only genuinely unblocked tasks may run in parallel"
