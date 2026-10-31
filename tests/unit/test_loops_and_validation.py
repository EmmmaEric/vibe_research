"""Validation levels, the two loops, and the stop policy.

The central honesty property under test: **no amount of simulated or unexecuted
work can be reported as goal attainment.** Also covered: Loop A and Loop B read
disjoint test categories, and every loop terminates for a stated reason.
"""

from __future__ import annotations

import pytest

from research_harness.agents.diagnosis import Action, DiagnosisAgent
from research_harness.core.schemas import (
    Budget,
    ConstraintStatus,
    ExperimentResult,
    MetricDirection,
    MetricObservation,
    Provenance,
    SuccessMetric,
    TestCase,
    TestOutcome,
    TestPlan,
    MaturityLevel,
    ValidationResult,
)
from research_harness.evaluation.decision_engine import (
    build_ranking,
    build_validation_result,
    determine_maturity_level,
    rank_experiments,
    should_stop,
    verdicts_for,
)
from research_harness.evaluation.mcda import evaluate_proposal
from research_harness.evaluation.rubric import build_rubric
from research_harness.experiments.validation import (
    CONFORMANCE_CATEGORIES,
    GOAL_CATEGORIES,
    check_implementation_conformance,
    evidence_quality_gaps,
)

from ..conftest import make_contract, make_proposal, passing, scores_for

SPEED = SuccessMetric(
    key="speed", description="speedup", unit="x",
    target_threshold=2.0, direction=MetricDirection.MAXIMIZE,
)


def _plan() -> TestPlan:
    return TestPlan(
        plan_id="p1", run_id="t", proposal_id="PA",
        cases=[
            TestCase(test_id="T001", category="contract", description="d",
                     procedure="p", acceptance_criterion="a", executable=True),
            TestCase(test_id="T002", category="correctness", description="d",
                     procedure="p", acceptance_criterion="a", executable=True),
            TestCase(test_id="T003", category="goal_acceptance", description="d",
                     procedure="p", acceptance_criterion="speed >= 2.0",
                     measures_metric="speed", executable=True),
        ],
    )


def _experiment(
    *,
    simulated: bool = False,
    executed: bool = True,
    human: bool = False,
    outcomes: list[TestOutcome] | None = None,
    observations: list[MetricObservation] | None = None,
) -> ExperimentResult:
    plan = _plan()
    return ExperimentResult(
        experiment_id="e1", run_id="t", proposal_id="PA", plan_id=plan.plan_id,
        plan_hash=plan.content_hash(), adapter_id="x",
        is_simulated=simulated, executed=executed, requires_human_input=human,
        outcomes=outcomes or [], observations=observations or [],
    )


def _measured(value: float, prov: Provenance = Provenance.MEASURED) -> MetricObservation:
    return MetricObservation(
        metric_key="speed", value=value, unit="x", provenance=prov,
        n_repeats=5, stddev=0.1, seed=1,
    )


# --------------------------------------------------------------------------
# The simulated ceiling
# --------------------------------------------------------------------------


def test_simulated_experiment_cannot_exceed_l3_even_with_perfect_numbers():
    contract = make_contract(metrics=[SPEED])
    exp = _experiment(
        simulated=True,
        outcomes=[TestOutcome(test_id=t, passed=True) for t in ("T001", "T002", "T003")],
        observations=[_measured(999.0, Provenance.SIMULATED)],
    )
    level, reason = determine_maturity_level(
        contract=contract, experiment=exp,
        verdicts=verdicts_for(contract, exp),
        implementation_conforms=True,
        constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    assert level is MaturityLevel.R3_PROTOTYPED
    assert "SIMULATED" in reason


def test_goal_satisfied_is_never_true_from_simulated_data():
    contract = make_contract(metrics=[SPEED])
    exp = _experiment(
        simulated=True,
        outcomes=[TestOutcome(test_id="T003", passed=True)],
        observations=[_measured(500.0, Provenance.SIMULATED)],
    )
    result = build_validation_result(
        contract=contract, proposal_id="PA", experiment=exp,
        implementation_conforms=True, implementation_notes="",
        failed_test_ids=[], constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    assert result.goal_satisfied is not True
    assert result.maturity_level is MaturityLevel.R3_PROTOTYPED
    assert any("SIMULATED" in g for g in result.evidence_gaps)


def test_real_measurement_meeting_the_threshold_reaches_l5():
    contract = make_contract(metrics=[SPEED])
    exp = _experiment(observations=[_measured(3.0)])
    result = build_validation_result(
        contract=contract, proposal_id="PA", experiment=exp,
        implementation_conforms=True, implementation_notes="",
        failed_test_ids=[], constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    assert result.maturity_level is MaturityLevel.R5_GOAL_ACHIEVED
    assert result.goal_satisfied is True


def test_real_measurement_missing_the_threshold_is_a_research_result_not_success():
    contract = make_contract(metrics=[SPEED])
    exp = _experiment(observations=[_measured(1.1)])
    result = build_validation_result(
        contract=contract, proposal_id="PA", experiment=exp,
        implementation_conforms=True, implementation_notes="",
        failed_test_ids=[], constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    assert result.goal_satisfied is False
    assert result.maturity_level is MaturityLevel.R4_VALIDATED
    assert "did not reach" in result.blocking_reason


def test_unverified_hard_constraint_blocks_l5():
    contract = make_contract(metrics=[SPEED])
    exp = _experiment(observations=[_measured(9.0)])
    level, reason = determine_maturity_level(
        contract=contract, experiment=exp, verdicts=verdicts_for(contract, exp),
        implementation_conforms=True, constraint_status=ConstraintStatus.UNKNOWN,
        has_real_evidence=True, has_test_plan=True,
    )
    assert level is MaturityLevel.R4_VALIDATED
    assert "unverified" in reason


def test_constraint_violation_is_rejected_outright():
    contract = make_contract(metrics=[SPEED])
    exp = _experiment(observations=[_measured(9.0)])
    level, _ = determine_maturity_level(
        contract=contract, experiment=exp, verdicts=verdicts_for(contract, exp),
        implementation_conforms=True, constraint_status=ConstraintStatus.FAIL,
        has_real_evidence=True, has_test_plan=True,
    )
    assert level is MaturityLevel.REJECTED


def test_human_required_is_inconclusive_not_merely_designed():
    """An attempted-but-blocked validation is distinct from "never run"."""
    contract = make_contract(metrics=[SPEED])
    exp = _experiment(human=True, executed=False)
    level, reason = determine_maturity_level(
        contract=contract, experiment=exp, verdicts=verdicts_for(contract, exp),
        implementation_conforms=None, constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    assert level is MaturityLevel.INCONCLUSIVE
    assert "real-world input" in reason


def test_no_experiment_with_a_plan_is_l2_designed():
    contract = make_contract(metrics=[SPEED])
    level, _ = determine_maturity_level(
        contract=contract, experiment=None, verdicts=[],
        implementation_conforms=None, constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    assert level is MaturityLevel.R2_DESIGNED


def test_nothing_gathered_and_nothing_run_is_inconclusive():
    level, reason = determine_maturity_level(
        contract=make_contract(metrics=[SPEED]), experiment=None, verdicts=[],
        implementation_conforms=None, constraint_status=ConstraintStatus.PASS,
        has_real_evidence=False, has_test_plan=False,
    )
    assert level is MaturityLevel.INCONCLUSIVE
    assert "nothing has been verified" in reason


def test_unmeasured_metric_is_not_counted_as_failed():
    """"We did not measure it" and "it failed" must stay distinct."""
    contract = make_contract(
        metrics=[SPEED, SuccessMetric(
            key="memory", description="peak memory", unit="MB",
            target_threshold=100.0, direction=MetricDirection.MINIMIZE,
        )]
    )
    exp = _experiment(observations=[_measured(5.0)])
    verdicts = verdicts_for(contract, exp)
    memory = next(v for v in verdicts if v.metric_key == "memory")
    assert memory.met is None and memory.observed is None

    level, reason = determine_maturity_level(
        contract=contract, experiment=exp, verdicts=verdicts,
        implementation_conforms=True, constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    assert level is MaturityLevel.R4_VALIDATED
    assert "never measured" in reason


def test_evidence_quality_flags_single_run_without_variance():
    contract = make_contract(metrics=[SPEED])
    exp = _experiment(
        observations=[
            MetricObservation(
                metric_key="speed", value=3.0, unit="x",
                provenance=Provenance.MEASURED, n_repeats=1, stddev=None,
            )
        ]
    )
    gaps = evidence_quality_gaps(contract=contract, experiment=exp)
    assert any("not a measurement" in g for g in gaps)


# --------------------------------------------------------------------------
# Loop A vs Loop B are structurally distinct
# --------------------------------------------------------------------------


def test_the_two_loops_read_disjoint_test_categories():
    assert set(CONFORMANCE_CATEGORIES).isdisjoint(set(GOAL_CATEGORIES))


def test_goal_failure_alone_does_not_count_as_nonconformance():
    """Correctly built but missing the goal is a research result, not a defect."""
    plan = _plan()
    exp = _experiment(
        outcomes=[
            TestOutcome(test_id="T001", passed=True),
            TestOutcome(test_id="T002", passed=True),
            TestOutcome(test_id="T003", passed=False),  # goal_acceptance fails
        ]
    )
    conforms, notes, failed = check_implementation_conformance(plan=plan, experiment=exp)
    assert conforms is True
    assert failed == []  # the goal test is Loop B's business, not Loop A's
    assert "conformance case(s) passed" in notes


def test_conformance_failure_is_reported_as_an_implementation_defect():
    plan = _plan()
    exp = _experiment(
        outcomes=[
            TestOutcome(test_id="T001", passed=True),
            TestOutcome(test_id="T002", passed=False),
            TestOutcome(test_id="T003", passed=True),
        ]
    )
    conforms, notes, failed = check_implementation_conformance(plan=plan, experiment=exp)
    assert conforms is False
    assert failed == ["T002"]
    assert "implementation defect" in notes
    assert "says nothing yet about the research goal" in notes


def test_unrun_conformance_tests_give_unknown_not_pass():
    plan = _plan()
    exp = _experiment(
        outcomes=[TestOutcome(test_id=t, passed=None) for t in ("T001", "T002")]
    )
    conforms, notes, _ = check_implementation_conformance(plan=plan, experiment=exp)
    assert conforms is None
    assert "were executed" in notes


# --------------------------------------------------------------------------
# Diagnosis routes to the correct loop
# --------------------------------------------------------------------------


def _validation(**kw) -> ValidationResult:
    base = dict(
        run_id="t", proposal_id="PA", experiment_id="e1",
        implementation_conforms=True, goal_satisfied=None,
        maturity_level=MaturityLevel.R3_PROTOTYPED,
    )
    base.update(kw)
    return ValidationResult(**base)


def test_nonconformance_routes_to_the_implementation_loop():
    d = DiagnosisAgent().diagnose(
        validation=_validation(implementation_conforms=False, failed_test_ids=["T002"]),
        repair_attempts=0, max_repair_attempts=2,
        remaining_candidates=["PB"], has_real_evidence=True, evidence_exhausted=False,
    )
    assert d.action is Action.REPAIR_IMPLEMENTATION
    assert d.loop == "implementation"


def test_exhausted_repairs_switch_candidate_rather_than_retry_forever():
    d = DiagnosisAgent().diagnose(
        validation=_validation(implementation_conforms=False, failed_test_ids=["T002"]),
        repair_attempts=2, max_repair_attempts=2,
        remaining_candidates=["PB"], has_real_evidence=True, evidence_exhausted=False,
    )
    assert d.action is Action.SWITCH_CANDIDATE
    assert d.loop == "research"


def test_goal_shortfall_routes_to_the_research_loop():
    d = DiagnosisAgent().diagnose(
        validation=_validation(goal_satisfied=False),
        repair_attempts=0, max_repair_attempts=2,
        remaining_candidates=["PB"], has_real_evidence=True, evidence_exhausted=False,
    )
    assert d.loop == "research"
    assert d.action is Action.GATHER_EVIDENCE


def test_constraint_violation_is_not_repairable():
    d = DiagnosisAgent().diagnose(
        validation=_validation(maturity_level=MaturityLevel.REJECTED),
        repair_attempts=0, max_repair_attempts=5,
        remaining_candidates=["PB"], has_real_evidence=True, evidence_exhausted=False,
    )
    assert d.action is Action.SWITCH_CANDIDATE
    assert "non-compensatory" in d.rationale


def test_no_candidates_left_concludes_infeasible():
    d = DiagnosisAgent().diagnose(
        validation=_validation(maturity_level=MaturityLevel.REJECTED),
        repair_attempts=0, max_repair_attempts=5,
        remaining_candidates=[], has_real_evidence=True, evidence_exhausted=False,
    )
    assert d.action is Action.CONCLUDE_INFEASIBLE


def test_an_action_is_not_retried_after_being_tried():
    agent = DiagnosisAgent()
    v = _validation(goal_satisfied=False)
    first = agent.diagnose(
        validation=v, repair_attempts=0, max_repair_attempts=2,
        remaining_candidates=["PB"], has_real_evidence=True, evidence_exhausted=False,
    )
    assert first.action is Action.GATHER_EVIDENCE
    agent.record_attempt("PA", Action.GATHER_EVIDENCE)
    second = agent.diagnose(
        validation=v, repair_attempts=0, max_repair_attempts=2,
        remaining_candidates=["PB"], has_real_evidence=True, evidence_exhausted=False,
    )
    assert second.action is not Action.GATHER_EVIDENCE


def test_external_validation_is_requested_rather_than_simulated_again():
    d = DiagnosisAgent().diagnose(
        validation=_validation(
            maturity_level=MaturityLevel.INCONCLUSIVE,
            external_validation_plan=["interview 12 users"],
        ),
        repair_attempts=0, max_repair_attempts=2,
        remaining_candidates=["PB"], has_real_evidence=True, evidence_exhausted=False,
    )
    assert d.action is Action.REQUEST_EXTERNAL_VALIDATION
    assert "cost and no information" in d.rationale


# --------------------------------------------------------------------------
# Stop policy
# --------------------------------------------------------------------------


def _stop_args(**kw):
    base = dict(
        contract=make_contract(budget=Budget(max_iterations=3, max_repair_attempts=2)),
        iteration=0, elapsed_seconds=0.0, tokens_used=0, cost_usd=0.0,
        consecutive_no_gain=0, repair_attempts=0, goal_satisfied=None,
        candidates_remaining=2,
    )
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"goal_satisfied": True}, "goal_satisfied"),
        ({"candidates_remaining": 0}, "no_viable_candidates"),
        ({"iteration": 3}, "max_iterations"),
        ({"consecutive_no_gain": 2}, "no_progress"),
        ({"repair_attempts": 3}, "repair_attempts_exhausted"),
    ],
)
def test_each_budget_limit_stops_with_its_own_reason(overrides, reason):
    decision = should_stop(**_stop_args(**overrides))
    assert decision.should_stop
    assert decision.reason == reason
    assert decision.detail


def test_token_time_and_cost_ceilings_each_stop():
    contract = make_contract(
        budget=Budget(
            max_iterations=99, max_tokens=100,
            max_wall_clock_seconds=10.0, max_cost_usd=1.0,
        )
    )
    for kw, reason in (
        ({"tokens_used": 100}, "token_budget_exhausted"),
        ({"elapsed_seconds": 10.0}, "wall_clock_exhausted"),
        ({"cost_usd": 1.0}, "cost_budget_exhausted"),
    ):
        d = should_stop(**_stop_args(contract=contract, **kw))
        assert d.should_stop and d.reason == reason


def test_continues_while_budget_remains():
    d = should_stop(**_stop_args())
    assert not d.should_stop
    assert d.reason == "continue"


# --------------------------------------------------------------------------
# Value of information
# --------------------------------------------------------------------------


def test_experiment_priority_is_impact_over_cost_and_marked_predicted():
    contract = make_contract()
    rubric = build_rubric(contract)
    props = [
        make_proposal("PA", assumptions=["cheap unknown"], poc_cost=1.0),
        make_proposal("PB", assumptions=["expensive unknown"], poc_cost=8.0),
    ]
    evals = [
        evaluate_proposal(
            run_id="t", proposal_id=p.proposal_id, rubric=rubric,
            dimension_scores=scores_for(rubric, 3, unknown=("evidence_strength",)),
            constraint_status=ConstraintStatus.PASS,
            constraint_assessments=passing([]),
            unresolved_assumptions=p.critical_assumptions,
        )
        for p in props
    ]
    ranking = build_ranking(
        run_id="t", rubric_version=rubric.version, evaluations=evals, top_k=2
    )
    candidates = rank_experiments(proposals=props, evaluations=evals, ranking=ranking)

    assert candidates
    assert all(c.impact_is_predicted for c in candidates), (
        "predicted impact must never be presented as a measured quantity"
    )
    for c in candidates:
        # priority is stored rounded to 4dp, so compare at that resolution.
        assert c.priority == pytest.approx(
            c.predicted_decision_impact / c.estimated_cost_units, abs=5e-5
        )
    # Sorted by priority, so the cheap resolvable unknown comes first.
    assert candidates == sorted(candidates, key=lambda c: (-c.priority, c.label))
    assert "cheap unknown" in candidates[0].targets_assumption


def test_measured_but_thresholdless_is_not_reported_as_unmeasured():
    """Three states, not two: decided / undecidable / unobserved.

    Calling an observed-but-thresholdless metric "never measured" blames the
    experiment for a gap in the contract, sending the reader to fix the wrong
    thing. Surfaced by a real run whose model declared metrics with no targets.
    """
    contract = make_contract(
        metrics=[SuccessMetric(key="speed", description="d", unit="x")]  # no target
    )
    exp = _experiment(observations=[_measured(3.0)])
    verdicts = verdicts_for(contract, exp)

    assert verdicts[0].observed == 3.0, "the value WAS observed"
    assert verdicts[0].met is None, "but there is no threshold to decide it"

    level, reason = determine_maturity_level(
        contract=contract, experiment=exp, verdicts=verdicts,
        implementation_conforms=True, constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    assert level is MaturityLevel.R3_PROTOTYPED
    assert "no numeric threshold" in reason
    assert "gap in the contract, not in the experiment" in reason
    assert "never measured" not in reason

    result = build_validation_result(
        contract=contract, proposal_id="PA", experiment=exp,
        implementation_conforms=True, implementation_notes="",
        failed_test_ids=[], constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    gap = next(g for g in result.evidence_gaps if "speed" in g)
    assert "was measured (3.0)" in gap
    assert "no threshold" in gap


def test_truly_unobserved_still_says_never_measured():
    contract = make_contract(metrics=[SPEED])
    exp = _experiment(observations=[])
    level, reason = determine_maturity_level(
        contract=contract, experiment=exp,
        verdicts=verdicts_for(contract, exp),
        implementation_conforms=True, constraint_status=ConstraintStatus.PASS,
        has_real_evidence=True, has_test_plan=True,
    )
    assert level is MaturityLevel.R3_PROTOTYPED
    assert "observed at all" in reason
    assert "no numeric threshold" not in reason
