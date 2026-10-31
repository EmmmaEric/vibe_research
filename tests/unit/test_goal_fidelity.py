"""Goal fidelity: the contract's authority over what "success" means.

The invariants here are the ones that stop a harness from quietly moving the
goalposts: the original idea is immutable, revisions are versioned and reasoned,
acceptance criteria come from the contract, and execution roles cannot amend the
plan they are judged by.
"""

from __future__ import annotations

import pytest

from research_harness.agents.executor import Executor
from research_harness.agents.goal_analyst import GoalAnalyst
from research_harness.agents.test_designer import TestDesigner
from research_harness.core.schemas import (
    MetricDirection,
    Origin,
    SuccessMetric,
    TestCase,
    MaturityLevel,
)
from research_harness.experiments.planner import (
    EXECUTION_ROLES,
    TestPlanRegistry,
    TestPlanTampering,
)

from ..conftest import make_contract, make_proposal


# --------------------------------------------------------------------------
# Contract immutability and versioning
# --------------------------------------------------------------------------


def test_original_idea_cannot_be_rewritten():
    c = make_contract(idea="the user's exact words")
    with pytest.raises(ValueError, match="immutable"):
        c.revise(reason="tidying", author="system", changes={"original_idea": "nicer"})
    assert c.original_idea == "the user's exact words"


def test_revision_bumps_version_and_records_why():
    c = make_contract()
    revised = c.revise(
        reason="requester supplied a real threshold",
        author="user",
        changes={"goal_type": "cost_reduction"},
    )
    assert revised.version == 2
    assert c.version == 1  # original untouched
    assert revised.goal_type == "cost_reduction"
    rev = revised.revisions[-1]
    assert rev.reason and rev.author == "user"
    assert rev.changed_fields == ["goal_type"]
    assert revised.original_idea == c.original_idea


def test_empty_idea_is_rejected():
    with pytest.raises(ValueError):
        make_contract(idea="   ")


# --------------------------------------------------------------------------
# Assumed vs. stated
# --------------------------------------------------------------------------


def test_harness_invented_metrics_are_flagged_assumed(provider):
    c = GoalAnalyst(provider).build_contract(
        run_id="t", idea="something with no stated numbers at all"
    )
    assert c.success_metrics
    assert all(m.assumed for m in c.success_metrics), (
        "metrics the harness invented must be marked assumed so a report can "
        "distinguish them from the requester's own goals"
    )
    assert any("placeholder" in a.lower() for a in c.assumptions)


def test_missing_threshold_is_surfaced_as_an_unknown(provider):
    c = GoalAnalyst(provider).build_contract(
        run_id="t", idea="make the thing better somehow"
    )
    no_target = [m.key for m in c.success_metrics if m.target_threshold is None]
    assert no_target
    for key in no_target:
        assert any(key in u for u in c.unknowns), (
            f"metric {key} has no threshold but this is not recorded as an unknown"
        )


def test_stated_numeric_ceiling_is_captured_as_not_assumed(provider):
    c = GoalAnalyst(provider).build_contract(
        run_id="t", idea="build a home energy monitor for under 50 USD per unit"
    )
    stated = [c_ for c_ in c.hard_constraints if not c_.assumed]
    assert stated, "an explicit ceiling in the idea text should become a stated constraint"
    assert any("50" in c_.description for c_ in stated)


def test_every_contract_gets_a_machine_checkable_budget_constraint(provider):
    c = GoalAnalyst(provider).build_contract(run_id="t", idea="anything at all")
    budget = next(x for x in c.hard_constraints if x.key == "within_cost_budget")
    assert budget.limit_value is not None
    assert budget.limit_metric == "declared_cost_units"
    assert budget.limit_direction is MetricDirection.MINIMIZE


def test_metric_threshold_comparison_respects_direction():
    up = SuccessMetric(
        key="s", description="d", unit="x", target_threshold=2.0,
        direction=MetricDirection.MAXIMIZE,
    )
    down = SuccessMetric(
        key="s", description="d", unit="ms", target_threshold=100.0,
        direction=MetricDirection.MINIMIZE,
    )
    assert up.is_met(2.0) and up.is_met(3.0) and not up.is_met(1.9)
    assert down.is_met(100.0) and down.is_met(50.0) and not down.is_met(101.0)
    # No threshold -> cannot be "met" by default.
    assert not SuccessMetric(key="s", description="d", unit="x").is_met(999.0)


# --------------------------------------------------------------------------
# Test plans derive from the contract, not the implementation
# --------------------------------------------------------------------------


def test_goal_acceptance_criteria_restate_contract_thresholds(provider):
    contract = make_contract(
        metrics=[
            SuccessMetric(
                key="latency_ms", description="p95 latency", unit="ms",
                target_threshold=250.0, direction=MetricDirection.MINIMIZE,
            )
        ]
    )
    plan = TestDesigner(provider).design(
        contract=contract, proposal=make_proposal("PA")
    )
    goal_cases = plan.cases_for("goal_acceptance")
    assert goal_cases
    criterion = next(c.acceptance_criterion for c in goal_cases if c.measures_metric == "latency_ms")
    assert "250.0" in criterion and "<=" in criterion


def test_plan_covers_all_six_categories(provider):
    plan = TestDesigner(provider).design(
        contract=make_contract(), proposal=make_proposal("PA")
    )
    for category in (
        "contract", "correctness", "edge_robustness",
        "comparison", "goal_acceptance", "evidence_quality",
    ):
        assert plan.cases_for(category), f"no {category} case was generated"


def test_metric_without_a_threshold_yields_an_undecidable_criterion(provider):
    contract = make_contract(
        metrics=[SuccessMetric(key="vibes", description="d", unit="?")]
    )
    plan = TestDesigner(provider).design(
        contract=contract, proposal=make_proposal("PA")
    )
    case = next(c for c in plan.cases_for("goal_acceptance") if c.measures_metric == "vibes")
    assert "UNDECIDABLE" in case.acceptance_criterion
    assert case.requires_external_resource


def test_comparison_case_names_an_independent_baseline(provider):
    plan = TestDesigner(provider).design(
        contract=make_contract(), proposal=make_proposal("PA")
    )
    assert plan.independent_baseline
    assert any(c.baseline for c in plan.cases_for("comparison"))


def test_randomness_controls_are_specified(provider):
    plan = TestDesigner(provider).design(
        contract=make_contract(), proposal=make_proposal("PA")
    )
    assert "seed" in plan.randomness_controls.lower()
    assert "repeat" in plan.randomness_controls.lower()


# --------------------------------------------------------------------------
# Plan custody
# --------------------------------------------------------------------------


def test_hash_changes_when_criteria_change(provider):
    plan = TestDesigner(provider).design(
        contract=make_contract(), proposal=make_proposal("PA")
    )
    before = plan.content_hash()
    weakened = plan.model_copy(
        update={
            "cases": [
                c.model_copy(update={"acceptance_criterion": "anything goes"})
                for c in plan.cases
            ]
        }
    )
    assert weakened.content_hash() != before


def test_registry_detects_a_tampered_plan(provider):
    registry = TestPlanRegistry()
    plan = TestDesigner(provider).design(
        contract=make_contract(), proposal=make_proposal("PA")
    )
    registry.register(plan)
    registry.verify_unchanged(plan)  # unchanged -> fine

    weakened = plan.model_copy(
        update={"cases": [c.model_copy(update={"acceptance_criterion": "pass"})
                          for c in plan.cases]}
    )
    with pytest.raises(TestPlanTampering, match="content changed"):
        registry.verify_unchanged(weakened)


def test_unregistered_plan_cannot_judge_a_result(provider):
    registry = TestPlanRegistry()
    plan = TestDesigner(provider).design(
        contract=make_contract(), proposal=make_proposal("PA")
    )
    with pytest.raises(TestPlanTampering, match="never registered"):
        registry.verify_unchanged(plan)


@pytest.mark.parametrize("role", sorted(EXECUTION_ROLES))
def test_execution_roles_cannot_amend_the_plan(provider, role):
    registry = TestPlanRegistry()
    plan = TestDesigner(provider).design(
        contract=make_contract(), proposal=make_proposal("PA")
    )
    registry.register(plan)
    with pytest.raises(TestPlanTampering, match="may not amend"):
        registry.amend(plan, reason="tests are too strict", author=role, cases=[])


def test_executor_amend_method_always_refuses():
    ex = Executor(TestPlanRegistry())
    with pytest.raises(PermissionError, match="may not amend"):
        ex.amend_plan()


def test_test_designer_may_amend_with_a_reason(provider):
    registry = TestPlanRegistry()
    plan = TestDesigner(provider).design(
        contract=make_contract(), proposal=make_proposal("PA")
    )
    registry.register(plan)

    new_cases = list(plan.cases) + [
        TestCase(
            test_id="T900", category="correctness", description="extra",
            procedure="do it", acceptance_criterion="it holds",
        )
    ]
    amended = registry.amend(
        plan,
        reason="requester clarified an additional correctness requirement",
        author="test_designer",
        cases=new_cases,
    )
    assert amended.version == plan.version + 1
    assert amended.plan_id != plan.plan_id
    assert amended.amendments[-1].author == "test_designer"
    assert "clarified" in amended.amendments[-1].reason
    # Both versions remain verifiable; history is not rewritten.
    registry.verify_unchanged(amended)
    registry.verify_unchanged(plan)


def test_amendment_without_a_reason_is_refused(provider):
    registry = TestPlanRegistry()
    plan = TestDesigner(provider).design(
        contract=make_contract(), proposal=make_proposal("PA")
    )
    registry.register(plan)
    with pytest.raises(TestPlanTampering, match="requires a stated reason"):
        registry.amend(plan, reason="  ", author="test_designer")


# --------------------------------------------------------------------------
# Metric override guards (exercised directly; the CLI wraps these)
# --------------------------------------------------------------------------


def _override(contract, set_metrics=None, remove_metrics=None):
    from research_harness.core.orchestrator import Orchestrator
    from research_harness.core.state import ResearchState

    return Orchestrator._apply_metric_overrides(
        contract,
        set_metrics or {},
        remove_metrics or [],
        ResearchState(run_id="t"),
    )


def test_requester_stated_metric_cannot_be_removed():
    """Dropping a goal the requester actually stated would redefine success."""
    contract = make_contract(
        metrics=[
            SuccessMetric(
                key="stated", description="the user asked for this", unit="x",
                target_threshold=2.0, origin=Origin.USER_SPECIFIED,
            ),
            SuccessMetric(
                key="guessed", description="harness placeholder", unit="x",
                target_threshold=1.0, origin=Origin.ASSUMED,
            ),
        ]
    )
    with pytest.raises(ValueError, match="refusing to remove"):
        _override(contract, remove_metrics=["stated"])

    # The assumed placeholder may be dropped, with the reason recorded.
    revised = _override(contract, remove_metrics=["guessed"])
    assert {m.key for m in revised.success_metrics} == {"stated"}
    assert "removed assumed placeholder" in revised.revisions[-1].reason


def test_cannot_remove_every_metric():
    contract = make_contract(
        metrics=[
            SuccessMetric(key="only", description="d", unit="x", origin=Origin.ASSUMED)
        ]
    )
    with pytest.raises(ValueError, match="no metric cannot decide"):
        _override(contract, remove_metrics=["only"])


def test_setting_a_threshold_marks_the_metric_as_stated():
    contract = make_contract(
        metrics=[
            SuccessMetric(key="m", description="d", unit="x", origin=Origin.ASSUMED)
        ]
    )
    revised = _override(contract, set_metrics={"m": 7.0})
    metric = revised.success_metrics[0]
    assert metric.target_threshold == 7.0
    assert metric.assumed is False
    assert revised.version == 2
    assert "set target" in revised.revisions[-1].reason


def test_contradictory_set_and_remove_is_refused():
    contract = make_contract(
        metrics=[
            SuccessMetric(key="m", description="d", unit="x", origin=Origin.ASSUMED),
            SuccessMetric(key="n", description="d", unit="x", origin=Origin.ASSUMED),
        ]
    )
    with pytest.raises(ValueError, match="opposite intentions"):
        _override(contract, set_metrics={"m": 1.0}, remove_metrics=["m"])


# --------------------------------------------------------------------------
# Metric binding for real experiments
# --------------------------------------------------------------------------


def test_a_stale_metric_binding_fails_loudly(tmp_path, bundled_fixture_dir):
    """A mapping that names a non-existent metric must not bind silently.

    With a real model, metric keys are emergent and differ between runs, so a
    stale --map-metric key is likely. Binding nothing would make the run report
    "no contract metric was measured" -- true, but it reads as the experiment's
    fault rather than the mapping's.
    """
    from research_harness.core.budget import ArtifactStore
    from research_harness.core.orchestrator import Orchestrator
    from research_harness.evaluation.judge_adapter import MockJudgeAdapter
    from research_harness.experiments.adapters import PytestExperimentAdapter
    from research_harness.providers.simulated import SimulatedProvider
    from research_harness.tools.search import FixtureSearchTool

    examples = pytest.importorskip("pathlib").Path(__file__).resolve().parents[2]
    project = examples / "examples" / "algorithm_idea"
    if not (project / "tests").is_dir():
        pytest.skip("bundled example A is not present")

    orch = Orchestrator(
        provider=SimulatedProvider(seed=1),
        judge=MockJudgeAdapter(),
        search_tools=[FixtureSearchTool(bundled_fixture_dir)],
        experiment_adapter=PytestExperimentAdapter(
            project_dir=project,
            metric_aliases={"not_a_contract_metric": "candidate_median_seconds"},
        ),
        store=ArtifactStore(tmp_path / "run"),
    )
    with pytest.raises(ValueError, match="does not declare"):
        orch.start(idea="speed up sparse matrix multiplication")


def test_a_valid_metric_binding_is_accepted_and_recorded(tmp_path, bundled_fixture_dir):
    from research_harness.core.budget import ArtifactStore
    from research_harness.core.orchestrator import Orchestrator
    from research_harness.core.schemas import Budget, MaturityLevel
    from research_harness.evaluation.judge_adapter import MockJudgeAdapter
    from research_harness.experiments.adapters import PytestExperimentAdapter
    from research_harness.providers.simulated import SimulatedProvider
    from research_harness.tools.search import FixtureSearchTool

    root = pytest.importorskip("pathlib").Path(__file__).resolve().parents[2]
    project = root / "examples" / "algorithm_idea"
    if not (project / "tests").is_dir():
        pytest.skip("bundled example A is not present")

    orch = Orchestrator(
        provider=SimulatedProvider(seed=1),
        judge=MockJudgeAdapter(),
        search_tools=[FixtureSearchTool(bundled_fixture_dir)],
        experiment_adapter=PytestExperimentAdapter(
            project_dir=project,
            # The simulated analyst uses template names, so this one exists.
            metric_aliases={"speedup_vs_baseline": "speedup_vs_baseline"},
        ),
        store=ArtifactStore(tmp_path / "run"),
    )
    state = orch.start(
        idea="speed up sparse matrix multiplication",
        domain_hint="ai_algorithm",
        budget=Budget(max_iterations=6),
        desired_maturity_level=MaturityLevel.R5_GOAL_ACHIEVED,
    )
    assert state.experiments
    experiment = state.experiments[-1]
    assert experiment.is_simulated is False, "the pytest adapter really ran"
    # The binding is recorded on the result, because it is a research judgement.
    assert "speedup_vs_baseline" in experiment.environment.get("metric_aliases", "")
    measured = {o.metric_key for o in experiment.observations}
    assert "speedup_vs_baseline" in measured
