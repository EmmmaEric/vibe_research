"""Integration: the router-driven orchestrator, offline.

Note on budgets: every router action counts as an iteration now (matching the
spec's ``max_iterations = 8`` default), so a run needs several to complete a
cycle. ``max_iterations=1`` means "take one action", not "do one research
round" — the budget tests below rely on that.
"""

from __future__ import annotations

import json

import pytest

from research_harness.core.budget import ArtifactStore
from research_harness.core.orchestrator import Orchestrator
from research_harness.core.schemas import (
    ApprovalPolicy,
    Budget,
    ConstraintStatus,
    EventType,
    MaturityLevel,
    Mode,
    Phase,
    RouteAction,
    RunOutcome,
)
from research_harness.core.state import ResearchState
from research_harness.evaluation.judge_adapter import MockJudgeAdapter
from research_harness.experiments.adapters import (
    ManualEvidenceAdapter,
    SimulatedExperimentAdapter,
)
from research_harness.memory.repository import Repository
from research_harness.memory.retrieval import MemoryRetrieval
from research_harness.providers.simulated import SimulatedProvider
from research_harness.tools.search import FixtureSearchTool

ALGORITHM_IDEA = (
    "Explore a new algorithm to make sparse matrix multiplication faster on "
    "commodity CPUs under a fixed memory budget"
)
PRODUCT_IDEA = (
    "Design an offline document collaboration product for small businesses and "
    "assess its commercial viability"
)
FULL = Budget(max_iterations=10)


def build(tmp_path, *, adapter="simulated", seed=7, repo=None, fixtures=None):
    provider = SimulatedProvider(seed=seed)
    exp_adapter = (
        SimulatedExperimentAdapter(provider, seed=seed)
        if adapter == "simulated"
        else ManualEvidenceAdapter(reason="test: real-world input required")
    )
    return Orchestrator(
        provider=provider,
        judge=MockJudgeAdapter(),
        search_tools=[FixtureSearchTool(fixtures)],
        experiment_adapter=exp_adapter,
        store=ArtifactStore(tmp_path / "run"),
        repository=repo,
        seed=seed,
    )


@pytest.fixture
def orch(tmp_path, bundled_fixture_dir):
    return build(tmp_path, fixtures=bundled_fixture_dir)


def run(orch, *, idea=ALGORITHM_IDEA, mode=Mode.ADAPTIVE, budget=FULL, **kw):
    return orch.start(idea=idea, mode=mode, budget=budget, **kw)


# --------------------------------------------------------------------------
# Router-driven cycle
# --------------------------------------------------------------------------


def test_full_cycle_reaches_a_terminal_state_with_an_outcome(orch):
    state = run(orch)

    assert state.phase in (Phase.DONE, Phase.STOPPED)
    assert state.outcome is not None
    assert state.stop_decision is not None
    assert state.stop_decision.reason and state.stop_decision.detail
    assert state.final_maturity_level is not None

    assert state.contract is not None
    assert state.evidence
    assert state.proposals
    assert state.rubric is not None
    assert state.evaluations
    assert state.ranking is not None
    assert state.test_plans
    assert state.experiments
    assert state.validations


def test_there_is_no_fixed_pipeline_only_routed_actions(orch):
    """The control flow must be reconstructible from routing decisions alone."""
    state = run(orch)
    assert state.routing_decisions
    for d in state.routing_decisions:
        assert d.rule_fired, "every decision names the rule that fired"
        assert d.rationale
        assert d.trigger
        assert d.decided_by in ("rule", "model", "budget", "mode")
    # It ends on a terminal action, not by falling off a phase list.
    assert state.routing_decisions[-1].action in (
        RouteAction.STOP,
        RouteAction.REQUEST_HUMAN,
    )
    # And realised effects were attached after each non-terminal action.
    assert any(d.realised_new_evidence is not None for d in state.routing_decisions)


def test_evidence_is_gathered_before_candidates_are_scored(orch):
    """Scoring on absent evidence would be confident noise."""
    state = run(orch)
    actions = state.route_actions()
    assert actions[0] == RouteAction.GATHER_EVIDENCE.value


def test_event_log_records_the_spec_vocabulary(orch):
    state = run(orch)
    kinds = {e.event for e in state.events}
    for required in (
        EventType.GOAL_DEFINED,
        EventType.ROUTE_DECIDED,
        EventType.SOURCE_RETRIEVED,
        EventType.PROPOSAL_CREATED,
        EventType.RUBRIC_CREATED,
        EventType.SCORE_RECORDED,
        EventType.EXPERIMENT_PLANNED,
        EventType.EXPERIMENT_RUN,
        EventType.VALIDATION_RECORDED,
        EventType.RUN_STOPPED,
    ):
        assert required in kinds, f"missing event {required.value}"
    # Append-only: sequence numbers are dense and increasing.
    assert [e.seq for e in state.events] == list(range(1, len(state.events) + 1))


# --------------------------------------------------------------------------
# Modes  (spec 16)
# --------------------------------------------------------------------------


def test_goal_only_uses_one_agent_one_candidate_and_no_tournament(orch):
    state = run(orch, mode=Mode.GOAL_ONLY)
    assert state.mode is Mode.GOAL_ONLY
    assert len(state.proposals) == 1, "the baseline must not fan out into candidates"
    assert not state.tournament_ran
    forbidden = {
        RouteAction.RUN_PROPOSAL_TOURNAMENT.value,
        RouteAction.SPAWN_SUBAGENT.value,
    }
    assert not (set(state.route_actions()) & forbidden)
    # It still reaches a real conclusion with independent validation.
    assert state.validations
    assert state.final_maturity_level is not None


def test_tournament_mode_competes_candidates(tmp_path, bundled_fixture_dir):
    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o, mode=Mode.TOURNAMENT)
    assert state.tournament_ran
    assert len(state.proposals) >= 2
    assert RouteAction.RUN_PROPOSAL_TOURNAMENT.value in state.route_actions()
    assert state.ranking.pairwise is not None


def test_adaptive_mode_can_reach_the_multi_agent_actions(tmp_path, bundled_fixture_dir):
    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o, mode=Mode.ADAPTIVE)
    rich = {
        RouteAction.RUN_PROPOSAL_TOURNAMENT.value,
        RouteAction.SPAWN_SUBAGENT.value,
    }
    assert set(state.route_actions()) & rich, (
        "adaptive mode escalated to nothing; the router is effectively goal_only"
    )


def test_a_mode_downgrade_is_recorded_not_silent(tmp_path, bundled_fixture_dir):
    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o, mode=Mode.GOAL_ONLY)
    gated = [d for d in state.routing_decisions if d.decided_by == "mode"]
    if gated:  # only when a rule actually wanted a richer action
        for d in gated:
            assert "does not permit" in d.rationale


# --------------------------------------------------------------------------
# Candidates and scoring
# --------------------------------------------------------------------------


def test_candidates_are_substantively_different(orch):
    from research_harness.agents.proposal_designer import jaccard

    state = run(orch, mode=Mode.TOURNAMENT)
    assert len(state.proposals) >= 2
    mechanisms = [
        p.core_mechanism + " " + " ".join(p.key_steps) for p in state.proposals
    ]
    for i in range(len(mechanisms)):
        for j in range(i + 1, len(mechanisms)):
            assert jaccard(mechanisms[i], mechanisms[j]) < 0.72
    assert len({p.strategy_archetype for p in state.proposals}) == len(state.proposals)


def test_every_candidate_gets_alignment_constraints_scores_and_risk(orch):
    state = run(orch, mode=Mode.TOURNAMENT)
    rubric_keys = {d.key for d in state.required_rubric.dimensions}
    for p in state.proposals:
        ev = state.evaluation(p.proposal_id)
        assert ev is not None
        assert {s.dimension_key for s in ev.dimension_scores} == rubric_keys
        assert {a.constraint_key for a in ev.constraint_assessments} == {
            c.key for c in state.required_contract.hard_constraints
        }
        assert p.failure_modes and p.goal_mapping
        for s in ev.dimension_scores:
            assert s.justification


def test_constraint_violator_is_excluded_from_selection(orch):
    state = run(orch, mode=Mode.TOURNAMENT)
    violators = [e for e in state.evaluations if not e.eligible]
    assert violators, "expected the over-budget exploratory candidate to fail the gate"
    for v in violators:
        assert v.constraint_status is ConstraintStatus.FAIL
        assert "within_cost_budget" in v.violated_constraints
        assert v.proposal_id != state.selected_proposal_id
        assert v.proposal_id in state.ranking.excluded_for_constraints
        assert v.proposal_id not in state.ranking.top_k


def test_pairwise_swap_and_stability_are_recorded(orch):
    state = run(orch, mode=Mode.TOURNAMENT)
    pw = state.ranking.pairwise
    assert pw is not None
    assert pw.comparisons == 2 * len(pw.outcomes)
    for o in pw.outcomes:
        assert o.forward.first_proposal_id == o.proposal_x
        assert o.reverse.first_proposal_id == o.proposal_y
    assert 0.0 <= pw.position_bias_rate <= 1.0
    assert state.ranking.stability is not None and state.ranking.stability.findings


# --------------------------------------------------------------------------
# Honesty
# --------------------------------------------------------------------------


def test_simulated_run_is_labelled_and_capped_at_r3(orch):
    state = run(orch)
    assert state.final_maturity_level is not MaturityLevel.R5_GOAL_ACHIEVED
    assert state.latest_validation().goal_satisfied is not True
    assert any("SIMULATED" in w for w in state.warnings)
    assert all(
        o.provenance.value == "simulated"
        for exp in state.experiments
        for o in exp.observations
    )


def test_unexecutable_validation_is_inconclusive_with_an_external_plan(
    tmp_path, bundled_fixture_dir
):
    o = build(tmp_path, adapter="manual", fixtures=bundled_fixture_dir)
    state = run(o, idea=PRODUCT_IDEA, mode=Mode.TOURNAMENT)

    assert state.final_maturity_level is MaturityLevel.INCONCLUSIVE
    assert state.outcome is RunOutcome.REQUIRES_EXTERNAL_TEST
    v = state.latest_validation()
    assert v.goal_satisfied is None
    assert v.external_validation_plan
    assert state.route_actions()[-1] == RouteAction.REQUEST_HUMAN.value


def test_a_stalled_run_stops_rather_than_looping(orch):
    """A run with nothing left to learn must stop, not spin to budget."""
    state = run(orch, budget=Budget(max_iterations=30, max_consecutive_no_gain=2))
    assert state.phase in (Phase.DONE, Phase.STOPPED)
    assert state.iteration < 30, "the run burned its whole budget making no progress"
    assert state.outcome in (
        RunOutcome.STAGNATED,
        RunOutcome.GOAL_ACHIEVED,
        RunOutcome.REQUIRES_EXTERNAL_TEST,
        RunOutcome.PROPOSAL_INVALID,
        RunOutcome.EVIDENCE_INSUFFICIENT,
    )


# --------------------------------------------------------------------------
# Budget and requested depth
# --------------------------------------------------------------------------


def test_a_tiny_budget_stops_immediately_with_a_budget_reason(orch):
    state = run(orch, budget=Budget(max_iterations=1))
    assert state.stop_decision.reason == "budget_exhausted"
    assert state.outcome is RunOutcome.BUDGET_EXHAUSTED
    assert state.iteration <= 1
    # And it does not claim to have verified anything.
    assert state.final_maturity_level in (
        MaturityLevel.INCONCLUSIVE,
        MaturityLevel.R1_RESEARCHED,
    )


def test_depth_r1_stops_after_research_without_designing_or_testing(orch):
    state = run(orch, desired_maturity_level=MaturityLevel.R1_RESEARCHED)
    assert state.evidence
    assert not state.experiments
    assert not state.proposals
    assert state.stop_decision.reason == "requested_depth_reached"


def test_depth_r2_plans_but_does_not_execute(orch):
    state = run(orch, desired_maturity_level=MaturityLevel.R2_DESIGNED)
    assert state.test_plans
    assert not state.experiments
    assert state.final_maturity_level is MaturityLevel.R2_DESIGNED


# --------------------------------------------------------------------------
# Reweighting
# --------------------------------------------------------------------------


def test_reweighting_reranks_and_archives_the_previous_rubric(orch):
    state = run(orch, mode=Mode.TOURNAMENT)
    before_version = state.rubric.version
    before = [e.proposal_id for e in sorted(state.ranking.entries, key=lambda x: x.rank)]

    state = orch.reweight(state, {"cost_time": 25.0}, reason="cost became critical")

    assert state.rubric.version != before_version
    assert any(h.version == before_version for h in state.rubric_history)
    assert all(e.rubric_version == state.rubric.version for e in state.evaluations)
    after = [e.proposal_id for e in sorted(state.ranking.entries, key=lambda x: x.rank)]
    assert before != after or state.ranking.rubric_version != before_version
    assert state.selected_proposal_id not in state.ranking.excluded_for_constraints


def test_reweighting_an_unknown_dimension_is_rejected(orch):
    state = run(orch)
    with pytest.raises(ValueError, match="unknown dimension"):
        orch.reweight(state, {"not_a_dimension": 1.0}, reason="typo")


# --------------------------------------------------------------------------
# Persistence, resume, approval
# --------------------------------------------------------------------------


def test_state_round_trips_through_disk(orch, tmp_path):
    state = run(orch)
    reloaded = ResearchState.load(state.save(tmp_path / "state.json"))

    assert reloaded.run_id == state.run_id
    assert reloaded.mode is state.mode
    assert reloaded.phase is state.phase
    assert reloaded.outcome is state.outcome
    assert reloaded.required_contract.original_idea == ALGORITHM_IDEA
    assert len(reloaded.proposals) == len(state.proposals)
    assert reloaded.route_actions() == state.route_actions()
    assert [e.seq for e in reloaded.events] == [e.seq for e in state.events]
    assert reloaded.final_maturity_level is state.final_maturity_level


def test_approval_gate_pauses_before_experimenting_and_resumes(
    tmp_path, bundled_fixture_dir
):
    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(
        o, approval_policy=ApprovalPolicy(require_approval_for=["experiment"])
    )

    assert state.phase is Phase.AWAITING_APPROVAL
    assert state.awaiting_approval_stage == "experiment"
    assert not state.experiments, "no experiment may run before approval"
    assert o.resume(state).phase is Phase.AWAITING_APPROVAL, "resume must not bypass"

    resumed = o.approve(state, "experiment")
    assert resumed.phase in (Phase.DONE, Phase.STOPPED)
    assert resumed.experiments
    assert "experiment" in resumed.approved_stages
    assert any(e.event is EventType.HUMAN_APPROVAL for e in resumed.events)


def test_resume_does_not_repeat_a_completed_experiment(tmp_path, bundled_fixture_dir):
    """Spec section 12: side effects must not replay on resume."""
    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o)
    experiments_before = len(state.experiments)
    keys_before = [e.idempotency_key for e in state.events if e.idempotency_key]
    assert keys_before, "the experiment should have registered an idempotency key"

    # Re-open the finished run and force another pass with a fresh orchestrator.
    saved = state.save(tmp_path / "done.json")
    fresh = build(tmp_path, fixtures=bundled_fixture_dir)
    reloaded = ResearchState.load(saved)
    reloaded.phase = Phase.RUNNING
    reloaded.stop_decision = None
    reloaded.outcome = None
    again = fresh.resume(reloaded)

    assert len(again.experiments) == experiments_before, (
        "a completed experiment was executed a second time on resume"
    )


# --------------------------------------------------------------------------
# Artifacts and report
# --------------------------------------------------------------------------


def test_artifacts_match_the_specified_layout(orch, tmp_path):
    run(orch, mode=Mode.TOURNAMENT)
    root = tmp_path / "run"
    for rel in (
        "research_contract.json", "evidence.jsonl", "decisions.jsonl",
        "routing_decisions.jsonl", "events.jsonl", "run_summary.json",
        "state.json", "final_report.md", "research_report.md",
        "evaluation/rubric.json", "evaluation/scores.json",
        "evaluation/ranking.json", "evaluation/pairwise.json",
        "evaluation/next_action.json", "evaluation/ranking_report.md",
    ):
        assert (root / rel).is_file(), f"missing artifact {rel}"
    assert list((root / "proposals").glob("proposal_*.json"))
    assert list((root / "experiments").glob("test_plan_*.json"))
    assert list((root / "experiments").glob("run_*"))


def test_run_summary_has_no_placeholder_numbers(orch, tmp_path):
    run(orch)
    summary = json.loads((tmp_path / "run" / "run_summary.json").read_text("utf-8"))
    assert summary["mode"] == "adaptive"
    assert summary["maturity_level"]
    assert summary["routing_actions"]
    assert summary["used_simulated_provider"] is True
    # Unmeasured things are null, not zero-as-a-guess.
    assert summary["goal_satisfied"] in (None, True, False)


def test_report_contains_all_ten_sections_and_traceable_citations(orch, tmp_path):
    state = run(orch, mode=Mode.TOURNAMENT)
    report = (tmp_path / "run" / "final_report.md").read_text(encoding="utf-8")

    for heading in (
        "## 1. Original idea and success criteria",
        "## 2. Key evidence, sources and counter-evidence",
        "## 3. Candidates: strengths, limits and risks",
        "## 4. Scoring dimensions, weights, ranking and uncertainty",
        "## 5. Selected candidate and why",
        "## 6. Test plan and raw experimental results",
        "## 7. Did the implementation conform, and was the goal met?",
        "## 8. Failed assumptions, missing evidence, risks and next actions",
        "## 9. Reproduction",
        "## 10. Conclusion level",
    ):
        assert heading in report, f"report is missing section: {heading}"

    assert ALGORITHM_IDEA in report
    assert state.final_maturity_level.value in report
    assert "SIMULATED" in report
    assert "not a claim of global optimality" in report
    assert any(e.evidence_id in report for e in state.evidence)


# --------------------------------------------------------------------------
# Determinism and cross-domain generality
# --------------------------------------------------------------------------


def test_same_idea_and_seed_reproduce_the_same_run(tmp_path, bundled_fixture_dir):
    results = []
    for i in range(2):
        o = build(tmp_path / f"r{i}", seed=99, fixtures=bundled_fixture_dir)
        s = run(o)
        results.append(
            (
                s.route_actions(),
                [e.proposal_id for e in sorted(s.ranking.entries, key=lambda x: x.rank)],
                [round(e.point_score, 6) for e in s.evaluations],
                s.selected_proposal_id,
                s.final_maturity_level,
                s.outcome,
            )
        )
    assert results[0] == results[1], "the offline pipeline must be reproducible"


def test_two_unrelated_domains_run_on_the_same_core(tmp_path, bundled_fixture_dir):
    runs = {}
    for name, idea in (("algorithm", ALGORITHM_IDEA), ("product", PRODUCT_IDEA)):
        o = build(tmp_path / name, fixtures=bundled_fixture_dir)
        runs[name] = (o, run(o, idea=idea))

    algo_o, algo = runs["algorithm"]
    prod_o, prod = runs["product"]

    assert type(algo_o) is type(prod_o)
    assert type(algo) is type(prod) is ResearchState
    assert algo_o.router.__class__ is prod_o.router.__class__
    assert algo_o.general.__class__ is prod_o.general.__class__

    for s in (algo, prod):
        assert s.ranking is not None and s.evaluations and s.validations
        assert s.final_maturity_level is not None and s.outcome is not None

    assert algo.required_contract.domain_hint != prod.required_contract.domain_hint
    assert {m.key for m in algo.required_contract.success_metrics} != {
        m.key for m in prod.required_contract.success_metrics
    }
    algo_dims = {d.key for d in algo.required_rubric.dimensions}
    prod_dims = {d.key for d in prod.required_rubric.dimensions}
    assert algo_dims & prod_dims, "the two domains share no scoring dimension at all"


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


def test_run_is_persisted_to_sqlite_and_searchable(tmp_path, bundled_fixture_dir):
    repo = Repository(tmp_path / "memory.sqlite3")
    try:
        o = build(tmp_path, repo=repo, fixtures=bundled_fixture_dir)
        state = run(o, mode=Mode.TOURNAMENT)

        assert any(r["run_id"] == state.run_id for r in repo.list_runs())
        chain = repo.causal_chain(state.run_id)
        for stage in (
            "contract", "evidence", "proposals", "scores", "test_plans",
            "experiments", "validations",
        ):
            assert chain[stage], f"causal chain stage '{stage}' was not persisted"
        assert any(row["eligible"] == 0 for row in chain["scores"]), (
            "the constraint-violating candidate's ineligibility was not persisted"
        )
        assert MemoryRetrieval(repo).search("sparse matrix")
    finally:
        repo.close()


def test_memory_search_survives_fts_operator_characters(tmp_path, bundled_fixture_dir):
    repo = Repository(tmp_path / "memory.sqlite3")
    try:
        o = build(tmp_path, repo=repo, fixtures=bundled_fixture_dir)
        run(o, budget=Budget(max_iterations=4))
        retrieval = MemoryRetrieval(repo)
        for query in ('matrix "AND" *', "cost: (high)", "a-b-c", "NOT", ""):
            retrieval.search(query)  # must not raise
    finally:
        repo.close()


# --------------------------------------------------------------------------
# Provenance integrity across a resume
# --------------------------------------------------------------------------


def test_provenance_reflects_the_backends_that_actually_ran(
    tmp_path, bundled_fixture_dir
):
    """A resumed run given a different adapter must not report the old one.

    Provenance used to be frozen at start(), so a run resumed onto a real
    adapter still reported "SIMULATED" -- wrong in the safe direction there,
    but the inverse is the same bug and would be the dangerous one.
    """
    first = build(tmp_path, adapter="simulated", fixtures=bundled_fixture_dir)
    state = run(first, desired_maturity_level=MaturityLevel.R1_RESEARCHED)
    assert state.provenance.experiment_adapter_id == "simulated"

    # Reopen with a different adapter, on the same artifact store.
    second = build(tmp_path, adapter="manual", fixtures=bundled_fixture_dir)
    deepened = second.deepen(state, MaturityLevel.R3_PROTOTYPED)

    assert deepened.provenance.experiment_adapter_id == "manual"
    assert any("Backends changed on resume" in n for n in deepened.notes)
    assert any("simulated" in n and "manual" in n for n in deepened.notes)


def test_the_simulated_warning_describes_what_ran_not_what_was_configured(
    tmp_path, bundled_fixture_dir
):
    from research_harness.reports.generator import ReportGenerator

    o = build(tmp_path, adapter="simulated", fixtures=bundled_fixture_dir)
    state = run(o)
    reporter = ReportGenerator()

    assert any(e.is_simulated for e in state.experiments)
    assert any("SIMULATED adapter" in w for w in reporter.all_warnings(state))
    # The claim is derived, so it is not baked into the persisted state.
    assert not any("SIMULATED adapter" in w for w in state.warnings)


def test_a_design_only_run_makes_no_claim_about_an_adapter(
    tmp_path, bundled_fixture_dir
):
    from research_harness.reports.generator import ReportGenerator

    o = build(tmp_path, adapter="simulated", fixtures=bundled_fixture_dir)
    state = run(o, desired_maturity_level=MaturityLevel.R1_RESEARCHED)

    assert not state.experiments
    warnings = ReportGenerator().all_warnings(state)
    assert not any("adapter" in w for w in warnings), (
        "nothing ran, so no adapter claim should be made"
    )


def test_deepening_refuses_to_lower_the_requested_maturity(
    tmp_path, bundled_fixture_dir
):
    """Asking for less verification after seeing results weakens the goal."""
    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o, desired_maturity_level=MaturityLevel.R5_GOAL_ACHIEVED)
    with pytest.raises(ValueError, match="refusing to lower"):
        o.deepen(state, MaturityLevel.R1_RESEARCHED)


def test_deepening_keeps_the_contract_and_versions_the_change(
    tmp_path, bundled_fixture_dir
):
    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o, desired_maturity_level=MaturityLevel.R1_RESEARCHED)
    before = state.required_contract
    keys_before = {m.key for m in before.success_metrics}

    deepened = o.deepen(state, MaturityLevel.R3_PROTOTYPED)
    after = deepened.required_contract

    # Same contract, deepened -- which is the whole point: emergent metric keys
    # stay valid, so a binding read from it does not go stale.
    assert after.original_idea == before.original_idea
    assert {m.key for m in after.success_metrics} == keys_before
    assert after.version == before.version + 1
    assert "requested maturity raised" in after.revisions[-1].reason
    assert after.desired_maturity_level is MaturityLevel.R3_PROTOTYPED
    # And it actually went further than the research-only run did.
    assert deepened.experiments


def test_the_selection_tracks_the_ranking_until_something_is_tested(
    tmp_path, bundled_fixture_dir
):
    """Re-scoring must move the selection, or its result is discarded.

    A real run experimented on the rank-3 candidate because a skeptic review
    re-scored and changed the ranking, while the stale selection survived.
    """
    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o, mode=Mode.TOURNAMENT)

    leader = next(
        e.proposal_id
        for e in sorted(state.ranking.entries, key=lambda x: x.rank)
        if e.eligible
    )
    tested = {e.proposal_id for e in state.experiments}

    # Whatever was experimented on must be a candidate the ranking actually
    # led with at the time, i.e. the leader or something already retired.
    assert state.selected_proposal_id in tested | {leader}
    for pid in tested:
        assert pid not in state.ranking.excluded_for_constraints, (
            "an experiment ran on a constraint-violating candidate"
        )


def test_a_selection_change_from_rescoring_is_recorded(tmp_path, bundled_fixture_dir):
    from research_harness.core.schemas import Budget

    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o, mode=Mode.ADAPTIVE, budget=Budget(max_iterations=10))
    # Not every run reorders, but when one does it must say so rather than
    # silently experimenting on a different candidate than the report shows.
    changes = [n for n in state.notes if "selection changed from" in n]
    for note in changes:
        assert "no work is discarded" in note


def test_an_already_tested_candidate_is_not_swapped_out_by_rescoring(
    tmp_path, bundled_fixture_dir
):
    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o, mode=Mode.TOURNAMENT)
    if not state.experiments:
        pytest.skip("this run executed nothing")
    tested = state.experiments[-1].proposal_id
    # Either it is still selected, or the router explicitly retired it.
    assert (
        state.selected_proposal_id == tested or tested in state.tried_proposal_ids
    ), "a tested candidate was dropped without being retired"


def test_a_selection_that_is_not_the_leader_explains_itself(
    tmp_path, bundled_fixture_dir
):
    """A tested candidate may outrank the leader; the report must say why.

    Otherwise "Selected: PA" beside "rank 1: PB" reads as a contradiction.
    """
    from research_harness.reports.generator import ReportGenerator

    o = build(tmp_path, fixtures=bundled_fixture_dir)
    state = run(o, mode=Mode.TOURNAMENT)
    status = ReportGenerator().status_report(state)

    leader = next(
        (
            e.proposal_id
            for e in sorted(state.ranking.entries, key=lambda x: x.rank)
            if e.eligible
        ),
        None,
    )
    if state.selected_proposal_id and leader and state.selected_proposal_id != leader:
        assert "already experimented on" in status or "please report it" in status
    # And the inconsistency warning must never fire on a healthy run.
    assert "please report it" not in status
