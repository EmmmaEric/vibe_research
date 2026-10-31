"""Adaptive router: rule precedence, mode gating, and auditability.

The spec's requirement (section 21.6) is that a simple task stays on one agent
while a complex, contested one can escalate — and that routing is never decided
by an opaque model opinion alone.
"""

from __future__ import annotations

import pytest

from research_harness.core.router import (
    ACTION_COST,
    MODE_ALLOWED,
    AdaptiveRouter,
    RouterContext,
    record_outcome,
)
from research_harness.core.schemas import Mode, RouteAction


def ctx(**kw) -> RouterContext:
    """A clean, mid-run context: evidence gathered, nothing contested."""
    base = dict(
        mode=Mode.ADAPTIVE,
        iteration=1,
        real_evidence_count=5,
        critical_unknowns=0,
        proposal_count=1,
    )
    base.update(kw)
    return RouterContext(**base)


def decide(**kw) -> RouteAction:
    return AdaptiveRouter().decide(ctx(**kw), run_id="t").action


# --------------------------------------------------------------------------
# Single-agent default  (spec 21.6)
# --------------------------------------------------------------------------


def test_simple_task_stays_on_a_single_agent():
    assert decide() is RouteAction.CONTINUE_SINGLE_AGENT


def test_one_clear_candidate_does_not_trigger_a_tournament():
    assert decide(
        proposal_count=1, scored=True, score_gap_top_two=None, ranking_unstable=False
    ) is RouteAction.CONTINUE_SINGLE_AGENT


def test_comfortable_score_gap_does_not_trigger_a_tournament():
    assert decide(
        proposal_count=3, scored=True, score_gap_top_two=30.0
    ) is RouteAction.CONTINUE_SINGLE_AGENT


# --------------------------------------------------------------------------
# Escalation  (spec 21.6)
# --------------------------------------------------------------------------


def test_close_candidates_trigger_a_tournament():
    assert decide(
        proposal_count=3, scored=True, score_gap_top_two=2.0
    ) is RouteAction.RUN_PROPOSAL_TOURNAMENT


def test_unstable_ranking_triggers_a_tournament_even_with_a_wide_gap():
    assert decide(
        proposal_count=2, scored=True, score_gap_top_two=40.0, ranking_unstable=True
    ) is RouteAction.RUN_PROPOSAL_TOURNAMENT


def test_a_tournament_is_not_run_twice():
    assert decide(
        proposal_count=3, scored=True, score_gap_top_two=1.0, tournament_ran=True
    ) is not RouteAction.RUN_PROPOSAL_TOURNAMENT


def test_high_complexity_spawns_a_subagent():
    assert decide(complexity_score=0.9) is RouteAction.SPAWN_SUBAGENT


def test_expert_review_spawns_a_subagent_regardless_of_complexity():
    assert decide(complexity_score=0.0, expert_review_needed=True) is (
        RouteAction.SPAWN_SUBAGENT
    )


# --------------------------------------------------------------------------
# Evidence first
# --------------------------------------------------------------------------


def test_no_evidence_routes_to_gathering_before_anything_else():
    assert decide(
        real_evidence_count=0, proposal_count=3, scored=True, score_gap_top_two=1.0
    ) is RouteAction.GATHER_EVIDENCE


def test_a_critical_unknown_routes_to_gathering():
    assert decide(critical_unknowns=2) is RouteAction.GATHER_EVIDENCE


def test_exhausted_sources_stop_the_gathering_loop():
    """Otherwise a run with unreachable evidence would gather forever."""
    assert decide(
        real_evidence_count=0, evidence_sources_exhausted=True
    ) is not RouteAction.GATHER_EVIDENCE


def test_cheap_decisive_test_beats_further_analysis():
    assert decide(
        cheap_decisive_test_available=True, test_plan_ready=True, complexity_score=0.9
    ) is RouteAction.RUN_EXPERIMENT


def test_a_ready_plan_that_never_ran_gets_executed():
    assert decide(test_plan_ready=True, experiment_ran=False) is (
        RouteAction.RUN_EXPERIMENT
    )


# --------------------------------------------------------------------------
# Precedence: safety > budget > done > blocked > stagnation
# --------------------------------------------------------------------------


def test_unauthorized_action_outranks_everything():
    """Including a satisfied goal and an exhausted budget."""
    action = decide(
        unauthorized_action="external_write",
        budget_exhausted=True,
        goal_satisfied=True,
        real_evidence_count=0,
    )
    assert action is RouteAction.REQUEST_HUMAN


def test_pending_approval_outranks_budget_and_evidence():
    assert decide(
        pending_approval_stage="experiment",
        budget_exhausted=True,
        real_evidence_count=0,
    ) is RouteAction.REQUEST_HUMAN


def test_budget_exhaustion_outranks_a_satisfied_goal_report():
    d = AdaptiveRouter().decide(
        ctx(budget_exhausted=True, budget_detail="max_iterations", goal_satisfied=True),
        run_id="t",
    )
    assert d.action is RouteAction.STOP
    assert d.decided_by == "budget"
    assert "max_iterations" in d.rationale


def test_satisfied_goal_stops_rather_than_seeking_reassurance():
    assert decide(goal_satisfied=True, complexity_score=1.0) is RouteAction.STOP


def test_external_requirement_requests_a_human_instead_of_simulating():
    assert decide(requires_external_input=True, test_plan_ready=True) is (
        RouteAction.REQUEST_HUMAN
    )


def test_stagnation_stops_the_run():
    d = AdaptiveRouter().decide(
        ctx(consecutive_no_gain=2, max_consecutive_no_gain=2), run_id="t"
    )
    assert d.action is RouteAction.STOP
    assert d.rule_fired == "stagnated"


def test_stagnation_outranks_further_evidence_gathering():
    assert decide(
        consecutive_no_gain=3, max_consecutive_no_gain=2, real_evidence_count=0
    ) is RouteAction.STOP


# --------------------------------------------------------------------------
# The two loops stay distinct
# --------------------------------------------------------------------------


def test_implementation_defect_retests_within_the_repair_budget():
    d = AdaptiveRouter().decide(
        ctx(implementation_conforms=False, repair_attempts=0, max_repair_attempts=2),
        run_id="t",
    )
    assert d.action is RouteAction.RUN_EXPERIMENT
    assert d.rule_fired == "implementation_defect"
    assert "research question is untouched" in d.rationale


def test_exhausted_repairs_stop_patching_the_same_candidate():
    assert decide(
        implementation_conforms=False, repair_attempts=2, max_repair_attempts=2
    ) is not RouteAction.RUN_EXPERIMENT


def test_correct_build_that_misses_the_goal_recompetes_candidates():
    """A built-right-but-wrong candidate is a research result, not a bug."""
    d = AdaptiveRouter().decide(
        ctx(
            goal_satisfied=False,
            implementation_conforms=True,
            untried_candidates=2,
            experiment_ran=True,
            evidence_sources_exhausted=True,
        ),
        run_id="t",
    )
    assert d.action is RouteAction.RUN_PROPOSAL_TOURNAMENT
    assert d.rule_fired == "goal_missed_alternatives_remain"


def test_goal_missed_with_nothing_left_stops():
    d = AdaptiveRouter().decide(
        ctx(
            goal_satisfied=False,
            implementation_conforms=True,
            untried_candidates=0,
            experiment_ran=True,
            evidence_sources_exhausted=True,
        ),
        run_id="t",
    )
    assert d.action is RouteAction.STOP
    assert d.rule_fired == "options_exhausted"


# --------------------------------------------------------------------------
# Mode gating makes goal_only a real baseline  (spec 16)
# --------------------------------------------------------------------------


def test_goal_only_mode_cannot_reach_multi_agent_actions():
    forbidden = {RouteAction.SPAWN_SUBAGENT, RouteAction.RUN_PROPOSAL_TOURNAMENT}
    assert not (MODE_ALLOWED[Mode.GOAL_ONLY] & forbidden)


def test_goal_only_downgrades_a_tournament_to_single_agent():
    d = AdaptiveRouter().decide(
        ctx(mode=Mode.GOAL_ONLY, proposal_count=3, scored=True, score_gap_top_two=1.0),
        run_id="t",
    )
    assert d.action is RouteAction.CONTINUE_SINGLE_AGENT
    assert d.decided_by == "mode"
    assert "does not permit" in d.rationale
    assert "contested_candidates" in d.rule_fired


def test_goal_only_downgrades_a_subagent_spawn():
    d = AdaptiveRouter().decide(
        ctx(mode=Mode.GOAL_ONLY, complexity_score=0.95), run_id="t"
    )
    assert d.action is RouteAction.CONTINUE_SINGLE_AGENT
    assert d.decided_by == "mode"


def test_mode_gating_never_blocks_safety_or_stopping():
    """A downgrade must not swallow a REQUEST_HUMAN or a STOP."""
    for kw, expected in (
        ({"unauthorized_action": "payment"}, RouteAction.REQUEST_HUMAN),
        ({"budget_exhausted": True}, RouteAction.STOP),
        ({"requires_external_input": True}, RouteAction.REQUEST_HUMAN),
    ):
        d = AdaptiveRouter().decide(ctx(mode=Mode.GOAL_ONLY, **kw), run_id="t")
        assert d.action is expected


def test_adaptive_and_tournament_modes_permit_everything():
    for mode in (Mode.ADAPTIVE, Mode.TOURNAMENT):
        assert MODE_ALLOWED[mode] == frozenset(RouteAction)


# --------------------------------------------------------------------------
# Auditability  (spec 4.2)
# --------------------------------------------------------------------------


def test_every_decision_records_rule_trigger_and_predicted_cost():
    d = AdaptiveRouter().decide(ctx(complexity_score=0.9), run_id="run42")
    assert d.run_id == "run42"
    assert d.decision_id.startswith("run42-r")
    assert d.rule_fired
    assert d.rationale
    assert d.trigger and "mode=adaptive" in d.trigger
    assert d.predicted_cost_units == ACTION_COST[RouteAction.SPAWN_SUBAGENT]
    assert d.at


def test_decision_ids_are_sequential_within_a_router():
    router = AdaptiveRouter()
    ids = [router.decide(ctx(), run_id="t").decision_id for _ in range(3)]
    assert ids == ["t-r001", "t-r002", "t-r003"]


def test_a_model_suggestion_is_recorded_but_never_decides():
    """Rules win; the disagreement is kept so routing can be reviewed."""
    d = AdaptiveRouter().decide(
        ctx(
            real_evidence_count=0,  # rule: must gather evidence
            model_suggestion=RouteAction.RUN_PROPOSAL_TOURNAMENT,
            model_rationale="the candidates look interesting",
        ),
        run_id="t",
    )
    assert d.action is RouteAction.GATHER_EVIDENCE
    assert d.decided_by == "rule"
    assert d.model_suggestion is RouteAction.RUN_PROPOSAL_TOURNAMENT
    assert d.model_rationale
    assert d.overrode_model is True


def test_agreement_with_the_model_is_not_flagged_as_an_override():
    d = AdaptiveRouter().decide(
        ctx(real_evidence_count=0, model_suggestion=RouteAction.GATHER_EVIDENCE),
        run_id="t",
    )
    assert d.overrode_model is False


def test_no_model_suggestion_is_fine():
    d = AdaptiveRouter().decide(ctx(), run_id="t")
    assert d.model_suggestion is None
    assert d.overrode_model is False


def test_realised_outcome_can_be_attached_after_the_fact():
    d = AdaptiveRouter().decide(ctx(real_evidence_count=0), run_id="t")
    assert d.realised_new_evidence is None

    updated = record_outcome(d, new_evidence=4, maturity_gain=False)
    assert updated.realised_new_evidence == 4
    assert updated.realised_maturity_gain is False
    # The original record is not mutated; history stays append-only.
    assert d.realised_new_evidence is None


def test_every_action_has_a_cost_estimate():
    assert set(ACTION_COST) == set(RouteAction)
    assert ACTION_COST[RouteAction.RUN_PROPOSAL_TOURNAMENT] > ACTION_COST[
        RouteAction.CONTINUE_SINGLE_AGENT
    ]
    # Asking a human costs the harness nothing; it costs the human.
    assert ACTION_COST[RouteAction.REQUEST_HUMAN] == 0.0


def test_routing_is_deterministic_for_identical_signals():
    a = AdaptiveRouter().decide(ctx(complexity_score=0.8), run_id="t")
    b = AdaptiveRouter().decide(ctx(complexity_score=0.8), run_id="t")
    assert (a.action, a.rule_fired) == (b.action, b.rule_fired)
