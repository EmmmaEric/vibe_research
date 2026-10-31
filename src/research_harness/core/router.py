"""Adaptive research router (spec section 4.2).

This replaces the fixed CONTRACT -> RESEARCH -> PROPOSALS -> ... pipeline. The
spec forbids treating one sequence of stages as the right answer for every
goal, so the next action is *decided* each iteration rather than looked up in a
phase table.

**One decision point.** The router is the only component that chooses what
happens next. :class:`~research_harness.agents.diagnosis.DiagnosisAgent` is kept
as a *signal provider* -- it attributes a failure to a cause -- but it no longer
picks the action, because two components choosing the next step is how a control
flow becomes untraceable.

**Rules decide; the model advises.** The spec rules out routing on an opaque
model opinion. Every decision is made by a named, ordered rule that reads
explicit signals in :class:`RouterContext`. A model suggestion, when available,
is recorded alongside -- including when it disagreed and was overridden -- so
routing quality is reviewable. The rule order below is an honest heuristic
baseline, not a claim of optimality; ``realised_*`` fields on each decision
exist so it can be assessed against outcomes later.

**Mode gating** (spec section 16) is applied first: ``goal_only`` structurally
cannot reach the multi-agent actions, which is what makes it a real baseline
rather than the full system with a flag set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.schemas import Mode, RouteAction, RoutingDecision, utcnow


@dataclass
class RouterContext:
    """Every signal the rules may read. Explicit, so routing is testable.

    Deliberately plain data: the router must be decidable without a model, a
    network call, or access to the orchestrator's internals.
    """

    mode: Mode
    iteration: int

    # -- hard stops -----------------------------------------------------
    budget_exhausted: bool = False
    budget_detail: str = ""
    unauthorized_action: str | None = None
    pending_approval_stage: str | None = None

    # -- outcome so far -------------------------------------------------
    goal_satisfied: bool | None = None
    implementation_conforms: bool | None = None
    requires_external_input: bool = False
    failure_cause: str | None = None  # from the diagnosis agent

    # -- evidence -------------------------------------------------------
    real_evidence_count: int = 0
    critical_unknowns: int = 0
    evidence_sources_exhausted: bool = False

    # -- candidates -----------------------------------------------------
    proposal_count: int = 0
    scored: bool = False
    ranking_unstable: bool = False
    score_gap_top_two: float | None = None
    untried_candidates: int = 0
    tournament_ran: bool = False

    # -- experiments ----------------------------------------------------
    test_plan_ready: bool = False
    experiment_ran: bool = False
    cheap_decisive_test_available: bool = False
    repair_attempts: int = 0
    max_repair_attempts: int = 2

    # -- requested depth ------------------------------------------------
    #: True when the run has reached the maturity the requester asked for.
    #: Asking only for research must not silently run experiments.
    depth_reached: bool = False
    depth_detail: str = ""

    # -- progress -------------------------------------------------------
    consecutive_no_gain: int = 0
    max_consecutive_no_gain: int = 2

    # -- complexity -----------------------------------------------------
    complexity_score: float = 0.0  # 0..1
    expert_review_needed: bool = False
    #: True once a specialist has reviewed the current candidate set. High
    #: complexity alone would otherwise re-fire the spawn rule every iteration,
    #: re-reviewing the same candidates -- which a real run did twice.
    specialist_review_done: bool = False

    #: Advisory only. Never decides on its own.
    model_suggestion: RouteAction | None = None
    model_rationale: str = ""

    def tradeoffs_high(self, threshold: float = 8.0) -> bool:
        """True when the leading candidates are close enough to be contested."""
        return self.score_gap_top_two is not None and self.score_gap_top_two < threshold

    def critical_evidence_missing(self) -> bool:
        return self.real_evidence_count == 0 or self.critical_unknowns > 0


#: Cost estimate per action, in abstract units, for budgeting and for the
#: `predicted_cost_units` audit field. Rough by design: its job is to make
#: "this route is expensive" visible, not to be accurate.
ACTION_COST: dict[RouteAction, float] = {
    RouteAction.CONTINUE_SINGLE_AGENT: 1.0,
    RouteAction.GATHER_EVIDENCE: 1.5,
    RouteAction.SPAWN_SUBAGENT: 2.5,
    RouteAction.RUN_PROPOSAL_TOURNAMENT: 4.0,
    RouteAction.RUN_EXPERIMENT: 3.0,
    RouteAction.REQUEST_HUMAN: 0.0,
    RouteAction.STOP: 0.0,
}

#: Actions a mode is allowed to take. ``goal_only`` is a genuine baseline: it
#: cannot spawn helpers or run a tournament at all.
MODE_ALLOWED: dict[Mode, frozenset[RouteAction]] = {
    Mode.GOAL_ONLY: frozenset(
        {
            RouteAction.CONTINUE_SINGLE_AGENT,
            RouteAction.GATHER_EVIDENCE,
            RouteAction.RUN_EXPERIMENT,
            RouteAction.REQUEST_HUMAN,
            RouteAction.STOP,
        }
    ),
    Mode.ADAPTIVE: frozenset(RouteAction),
    Mode.TOURNAMENT: frozenset(RouteAction),
}


@dataclass
class _Rule:
    rule_id: str
    action: RouteAction
    rationale: str
    decided_by: str = "rule"


class AdaptiveRouter:
    """Chooses the next research action from explicit signals."""

    def __init__(self, *, tradeoff_threshold: float = 8.0) -> None:
        self.tradeoff_threshold = tradeoff_threshold
        self._counter = 0

    # ------------------------------------------------------------------

    def decide(self, ctx: RouterContext, *, run_id: str) -> RoutingDecision:
        rule = self._apply_rules(ctx)
        allowed = MODE_ALLOWED[ctx.mode]

        decided_by = rule.decided_by
        rationale = rule.rationale
        rule_id = rule.rule_id
        action = rule.action

        if action not in allowed:
            # Mode gating is a downgrade, never a block: a goal_only run that
            # would have run a tournament keeps working with its single agent.
            fallback = (
                RouteAction.CONTINUE_SINGLE_AGENT
                if RouteAction.CONTINUE_SINGLE_AGENT in allowed
                else RouteAction.STOP
            )
            rationale = (
                f"Rule '{rule_id}' selected {action.value}, which mode "
                f"'{ctx.mode.value}' does not permit; downgraded to "
                f"{fallback.value}. {rule.rationale}"
            )
            rule_id = f"{rule_id}+mode_gate"
            decided_by = "mode"
            action = fallback

        self._counter += 1
        return RoutingDecision(
            decision_id=f"{run_id}-r{self._counter:03d}",
            run_id=run_id,
            iteration=ctx.iteration,
            action=action,
            decided_by=decided_by,  # type: ignore[arg-type]
            rule_fired=rule_id,
            model_suggestion=ctx.model_suggestion,
            model_rationale=ctx.model_rationale,
            trigger=self._trigger(ctx),
            rationale=rationale,
            predicted_cost_units=ACTION_COST[action],
            at=utcnow(),
        )

    # ------------------------------------------------------------------

    def _apply_rules(self, ctx: RouterContext) -> _Rule:
        """Ordered rule table. First match wins; order encodes precedence.

        Safety and authorisation come before everything, then budget, then
        whether we are already done, then information-gathering, then strategy.
        Cheap and decisive beats expensive and thorough.
        """

        # 1. Safety / authorisation. Never routed around by a later rule, and
        #    never satisfiable by another agent recommending it.
        if ctx.unauthorized_action:
            return _Rule(
                "unauthorized_action",
                RouteAction.REQUEST_HUMAN,
                f"'{ctx.unauthorized_action}' needs explicit authorisation before "
                "anything else can proceed.",
            )
        if ctx.pending_approval_stage:
            return _Rule(
                "approval_pending",
                RouteAction.REQUEST_HUMAN,
                f"Stage '{ctx.pending_approval_stage}' is gated and awaiting approval.",
            )

        # 2. Budget. Checked before 'are we done' so an exhausted run reports
        #    its budget state rather than silently stopping for another reason.
        if ctx.budget_exhausted:
            return _Rule(
                "budget_exhausted",
                RouteAction.STOP,
                f"Budget exhausted: {ctx.budget_detail or 'no allowance remains'}.",
                decided_by="budget",
            )

        # 3. Already achieved: stop rather than spend more looking for comfort.
        if ctx.goal_satisfied is True:
            return _Rule(
                "goal_satisfied",
                RouteAction.STOP,
                "The goal is independently validated as met; further work would add "
                "cost without information.",
            )

        # 3b. The requester asked for less than full validation and we got there.
        if ctx.depth_reached:
            return _Rule(
                "requested_depth_reached",
                RouteAction.STOP,
                ctx.depth_detail
                or "The requested research depth has been reached; going further was "
                "not asked for.",
            )

        # 4. Blocked on the real world. More internal iteration cannot resolve it.
        if ctx.requires_external_input:
            return _Rule(
                "requires_external_input",
                RouteAction.REQUEST_HUMAN,
                "Validation needs real users, devices or physical measurement that "
                "this harness cannot obtain; simulating it would produce no "
                "information.",
            )

        # 5. Stagnation: several rounds with no gain means the current line is
        #    spent, whatever it looked like locally.
        if ctx.consecutive_no_gain >= ctx.max_consecutive_no_gain:
            return _Rule(
                "stagnated",
                RouteAction.STOP,
                f"{ctx.consecutive_no_gain} consecutive iteration(s) produced no "
                "measurable gain; continuing would burn budget without new "
                "information.",
            )

        # 6. Implementation defect with repair budget: fix and retest. This is
        #    cheap and does not touch the research question.
        if (
            ctx.implementation_conforms is False
            and ctx.repair_attempts < ctx.max_repair_attempts
        ):
            return _Rule(
                "implementation_defect",
                RouteAction.RUN_EXPERIMENT,
                f"The artefact does not match its own specification; repair attempt "
                f"{ctx.repair_attempts + 1} of {ctx.max_repair_attempts}. The "
                "research question is untouched.",
            )

        # 7. No evidence at all, or a named critical unknown. Everything
        #    downstream rests on this, so it comes before designing candidates.
        if ctx.critical_evidence_missing() and not ctx.evidence_sources_exhausted:
            reason = (
                "no non-simulated evidence has been gathered"
                if ctx.real_evidence_count == 0
                else f"{ctx.critical_unknowns} critical unknown(s) remain unverified"
            )
            return _Rule(
                "critical_evidence_missing",
                RouteAction.GATHER_EVIDENCE,
                f"Routing to evidence gathering because {reason}; scoring candidates "
                "on absent evidence would produce confident noise.",
            )

        # 8. A cheap decisive test beats more discussion.
        if ctx.cheap_decisive_test_available and ctx.test_plan_ready:
            return _Rule(
                "cheap_decisive_test",
                RouteAction.RUN_EXPERIMENT,
                "A cheap falsifiable test is available; running it reduces "
                "uncertainty faster than further analysis.",
            )

        # 8b. Tournament mode was explicitly requested: compete candidates once,
        #     even if the absolute scores look settled. This is what makes the
        #     mode an ablation arm rather than a hint.
        if ctx.mode is Mode.TOURNAMENT and not ctx.tournament_ran:
            return _Rule(
                "tournament_mode_requested",
                RouteAction.RUN_PROPOSAL_TOURNAMENT,
                "Mode 'tournament' was requested explicitly, so candidates compete "
                "head to head regardless of the absolute score spread.",
                decided_by="mode",
            )

        # 9. Genuinely contested multi-candidate decision -> let them compete.
        if ctx.proposal_count > 1 and not ctx.tournament_ran and (
            ctx.ranking_unstable or ctx.tradeoffs_high(self.tradeoff_threshold)
        ):
            why = (
                "the ranking is unstable under small weight changes"
                if ctx.ranking_unstable
                else f"the top two candidates are within "
                f"{ctx.score_gap_top_two:.1f} points"
            )
            return _Rule(
                "contested_candidates",
                RouteAction.RUN_PROPOSAL_TOURNAMENT,
                f"Running a head-to-head tournament because {why}, so the absolute "
                "scores alone do not settle the choice.",
            )

        # 10. Correctly built but the goal is not met, and alternatives exist.
        if (
            ctx.goal_satisfied is False
            and ctx.implementation_conforms is not False
            and ctx.untried_candidates > 0
        ):
            return _Rule(
                "goal_missed_alternatives_remain",
                RouteAction.RUN_PROPOSAL_TOURNAMENT,
                "The candidate was built correctly but missed the goal, and "
                f"{ctx.untried_candidates} untried candidate(s) remain; re-competing "
                "them is the remaining source of new information.",
            )

        # 11. Hard problem or specialist judgement needed -> bring in help, but
        #     only once per candidate set.
        if not ctx.specialist_review_done and (
            ctx.expert_review_needed or ctx.complexity_score >= 0.7
        ):
            why = (
                "specialist review was requested"
                if ctx.expert_review_needed
                else f"estimated complexity {ctx.complexity_score:.2f} is high"
            )
            return _Rule(
                "needs_specialist",
                RouteAction.SPAWN_SUBAGENT,
                f"Spawning a focused subagent because {why}.",
            )

        # 12. Plan exists and nothing has been run yet -> run it.
        if ctx.test_plan_ready and not ctx.experiment_ran:
            return _Rule(
                "plan_ready_untested",
                RouteAction.RUN_EXPERIMENT,
                "A verification plan exists and has not been executed; an untested "
                "design supports no empirical claim.",
            )

        # 13. Nothing more to do and nothing left to try.
        if ctx.experiment_ran and ctx.untried_candidates == 0 and (
            ctx.evidence_sources_exhausted or ctx.goal_satisfied is False
        ):
            return _Rule(
                "options_exhausted",
                RouteAction.STOP,
                "Every candidate and evidence avenue available has been tried; "
                "reporting the unresolved question is the correct outcome.",
            )

        # 14. Default: keep the general agent working. The spec's single-agent
        #     preference is the fallback, not a special case.
        return _Rule(
            "default_single_agent",
            RouteAction.CONTINUE_SINGLE_AGENT,
            "No condition calls for extra machinery; the general agent continues.",
        )

    @staticmethod
    def _trigger(ctx: RouterContext) -> str:
        """Compact snapshot of the signals, for the audit record."""
        return (
            f"mode={ctx.mode.value} iter={ctx.iteration} "
            f"evidence={ctx.real_evidence_count} unknowns={ctx.critical_unknowns} "
            f"proposals={ctx.proposal_count} untried={ctx.untried_candidates} "
            f"unstable={ctx.ranking_unstable} gap={ctx.score_gap_top_two} "
            f"plan={ctx.test_plan_ready} ran={ctx.experiment_ran} "
            f"conforms={ctx.implementation_conforms} goal={ctx.goal_satisfied} "
            f"no_gain={ctx.consecutive_no_gain} repairs={ctx.repair_attempts}"
        )


def record_outcome(
    decision: RoutingDecision,
    *,
    new_evidence: int,
    maturity_gain: bool,
) -> RoutingDecision:
    """Attach the realised effect of a routing decision.

    The spec asks for predicted cost *and* post-hoc utility, so the heuristic
    table can eventually be assessed rather than trusted. This is the only place
    that writes the ``realised_*`` fields.
    """
    return decision.model_copy(
        update={
            "realised_new_evidence": new_evidence,
            "realised_maturity_gain": maturity_gain,
        }
    )
