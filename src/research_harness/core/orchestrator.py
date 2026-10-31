"""Orchestrator: a router-driven research loop.

The v2 spec forbids a fixed stage sequence, so there is no phase table here.
Each iteration the orchestrator assembles the current signals into a
:class:`RouterContext`, asks the :class:`AdaptiveRouter` what to do, executes
exactly that one action, records what it achieved, and loops.

Division of responsibility:

* **Router** decides *what happens next*, from explicit signals and named rules.
* **General agent / subagents** do open-ended research and design.
* **Deterministic code here and in ``evaluation/``** owns every gate that could
  be gamed: constraint filtering, scoring arithmetic, maturity assignment,
  budget, approval, and test-plan custody.
* **Diagnosis** attributes failures to causes; it no longer picks actions,
  because two components choosing the next step makes control flow untraceable.

Side effects are guarded by idempotency keys in the event log, so resuming a run
cannot re-run an experiment that already completed.
"""

from __future__ import annotations

import uuid

from ..agents.diagnosis import DiagnosisAgent
from ..agents.executor import Executor
from ..agents.general import GeneralResearchAgent
from ..agents.goal_analyst import GoalAnalyst
from ..agents.proposal_designer import ProposalDesigner
from ..agents.researcher import Researcher
from ..agents.skeptic import Skeptic
from ..agents.test_designer import TestDesigner
from ..agents.validator import IndependentValidator
from ..evaluation import constraints as constraint_mod
from ..evaluation.decision_engine import (
    build_ranking,
    scope_qualified_claim,
    select_winner,
    should_stop,
)
from ..evaluation.judge_adapter import JudgeAdapter
from ..evaluation.mcda import evaluate_proposal, rescore
from ..evaluation.next_action import select_next_action
from ..evaluation.pairwise import run_pairwise
from ..evaluation.rubric import build_rubric
from ..evaluation.stability import relabel_invariance, weight_sensitivity
from ..experiments.adapters import ExperimentAdapter
from ..experiments.planner import TestPlanRegistry
from ..memory.events import EventLog, idempotency_key
from ..memory.repository import Repository
from ..reports.generator import ReportGenerator
from .budget import ArtifactStore, BudgetTracker, environment_fingerprint
from .loops import ImplementationRepairLoop
from .router import AdaptiveRouter, RouterContext, record_outcome
from .schemas import (
    ApprovalPolicy,
    Budget,
    ConstraintStatus,
    DecisionRecord,
    EventType,
    MaturityLevel,
    Mode,
    Origin,
    Phase,
    RouteAction,
    RunOutcome,
    StopDecision,
    utcnow,
)
from .state import ResearchState, RunProvenance

#: Hard ceiling on router iterations. A backstop against a routing cycle,
#: independent of the research budget.
MAX_ROUTER_STEPS = 60

_DEPTH_ORDER = {
    MaturityLevel.R1_RESEARCHED: 1,
    MaturityLevel.R2_DESIGNED: 2,
    MaturityLevel.R3_PROTOTYPED: 3,
    MaturityLevel.R4_VALIDATED: 4,
    MaturityLevel.R5_GOAL_ACHIEVED: 5,
}


class ApprovalRequired(RuntimeError):
    def __init__(self, stage: str, run_id: str) -> None:
        super().__init__(
            f"stage '{stage}' requires approval for run {run_id}. Approve with: "
            f"research-harness approve {run_id} --stage {stage}"
        )
        self.stage = stage
        self.run_id = run_id


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:10]}"


class Orchestrator:
    def __init__(
        self,
        *,
        provider,
        judge: JudgeAdapter,
        search_tools: list,
        experiment_adapter: ExperimentAdapter,
        store: ArtifactStore,
        repository: Repository | None = None,
        seed: int = 0,
        search_notes: list[str] | None = None,
        router: AdaptiveRouter | None = None,
    ) -> None:
        self.provider = provider
        self.judge = judge
        self.search_tools = search_tools
        self.adapter = experiment_adapter
        self.store = store.ensure()
        self.repo = repository
        self.seed = seed
        self.search_notes = search_notes or []

        self.registry = TestPlanRegistry()
        self.router = router or AdaptiveRouter()
        self.goal_analyst = GoalAnalyst(provider)
        self.general = GeneralResearchAgent(provider)
        self.researcher = Researcher(provider, search_tools)
        self.designer = ProposalDesigner(provider)
        self.skeptic = Skeptic(provider)
        self.test_designer = TestDesigner(provider)
        self.executor = Executor(self.registry)
        self.validator = IndependentValidator(self.registry)
        self.diagnoser = DiagnosisAgent()
        self.repair_loop = ImplementationRepairLoop(self.executor)
        self.reporter = ReportGenerator()
        self.tracker = BudgetTracker(provider)
        self.log: EventLog | None = None

    # ==================================================================
    # Entry points
    # ==================================================================

    def start(
        self,
        *,
        idea: str,
        run_id: str | None = None,
        mode: Mode = Mode.ADAPTIVE,
        budget: Budget | None = None,
        approval_policy: ApprovalPolicy | None = None,
        desired_maturity_level: MaturityLevel = MaturityLevel.R3_PROTOTYPED,
        domain_hint: str | None = None,
        set_metrics: dict[str, float] | None = None,
        remove_metrics: list[str] | None = None,
    ) -> ResearchState:
        rid = run_id or new_run_id()
        env = environment_fingerprint()
        state = ResearchState(
            run_id=rid,
            mode=mode,
            provenance=RunProvenance(
                provider_id=self.provider.provider_id,
                judge_id=getattr(self.judge, "judge_id", "unknown"),
                search_tool_ids=[t.tool_id for t in self.search_tools],
                experiment_adapter_id=self.adapter.adapter_id,
                seed=self.seed,
                python_version=env["python_version"],
                platform=env["platform"],
                is_simulated_provider=bool(getattr(self.provider, "is_simulated", True)),
            ),
        )
        self.log = EventLog(rid, self.store.root / "events.jsonl")
        state.add_notes(self.search_notes)
        self._honesty_warnings(state)

        contract = self.goal_analyst.build_contract(
            run_id=rid,
            idea=idea,
            budget=budget,
            approval_policy=approval_policy,
            desired_maturity_level=desired_maturity_level,
            domain_hint=domain_hint,
        )
        if set_metrics or remove_metrics:
            contract = self._apply_metric_overrides(
                contract, set_metrics or {}, remove_metrics or [], state
            )
        self._validate_metric_aliases(contract)
        state.set_contract(contract)
        self.store.contract.write_text(
            contract.model_dump_json(indent=2), encoding="utf-8"
        )
        if self.repo:
            self.repo.save_contract(contract)
        self.log.append(
            EventType.GOAL_DEFINED,
            f"Contract v{contract.version} established for mode '{mode.value}'.",
            payload={
                "goal_status": contract.goal_status,
                "metrics": [m.key for m in contract.success_metrics],
                "desired_maturity": desired_maturity_level.value,
            },
        )
        state.set_phase(Phase.RUNNING)
        self._persist(state)
        return self.resume(state)

    def deepen(
        self, state: ResearchState, level: MaturityLevel
    ) -> ResearchState:
        """Raise a run's requested maturity and let it continue.

        This is what makes the two-phase flow usable with a real model. Metric
        names are emergent, so you cannot bind an experiment's output to a
        contract metric before the contract exists -- but starting a fresh run
        regenerates the contract with *different* names. Deepening keeps the
        contract that is already fixed and versioned, so the keys you just read
        stay valid.

        Asking for deeper verification never weakens the goal, but it is still a
        contract change, so it is recorded as a revision with a reason.
        """
        contract = state.required_contract
        current = _DEPTH_ORDER.get(contract.desired_maturity_level, 3)
        wanted = _DEPTH_ORDER.get(level, 3)
        if wanted < current:
            raise ValueError(
                f"refusing to lower requested maturity from "
                f"{contract.desired_maturity_level.value} to {level.value}: "
                "asking for less verification after seeing results is how a goal "
                "gets quietly weakened. Start a new run instead."
            )
        if wanted == current:
            return state

        revised = contract.revise(
            reason=(
                f"requested maturity raised from "
                f"{contract.desired_maturity_level.value} to {level.value} by the "
                "operator, to continue the same contract rather than regenerate it"
            ),
            author="user",
            changes={"desired_maturity_level": level.value},
        )
        self._validate_metric_aliases(revised)
        state.set_contract(revised)
        self.store.contract.write_text(
            revised.model_dump_json(indent=2), encoding="utf-8"
        )
        if self.repo:
            self.repo.save_contract(revised)
        state.add_note(
            f"Requested maturity raised to {level.value} (contract v{revised.version}); "
            "the original idea and metric keys are unchanged."
        )
        # The run had stopped on 'requested_depth_reached'; reopen it.
        state.stop_decision = None
        state.outcome = None
        state.budget_state.consecutive_no_gain = 0
        state.evidence_exhausted = False
        state.set_phase(Phase.RUNNING)
        self._persist(state)
        return self.resume(state)

    def _validate_metric_aliases(self, contract) -> None:
        """Fail loudly when a metric binding names a metric that does not exist.

        A stale or misspelled ``--map-metric`` key used to bind nothing and the
        run would finish reporting "no contract metric was measured" -- true,
        but it looks like the experiment's fault rather than the mapping's. The
        error lists the contract's actual metric keys, which is also what a
        caller needs: a real model names metrics from the goal, so they differ
        between runs and cannot be known before the contract exists.
        """
        aliases = getattr(self.adapter, "metric_aliases", None) or {}
        if not aliases:
            return
        declared = {m.key for m in contract.success_metrics}
        unknown = sorted(set(aliases) - declared)
        if unknown:
            raise ValueError(
                f"--map-metric names contract metric(s) {unknown}, which this run's "
                f"contract does not declare. Declared metrics: {sorted(declared)}. "
                "A real model names metrics from the goal rather than from a "
                "template, so they differ between runs: inspect the contract "
                "(--depth R1_RESEARCHED, or research_contract.json) and map against "
                "its actual keys."
            )

    def _refresh_provenance(self, state: ResearchState) -> None:
        """Record the backends actually in use, and note any change.

        Provenance was captured once at ``start()``, so a resumed run that was
        given different backends reported the *original* ones -- a report
        claiming "SIMULATED adapter" while a real pytest run produced the
        measurement. Wrong in the safe direction there, but the inverse is the
        same bug and would be the dangerous one, so provenance is re-recorded
        every time the loop is entered and a change is written into the notes.
        """
        p = state.provenance
        now = {
            "provider_id": self.provider.provider_id,
            "judge_id": getattr(self.judge, "judge_id", "unknown"),
            "experiment_adapter_id": self.adapter.adapter_id,
            "search_tool_ids": [t.tool_id for t in self.search_tools],
            "is_simulated_provider": bool(getattr(self.provider, "is_simulated", True)),
        }
        changed = {
            field: (getattr(p, field), value)
            for field, value in now.items()
            if getattr(p, field) != value
        }
        if not changed:
            return
        for field, value in now.items():
            setattr(p, field, value)
        state.add_note(
            "Backends changed on resume: "
            + "; ".join(
                f"{field} {before!r} -> {after!r}" for field, (before, after) in changed.items()
            )
            + ". Results produced before and after this point came from different "
            "backends, which matters for reproducing them."
        )

    def resume(self, state: ResearchState) -> ResearchState:
        """Run the router loop until a terminal state."""
        self._refresh_provenance(state)
        if self.log is None or self.log.run_id != state.run_id:
            # Restoring the key set is what keeps resume from repeating a
            # completed side effect.
            self.log = EventLog.restore(
                state.run_id, state.events, self.store.root / "events.jsonl"
            )
        if state.phase in (Phase.DONE, Phase.STOPPED):
            return state
        if (
            state.phase is Phase.AWAITING_APPROVAL
            and state.awaiting_approval_stage
            and state.awaiting_approval_stage not in state.approved_stages
        ):
            # `resume` must not be a way around the gate. Approval is granted
            # through `approve()`, which records who granted it.
            return state
        state.set_phase(Phase.RUNNING)

        for _ in range(MAX_ROUTER_STEPS):
            decision = self.router.decide(self._context(state), run_id=state.run_id)
            state.routing_decisions.append(decision)
            self.log.append(
                EventType.ROUTE_DECIDED,
                f"[{decision.action.value}] {decision.rationale}",
                payload={
                    "rule": decision.rule_fired,
                    "decided_by": decision.decided_by,
                    "overrode_model": decision.overrode_model,
                    "predicted_cost": decision.predicted_cost_units,
                },
            )

            before_evidence = len(state.evidence)
            before_level = state.final_maturity_level

            try:
                terminal = self._execute(state, decision.action)
            except ApprovalRequired as exc:
                state.awaiting_approval_stage = exc.stage
                state.set_phase(Phase.AWAITING_APPROVAL)
                state.outcome = RunOutcome.HUMAN_ACTION_REQUIRED
                state.add_note(str(exc))
                self.log.append(EventType.RUN_STOPPED, str(exc))
                state.events = list(self.log.events)
                self._persist(state)
                return state
            except Exception as exc:  # a failed action must not lose the run
                state.add_warning(
                    f"Action {decision.action.value} failed: "
                    f"{type(exc).__name__}: {exc}. The run is stopped with state "
                    "preserved; nothing is fabricated to paper over the failure."
                )
                state.outcome = RunOutcome.BLOCKED
                state.stop_decision = StopDecision(
                    should_stop=True, reason="action_error", detail=str(exc)
                )
                return self._finish(state, Phase.STOPPED)

            state.routing_decisions[-1] = record_outcome(
                decision,
                new_evidence=len(state.evidence) - before_evidence,
                maturity_gain=state.final_maturity_level != before_level,
            )
            if not terminal:
                # STOP / REQUEST_HUMAN close the run; counting them would report a
                # budget as overspent by one.
                state.iteration += 1
                state.budget_state.iterations_used = state.iteration
            self.tracker.sync(state)
            state.events = list(self.log.events)
            self._persist(state)

            if terminal:
                if state.phase is Phase.AWAITING_APPROVAL:
                    state.events = list(self.log.events)
                    self._persist(state)
                    return state
                good = state.outcome in (
                    RunOutcome.GOAL_ACHIEVED,
                    RunOutcome.BUDGET_EXHAUSTED,
                    None,
                )
                return self._finish(state, Phase.DONE if good else Phase.STOPPED)

        state.add_warning(
            f"Router step ceiling ({MAX_ROUTER_STEPS}) reached; stopping to avoid a "
            "cycle. This indicates a routing bug, not a research conclusion."
        )
        state.outcome = RunOutcome.BLOCKED
        return self._finish(state, Phase.STOPPED)

    def approve(self, state: ResearchState, stage: str) -> ResearchState:
        if stage not in state.approved_stages:
            state.approved_stages.append(stage)
        state.awaiting_approval_stage = None
        contract = state.required_contract
        if self.log is None:
            self.log = EventLog.restore(
                state.run_id, state.events, self.store.root / "events.jsonl"
            )
        self.log.append(
            EventType.HUMAN_APPROVAL,
            f"Stage '{stage}' approved by the operator.",
            payload={"always_gated": stage in contract.approval_policy.always_gated},
        )
        if stage in contract.approval_policy.always_gated:
            state.add_warning(
                f"Stage '{stage}' has real-world effect and was approved explicitly "
                f"at {utcnow()}."
            )
        state.events = list(self.log.events)
        state.set_phase(Phase.RUNNING)
        self._persist(state)
        return self.resume(state)

    # ==================================================================
    # Router context
    # ==================================================================

    def _context(self, state: ResearchState) -> RouterContext:
        contract = state.required_contract
        b = state.budget_state
        validation = state.latest_validation()

        stop = should_stop(
            contract=contract,
            iteration=state.iteration,
            elapsed_seconds=b.elapsed_seconds,
            tokens_used=b.tokens_used,
            cost_usd=b.cost_usd,
            consecutive_no_gain=0,  # stagnation is its own router rule
            repair_attempts=b.repair_attempts,
            goal_satisfied=None,  # the router reads this separately
            candidates_remaining=1,
        )

        pending = None
        if (
            contract.approval_policy.needs_approval("experiment")
            and "experiment" not in state.approved_stages
            and state.selected_proposal_id
            and not state.experiments
        ):
            pending = "experiment"

        depth_reached, depth_detail = self._depth_reached(state)

        return RouterContext(
            mode=state.mode,
            iteration=state.iteration,
            budget_exhausted=stop.should_stop,
            budget_detail=stop.detail if stop.should_stop else "",
            pending_approval_stage=pending,
            goal_satisfied=validation.goal_satisfied if validation else None,
            implementation_conforms=(
                validation.implementation_conforms if validation else None
            ),
            requires_external_input=bool(
                validation
                and validation.external_validation_plan
                and validation.maturity_level is MaturityLevel.INCONCLUSIVE
            ),
            real_evidence_count=sum(1 for e in state.evidence if e.is_real),
            critical_unknowns=len(state.critical_unknowns),
            evidence_sources_exhausted=state.evidence_exhausted,
            proposal_count=len(state.proposals),
            scored=bool(state.evaluations),
            ranking_unstable=bool(
                state.ranking
                and state.ranking.stability
                and not state.ranking.stability.top1_stable
            ),
            score_gap_top_two=state.score_gap_top_two(),
            untried_candidates=len(state.remaining_candidates()),
            tournament_ran=state.tournament_ran,
            test_plan_ready=bool(state.selected_proposal_id),
            experiment_ran=bool(state.experiments),
            cheap_decisive_test_available=self._cheap_test(state),
            repair_attempts=b.repair_attempts,
            max_repair_attempts=contract.budget.max_repair_attempts,
            consecutive_no_gain=b.consecutive_no_gain,
            max_consecutive_no_gain=contract.budget.max_consecutive_no_gain,
            complexity_score=self._complexity(state),
            expert_review_needed=self._expert_needed(state),
            specialist_review_done=self._reviewed(state),
            depth_reached=depth_reached,
            depth_detail=depth_detail,
        )

    def _depth_reached(self, state: ResearchState) -> tuple[bool, str]:
        want = _DEPTH_ORDER.get(state.required_contract.desired_maturity_level, 3)
        if want <= 1 and state.evidence:
            return True, (
                "Only R1_RESEARCHED was requested and sourced research is complete; "
                "no candidate was designed or tested because that was not asked for."
            )
        if want <= 2 and state.selected_proposal_id and state.test_plans:
            return True, (
                "R2_DESIGNED was requested: a candidate and a verification plan "
                "exist, and no experiment was run because that was not asked for."
            )
        return False, ""

    def _cheap_test(self, state: ResearchState) -> bool:
        if not state.selected_proposal_id or state.experiments:
            return False
        try:
            proposal = state.proposal(state.selected_proposal_id)
        except KeyError:
            return False
        return proposal.poc.estimated_cost_units <= 2.0

    def _complexity(self, state: ResearchState) -> float:
        if not state.general_findings:
            return 0.0
        contract = state.required_contract
        concerns = (
            len(contract.success_metrics)
            + len(contract.hard_constraints)
            + len(state.critical_unknowns)
        )
        return min(1.0, round(concerns / 12.0, 3))

    @staticmethod
    def _reviewed(state: ResearchState) -> bool:
        """True when a specialist has already critiqued every current candidate."""
        if not state.proposals:
            return False
        reviewed = {c.proposal_id for c in state.critiques}
        return {p.proposal_id for p in state.proposals} <= reviewed

    def _expert_needed(self, state: ResearchState) -> bool:
        if state.critiques:
            return False  # a skeptic has already reviewed
        return any(
            c.category in ("legal", "safety")
            for c in state.required_contract.hard_constraints
        )

    # ==================================================================
    # Action handlers
    # ==================================================================

    def _execute(self, state: ResearchState, action: RouteAction) -> bool:
        """Run one routed action. Returns True when the run should end."""
        handlers = {
            RouteAction.CONTINUE_SINGLE_AGENT: self._act_general,
            RouteAction.GATHER_EVIDENCE: self._act_gather,
            RouteAction.SPAWN_SUBAGENT: self._act_subagent,
            RouteAction.RUN_PROPOSAL_TOURNAMENT: self._act_tournament,
            RouteAction.RUN_EXPERIMENT: self._act_experiment,
            RouteAction.REQUEST_HUMAN: self._act_request_human,
            RouteAction.STOP: self._act_stop,
        }
        return handlers[action](state)

    def _act_general(self, state: ResearchState) -> bool:
        contract = state.required_contract
        if not state.evidence:
            self._gather(state, targeted=False)

        # Snapshot so a step that contributes nothing is recorded as no-gain.
        # Without this the router keeps defaulting to the general agent and the
        # run spins until the budget dies -- the pointless self-loop the spec
        # warns about, which looks like work and produces none.
        before = (
            len(state.evidence),
            len(state.proposals),
            len(state.evaluations),
            tuple(state.critical_unknowns),
            state.selected_proposal_id,
        )

        report = self.general.solve(contract=contract, evidence=state.evidence)
        state.general_findings.append(report.findings)
        state.critical_unknowns = list(report.critical_unknowns)
        state.task_graph = list(report.task_graph)
        state.add_note(f"[general agent] {report.findings}")
        for c in report.counterexamples[:3]:
            state.add_note(f"[counterexample] {c}")

        self.log.append(
            EventType.PLAN_CREATED,
            f"General agent decomposed the goal into {len(report.task_graph)} "
            f"question(s) with {len(report.critical_unknowns)} critical unknown(s).",
            payload={
                "complexity": report.complexity_score,
                "cheap_test": report.cheap_test_available,
                "suggested": report.suggested_next_action.value
                if report.suggested_next_action
                else None,
            },
        )

        # One candidate in the baseline; the full archetype set otherwise.
        if not state.proposals:
            self._ensure_proposals(state, single=state.mode is Mode.GOAL_ONLY)
        if state.proposals and not state.evaluations:
            self._score(state)

        after = (
            len(state.evidence),
            len(state.proposals),
            len(state.evaluations),
            tuple(state.critical_unknowns),
            state.selected_proposal_id,
        )
        if after == before:
            state.budget_state.consecutive_no_gain += 1
            state.add_note(
                "The general agent had nothing new to add: no new evidence, "
                "candidate, score or open question. Counted as no progress so the "
                "run stops rather than re-deriving the same conclusion."
            )
        else:
            state.budget_state.consecutive_no_gain = 0
        return False

    def _act_gather(self, state: ResearchState) -> bool:
        before = len(state.evidence)
        target = self._information_target(state)
        self._gather(state, targeted=bool(target), assumption=target)
        added = state.evidence[before:]
        gained = len(added)
        gained_real = sum(1 for e in added if e.is_real)

        if gained == 0:
            state.budget_state.consecutive_no_gain += 1
            state.evidence_exhausted = True
            state.add_note(
                "Evidence gathering produced nothing new; the configured sources are "
                "exhausted for this question."
            )
        elif gained_real == 0:
            # Simulated items are not progress toward grounding a claim, so this
            # counts as no gain. Otherwise a fixture-only run would re-gather
            # forever while `real_evidence_count` stayed at zero.
            state.budget_state.consecutive_no_gain += 1
            state.evidence_exhausted = True
            state.add_note(
                f"Evidence gathering added {gained} item(s), none of them "
                "non-simulated. The configured sources cannot ground this question, "
                "so retrieval is treated as exhausted rather than retried."
            )
        else:
            state.budget_state.consecutive_no_gain = 0
            if state.evaluations:
                # Scores cited the old pool; reusing them against a changed
                # evidence base would silently misattribute the evidence.
                state.evaluations = []
                state.add_note(
                    f"{gained} new evidence item(s) arrived; previous scores were "
                    "discarded rather than reused against a changed evidence base."
                )
        if state.has_real_evidence():
            state.critical_unknowns = [
                u
                for u in state.critical_unknowns
                if "No non-simulated evidence" not in u
            ]
        if state.proposals and not state.evaluations:
            self._score(state)
        return False

    def _act_subagent(self, state: ResearchState) -> bool:
        contract = state.required_contract
        if not state.proposals:
            self._ensure_proposals(state, single=False)
        state.critiques = self.skeptic.critique_all(
            contract=contract, proposals=state.proposals, evidence=state.evidence
        )
        worst = max((c.severity for c in state.critiques), default=0)
        state.add_note(
            f"[subagent: skeptic] Reviewed {len(state.critiques)} candidate(s); "
            f"highest severity {worst}/5."
        )
        self.log.append(
            EventType.PROPOSAL_REVISED,
            f"Skeptic review of {len(state.critiques)} candidate(s).",
            payload={"max_severity": worst},
        )
        state.evaluations = []
        self._score(state)
        return False

    def _act_tournament(self, state: ResearchState) -> bool:
        contract = state.required_contract
        if len(state.proposals) < 2:
            self._ensure_proposals(state, single=False)
        if not state.evaluations:
            self._score(state)

        rubric = state.required_rubric
        pairwise = run_pairwise(
            judge=self.judge,
            contract=contract,
            rubric=rubric,
            proposals=state.proposals,
            evidence=state.evidence,
        )
        if pairwise.position_bias_rate > 0:
            state.add_warning(
                f"Judge position bias in {pairwise.position_bias_rate:.0%} of "
                "A/B-swapped pairs; the pairwise ordering is unreliable and is used "
                "only as a tie-break."
            )
        state.ranking = build_ranking(
            run_id=state.run_id,
            rubric_version=rubric.version,
            evaluations=state.evaluations,
            top_k=min(3, len(state.proposals)),
            pairwise=pairwise,
            stability=weight_sensitivity(rubric, state.evaluations),
        )
        state.tournament_ran = True

        winner = self._select_untried(state)
        if winner:
            state.selected_proposal_id = winner
            state.add_note(scope_qualified_claim(state.ranking, winner))
        self.log.append(
            EventType.SCORE_RECORDED,
            f"Tournament over {len(state.proposals)} candidate(s); selected {winner}.",
            payload={
                "consistency": pairwise.consistency_rate,
                "position_bias": pairwise.position_bias_rate,
                "top_k": state.ranking.top_k,
            },
        )
        self._write_evaluation_artifacts(state)
        return False

    def _act_experiment(self, state: ResearchState) -> bool:
        contract = state.required_contract
        if not state.selected_proposal_id:
            if not state.proposals:
                self._ensure_proposals(state, single=state.mode is Mode.GOAL_ONLY)
            if not state.evaluations:
                self._score(state)
            state.selected_proposal_id = self._select_untried(state)
        pid = state.selected_proposal_id
        if pid is None:
            state.add_warning(
                "No eligible candidate remains to test; every one is excluded by a "
                "hard constraint or has already failed."
            )
            state.outcome = RunOutcome.PROPOSAL_INVALID
            return True

        stage = "experiment"
        if (
            contract.approval_policy.needs_approval(stage)
            and stage not in state.approved_stages
        ):
            raise ApprovalRequired(stage, state.run_id)

        plan = self._ensure_test_plan(state, pid)

        validation = state.latest_validation()
        repairing = (
            validation is not None
            and validation.proposal_id == pid
            and validation.implementation_conforms is False
        )

        index = len(state.experiments) + 1
        key = idempotency_key(
            "experiment", state.run_id, pid, plan.content_hash(), index
        )
        if self.log.guard(key, action=f"experiment {index} for {pid}"):
            return True

        workdir = self.store.experiment_dir(index)
        if repairing:
            from ..agents.diagnosis import Action, Diagnosis

            diagnosis = Diagnosis(
                action=Action.REPAIR_IMPLEMENTATION,
                loop="implementation",
                cause="implementation_defect",
                hypothesis=(
                    f"Conformance failure(s) {', '.join(validation.failed_test_ids)} "
                    "are an implementation defect, not a refuted approach."
                ),
                expected_result="Failing conformance cases pass with no regression.",
                rationale="Routed by rule 'implementation_defect'.",
            )
            outcome = self.repair_loop.attempt(
                state=state,
                diagnosis=diagnosis,
                adapter=self.adapter,
                plan=plan,
                workdir=workdir,
                experiment_index=index,
            )
            state.add_note(f"[Loop A] {outcome.detail}")
            state.budget_state.consecutive_no_gain = (
                0 if outcome.progressed else state.budget_state.consecutive_no_gain + 1
            )
            experiment = outcome.experiment
        else:
            experiment = self.executor.run(
                adapter=self.adapter,
                contract=contract,
                proposal=state.proposal(pid),
                plan=plan,
                workdir=workdir,
                experiment_id=f"{state.run_id}-exp{index:03d}",
            )
            state.experiments.append(experiment)

        self.log.append(
            EventType.EXPERIMENT_RUN,
            f"Experiment {experiment.experiment_id} via '{experiment.adapter_id}' "
            f"(simulated={experiment.is_simulated}, executed={experiment.executed}).",
            payload={"error": experiment.error},
            key=key,
        )
        if self.repo:
            self.repo.save_experiment(experiment)
        if experiment.error:
            state.add_note(f"Experiment {experiment.experiment_id}: {experiment.error}")

        self._validate(state, pid, plan, experiment)
        return False

    def _validate(self, state: ResearchState, pid, plan, experiment) -> None:
        contract = state.required_contract
        ev = state.evaluation(pid)
        status = ev.constraint_status if ev else ConstraintStatus.UNKNOWN
        validation = self.validator.validate(
            contract=contract,
            proposal=state.proposal(pid),
            plan=plan,
            experiment=experiment,
            constraint_status=status,
            evidence=state.evidence,
        )
        state.validations.append(validation)
        state.final_maturity_level = validation.maturity_level
        if self.repo:
            self.repo.save_validation(validation)

        self.log.append(
            EventType.VALIDATION_RECORDED,
            f"{pid}: maturity={validation.maturity_level.value}, "
            f"conforms={validation.implementation_conforms}, "
            f"goal={validation.goal_satisfied}.",
            payload={"reason": validation.blocking_reason},
        )
        if validation.goal_satisfied is True:
            self.log.append(
                EventType.GOAL_VERIFIED,
                f"Original goal independently validated as met by {pid}.",
            )
        state.add_note(
            f"Independent validation of {pid}: {validation.maturity_level.value}; "
            f"conforms={validation.implementation_conforms}; "
            f"goal={validation.goal_satisfied}. {validation.blocking_reason}"
        )

        # Diagnosis attributes the cause; the router decides what to do about it.
        remaining = [p for p in state.remaining_candidates() if p != pid]
        diagnosis = self.diagnoser.diagnose(
            validation=validation,
            repair_attempts=state.budget_state.repair_attempts,
            max_repair_attempts=contract.budget.max_repair_attempts,
            remaining_candidates=remaining,
            has_real_evidence=state.has_real_evidence(),
            evidence_exhausted=state.evidence_exhausted,
        )
        state.record_decision(
            DecisionRecord(
                decision_id=f"{state.run_id}-d{len(state.decisions) + 1:03d}",
                run_id=state.run_id,
                iteration=state.iteration,
                loop=diagnosis.loop,
                hypothesis=diagnosis.hypothesis,
                action="attributed_cause",
                expected_result=diagnosis.expected_result,
                actual_result=f"cause={diagnosis.cause}",
                divergence_cause=diagnosis.cause,
                continue_reason=diagnosis.rationale,
            )
        )
        state.add_note(f"[diagnosis] cause='{diagnosis.cause}'. {diagnosis.rationale}")

        # A candidate that was built correctly but missed the goal is spent:
        # retiring it is what lets the router move to a different mechanism.
        if (
            validation.goal_satisfied is False
            and validation.implementation_conforms is not False
            and pid not in state.tried_proposal_ids
        ):
            state.tried_proposal_ids.append(pid)
            state.selected_proposal_id = None

    def _act_request_human(self, state: ResearchState) -> bool:
        # A gated stage is a *pause*: the operator can approve and the run
        # continues. That is different from needing real-world validation, which
        # ends the run. Conflating them would either lose a resumable run or
        # report a merely-paused one as finished.
        if state.awaiting_approval_stage is None:
            contract = state.required_contract
            if (
                contract.approval_policy.needs_approval("experiment")
                and "experiment" not in state.approved_stages
                and state.selected_proposal_id
                and not state.experiments
            ):
                state.awaiting_approval_stage = "experiment"
                state.outcome = RunOutcome.HUMAN_ACTION_REQUIRED
                state.add_note(
                    "Paused: stage 'experiment' is gated and needs explicit approval "
                    f"before anything runs. Approve with: research-harness approve "
                    f"{state.run_id} --stage experiment"
                )
                self.log.append(
                    EventType.RUN_STOPPED,
                    "Paused awaiting approval for stage 'experiment'.",
                )
                state.set_phase(Phase.AWAITING_APPROVAL)
                return True

        validation = state.latest_validation()
        if validation and validation.external_validation_plan:
            state.outcome = RunOutcome.REQUIRES_EXTERNAL_TEST
            state.stop_decision = StopDecision(
                should_stop=True,
                reason="requires_external_test",
                detail=validation.blocking_reason,
            )
        else:
            state.outcome = RunOutcome.HUMAN_ACTION_REQUIRED
            state.stop_decision = StopDecision(
                should_stop=True,
                reason="human_action_required",
                detail="An action requiring explicit authorisation is pending.",
            )
        return True

    def _act_stop(self, state: ResearchState) -> bool:
        validation = state.latest_validation()
        last = state.last_route()
        rule = (last.rule_fired or "") if last else ""

        if validation and validation.goal_satisfied is True:
            outcome, reason = RunOutcome.GOAL_ACHIEVED, "goal_satisfied"
        elif rule.startswith("budget_exhausted"):
            outcome, reason = RunOutcome.BUDGET_EXHAUSTED, "budget_exhausted"
        elif rule.startswith("stagnated"):
            outcome, reason = RunOutcome.STAGNATED, "no_progress"
        elif rule.startswith("requested_depth_reached"):
            outcome, reason = RunOutcome.BUDGET_EXHAUSTED, "requested_depth_reached"
        elif state.ranking is not None and not state.ranking.top_k:
            outcome, reason = RunOutcome.PROPOSAL_INVALID, "no_viable_candidates"
        elif validation and validation.goal_satisfied is False:
            outcome, reason = RunOutcome.PROPOSAL_INVALID, "goal_not_met"
        elif not state.has_real_evidence():
            outcome, reason = RunOutcome.EVIDENCE_INSUFFICIENT, "no_real_evidence"
        else:
            outcome, reason = RunOutcome.EVIDENCE_INSUFFICIENT, "options_exhausted"

        state.outcome = outcome
        state.stop_decision = StopDecision(
            should_stop=True,
            reason=reason,
            detail=(last.rationale if last else "Run complete."),
        )
        if state.final_maturity_level is None:
            state.final_maturity_level = self._maturity_without_experiment(state)
        return True

    def _maturity_without_experiment(self, state: ResearchState) -> MaturityLevel:
        """Honest level for a run that never executed anything."""
        if state.test_plans and state.selected_proposal_id:
            return MaturityLevel.R2_DESIGNED
        if state.has_real_evidence():
            return MaturityLevel.R1_RESEARCHED
        return MaturityLevel.INCONCLUSIVE

    # ==================================================================
    # Shared steps
    # ==================================================================

    def _gather(
        self, state: ResearchState, *, targeted: bool, assumption: str | None = None
    ) -> None:
        contract = state.required_contract
        before = len(state.evidence)
        if targeted and assumption:
            evidence, notes = self.researcher.gather_targeted(contract, assumption)
            state.add_note(
                f'Retrieval targeted the highest-value open question: "{assumption[:140]}".'
            )
        else:
            evidence, notes = self.researcher.gather(contract, limit=12)

        known = {e.claim for e in state.evidence}
        added = [e for e in evidence if e.claim not in known]
        for i, e in enumerate(added, start=before + 1):
            data = e.model_dump()
            data["evidence_id"] = f"E{i:03d}"
            state.evidence.append(type(e).model_validate(data))
        state.add_notes(notes)

        with self.store.evidence.open("w", encoding="utf-8") as fh:
            for e in state.evidence:
                fh.write(e.model_dump_json() + "\n")
        if self.repo:
            self.repo.save_evidence(state.run_id, state.evidence)
        self.log.append(
            EventType.SOURCE_RETRIEVED,
            f"Retrieved {len(added)} new item(s) "
            f"({sum(1 for e in added if e.is_real)} non-simulated).",
            payload={"targeted": targeted},
        )

    def _information_target(self, state: ResearchState) -> str | None:
        if state.critical_unknowns:
            return state.critical_unknowns[0]
        if state.ranking and state.evaluations:
            plan = select_next_action(
                proposals=state.proposals,
                evaluations=state.evaluations,
                ranking=state.ranking,
            )
            if plan.experiments:
                return plan.experiments[0].targets_assumption
        return None

    def _ensure_proposals(self, state: ResearchState, *, single: bool) -> None:
        contract = state.required_contract
        archetypes = ["balanced"] if single else None
        proposals, warnings = self.designer.design_all(
            contract=contract, evidence=state.evidence, archetypes=archetypes
        )
        state.proposals = proposals
        if not single:
            for w in warnings:
                state.add_warning(w)
        for p in proposals:
            self.store.proposal(p.proposal_id).write_text(
                p.model_dump_json(indent=2), encoding="utf-8"
            )
        if self.repo:
            self.repo.save_proposals(proposals)
        self.log.append(
            EventType.PROPOSAL_CREATED,
            f"Designed {len(proposals)} candidate(s) "
            f"({'single-agent baseline' if single else 'independent archetypes'}).",
        )

    def _score(self, state: ResearchState) -> None:
        contract = state.required_contract
        rubric = state.rubric or build_rubric(contract, version="v1")
        if state.rubric is None:
            state.set_rubric(rubric)
            self.log.append(
                EventType.RUBRIC_CREATED,
                f"Rubric {rubric.version} derived from contract v{contract.version} "
                f"with {len(rubric.dimensions)} dimension(s).",
            )

        evaluations = []
        for proposal in state.proposals:
            scoring = self.judge.score_rubric(
                goal=contract, rubric=rubric, proposal=proposal, evidence=state.evidence
            )
            assessments = constraint_mod.ensure_complete(
                contract, scoring.constraint_assessments
            )
            critique = state.critique_for(proposal.proposal_id)
            extra = list(scoring.unresolved_assumptions)
            if critique:
                extra += critique.unstated_dependencies[:2]
            evaluations.append(
                evaluate_proposal(
                    run_id=state.run_id,
                    proposal_id=proposal.proposal_id,
                    rubric=rubric,
                    dimension_scores=scoring.dimension_scores,
                    constraint_status=constraint_mod.combine(assessments),
                    constraint_assessments=assessments,
                    judge_disagreement=scoring.judge_disagreement,
                    unresolved_assumptions=list(dict.fromkeys(extra)),
                    recommended_next_test=scoring.recommended_next_test,
                )
            )
        state.evaluations = evaluations

        ok, detail = relabel_invariance(rubric, evaluations)
        if not ok:
            state.add_warning(f"Relabel-invariance check FAILED: {detail}")
        state.ranking = build_ranking(
            run_id=state.run_id,
            rubric_version=rubric.version,
            evaluations=evaluations,
            top_k=min(3, max(1, len(state.proposals))),
            pairwise=state.ranking.pairwise if state.ranking else None,
            stability=weight_sensitivity(rubric, evaluations),
        )
        if state.ranking.excluded_for_constraints:
            state.add_note(
                "Excluded for hard-constraint violation (non-compensatory): "
                f"{', '.join(state.ranking.excluded_for_constraints)}."
            )
        # Re-selection, not just first-selection. Re-scoring exists *because* the
        # ranking can move (a skeptic review changes the inputs), so keeping a
        # stale pick throws away the result: a real run experimented on the
        # rank-3 candidate while the report showed rank 1 as the leader.
        #
        # The exception is a candidate that has already been tested: switching
        # then would discard real work, and retiring it is the router's call via
        # `tried_proposal_ids`.
        current = state.selected_proposal_id
        already_tested = any(e.proposal_id == current for e in state.experiments)
        if current is None or not already_tested:
            reselected = self._select_untried(state)
            if reselected != current and current is not None:
                state.add_note(
                    f"Re-scoring moved the ranking: selection changed from {current} "
                    f"to {reselected}. Nothing had been tested on {current} yet, so "
                    "no work is discarded."
                )
            state.selected_proposal_id = reselected
        if state.selected_proposal_id:
            self._ensure_test_plan(state, state.selected_proposal_id)
        self.log.append(
            EventType.SCORE_RECORDED,
            f"Scored {len(evaluations)} candidate(s) against rubric {rubric.version}.",
            payload={"excluded": state.ranking.excluded_for_constraints},
        )
        self._write_evaluation_artifacts(state)

    def _ensure_test_plan(self, state: ResearchState, pid: str):
        """Author and hash-register the verification plan for a candidate.

        Called as soon as a candidate is selected, not when an experiment is
        about to run: that is what makes "test-first" structural rather than a
        convention, and it is why an R2_DESIGNED run has a plan without having
        executed anything.
        """
        existing = state.plan_for(pid)
        if existing is not None:
            self.registry.register(existing)
            return existing

        plan = self.test_designer.design(
            contract=state.required_contract, proposal=state.proposal(pid)
        )
        self.registry.register(plan)
        state.test_plans.append(plan)
        self.store.test_plan(pid, plan.version).write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )
        if self.repo:
            self.repo.save_test_plan(plan)
        self.log.append(
            EventType.EXPERIMENT_PLANNED,
            f"Test plan {plan.plan_id} registered (hash {plan.content_hash()}) from "
            f"contract v{plan.derived_from_contract_version}, before execution. The "
            "executor cannot amend it.",
        )
        return plan

    def _select_untried(self, state: ResearchState) -> str | None:
        """Highest-ranked eligible candidate that has not already failed."""
        if state.ranking is None:
            return None
        for entry in sorted(state.ranking.entries, key=lambda e: e.rank):
            if entry.eligible and entry.proposal_id not in state.tried_proposal_ids:
                return entry.proposal_id
        return None

    # ==================================================================
    # Finish, artifacts, persistence
    # ==================================================================

    def _finish(self, state: ResearchState, phase: Phase) -> ResearchState:
        self.tracker.sync(state)
        if state.final_maturity_level is None:
            state.final_maturity_level = self._maturity_without_experiment(state)
        if state.outcome is None:
            state.outcome = RunOutcome.EVIDENCE_INSUFFICIENT

        self.log.append(
            EventType.RUN_STOPPED,
            f"Run ended: outcome={state.outcome.value}, "
            f"maturity={state.final_maturity_level.value}.",
            payload={
                "reason": state.stop_decision.reason if state.stop_decision else None
            },
        )
        state.events = list(self.log.events)

        with self.store.decisions.open("w", encoding="utf-8") as fh:
            for d in state.decisions:
                fh.write(d.model_dump_json() + "\n")
        with (self.store.root / "routing_decisions.jsonl").open(
            "w", encoding="utf-8"
        ) as fh:
            for d in state.routing_decisions:
                fh.write(d.model_dump_json() + "\n")
        if self.repo:
            self.repo.save_decisions(state.decisions)
            self.repo.save_run(state)

        self.store.final_report.write_text(
            self.reporter.final_report(state, store=self.store), encoding="utf-8"
        )
        self.store.research_report.write_text(
            self.reporter.research_report(state), encoding="utf-8"
        )
        if state.ranking is not None:
            self.store.ranking_report.write_text(
                self.reporter.ranking_report(state), encoding="utf-8"
            )
        (self.store.root / "run_summary.json").write_text(
            self.reporter.run_summary(state).model_dump_json(indent=2), encoding="utf-8"
        )
        state.set_phase(phase)
        self._persist(state)
        return state

    def _honesty_warnings(self, state: ResearchState) -> None:
        if state.provenance.is_simulated_provider:
            state.add_warning(
                "This run used the SIMULATED provider. Research, proposals, critiques "
                "and judge scores came from deterministic local heuristics, not from a "
                "model reasoning about the problem. The pipeline is real; the content "
                "is a demonstration and is not research evidence."
            )
        # The experiment warning is NOT added here: at start() nothing has run,
        # and an adapter chosen now may not be the one that ends up producing
        # the results (a resumed run can be given a different one). It is
        # derived at report time from the experiments that actually happened.

    def _write_evaluation_artifacts(self, state: ResearchState) -> None:
        import json

        rubric = state.rubric
        if rubric is None:
            return
        self.store.evaluation_file("rubric.json").write_text(
            rubric.model_dump_json(indent=2), encoding="utf-8"
        )
        self.store.evaluation_file("scores.json").write_text(
            json.dumps(
                [e.model_dump(mode="json") for e in state.evaluations],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if state.ranking:
            self.store.evaluation_file("ranking.json").write_text(
                state.ranking.model_dump_json(indent=2), encoding="utf-8"
            )
            if state.ranking.pairwise:
                self.store.evaluation_file("pairwise.json").write_text(
                    state.ranking.pairwise.model_dump_json(indent=2), encoding="utf-8"
                )
            plan = select_next_action(
                proposals=state.proposals,
                evaluations=state.evaluations,
                ranking=state.ranking,
            )
            self.store.evaluation_file("next_action.json").write_text(
                plan.model_dump_json(indent=2), encoding="utf-8"
            )
        if self.repo:
            self.repo.save_rubric(rubric)
            self.repo.save_evaluations(state.evaluations)
            if state.ranking:
                self.repo.save_ranking(state.ranking)

    def _persist(self, state: ResearchState) -> None:
        self.tracker.sync(state)
        state.save(self.store.state)

    # ==================================================================
    # Weight changes
    # ==================================================================

    def reweight(
        self, state: ResearchState, weights: dict[str, float], *, reason: str
    ) -> ResearchState:
        old = state.required_rubric
        unknown = set(weights) - {d.key for d in old.dimensions}
        if unknown:
            raise ValueError(f"unknown dimension(s): {sorted(unknown)}")

        new_rubric = old.with_weights(
            weights, new_version=f"v{len(state.rubric_history) + 2}"
        )
        state.set_rubric(new_rubric)
        state.evaluations = [rescore(new_rubric, e) for e in state.evaluations]
        state.ranking = build_ranking(
            run_id=state.run_id,
            rubric_version=new_rubric.version,
            evaluations=state.evaluations,
            top_k=min(3, max(1, len(state.proposals))),
            pairwise=state.ranking.pairwise if state.ranking else None,
            stability=weight_sensitivity(new_rubric, state.evaluations),
        )
        winner = select_winner(state.ranking)
        state.selected_proposal_id = winner
        state.add_note(
            f"Weights changed ({reason}): rubric {old.version} -> "
            f"{new_rubric.version}. {scope_qualified_claim(state.ranking, winner)}"
        )
        self._write_evaluation_artifacts(state)
        if state.ranking:
            self.store.ranking_report.write_text(
                self.reporter.ranking_report(state), encoding="utf-8"
            )
        self._persist(state)
        return state

    # ==================================================================
    # Contract editing
    # ==================================================================

    @staticmethod
    def _apply_metric_overrides(
        contract, set_metrics: dict[str, float], remove_metrics: list[str], state
    ):
        """Apply user metric edits as a versioned, reasoned contract revision.

        Only harness-assumed metrics may be removed: dropping a goal the
        requester actually stated would change what success means.
        """
        keys = {m.key for m in contract.success_metrics}
        unknown = (set(set_metrics) | set(remove_metrics)) - keys
        if unknown:
            raise ValueError(
                f"unknown metric key(s) {sorted(unknown)}; declared metrics are "
                f"{sorted(keys)}"
            )
        contradictory = sorted(set(set_metrics) & set(remove_metrics))
        if contradictory:
            raise ValueError(
                f"metric(s) {contradictory} appear in both --set-metric and "
                "--remove-metric. Refusing to guess which you meant: setting a "
                "threshold and dropping the metric are opposite intentions."
            )
        protected = [
            k
            for k in remove_metrics
            if not next(m for m in contract.success_metrics if m.key == k).assumed
        ]
        if protected:
            raise ValueError(
                f"refusing to remove requester-stated metric(s) {protected}. Only "
                "metrics the harness assumed as placeholders may be dropped; removing "
                "a stated goal would change what success means."
            )

        reasons: list[str] = []
        metrics = []
        for m in contract.success_metrics:
            if m.key in remove_metrics:
                reasons.append(
                    f"removed assumed placeholder metric '{m.key}' (not applicable)"
                )
                continue
            if m.key in set_metrics:
                data = m.model_dump()
                old = data["target_threshold"]
                data["target_threshold"] = set_metrics[m.key]
                # Supplying a threshold makes it the requester's own goal.
                data["origin"] = Origin.USER_SPECIFIED.value
                metrics.append(type(m).model_validate(data))
                reasons.append(
                    f"set target for '{m.key}' to {set_metrics[m.key]} (was {old}); "
                    "now treated as requester-stated"
                )
            else:
                metrics.append(m)

        if not metrics:
            raise ValueError(
                "every success metric was removed; a contract with no metric cannot "
                "decide goal attainment"
            )
        revised = contract.revise(
            reason="user metric override at run start: " + "; ".join(reasons),
            author="user",
            changes={"success_metrics": [m.model_dump() for m in metrics]},
        )
        state.add_note(
            f"Contract revised to v{revised.version} by user: {'; '.join(reasons)}. "
            "The original idea is unchanged and the prior version is retained."
        )
        return revised
