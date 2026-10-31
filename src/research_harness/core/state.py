"""Research state: the single persisted object a run can be resumed from.

Everything the orchestrator needs to continue lives here as structured data --
not as a chat transcript. Resumption is therefore exact: load the JSON, and the
state machine continues from the recorded phase with the same contract, evidence,
scores and decision history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from .schemas import (
    Base,
    Critique,
    Mode,
    ResearchEvent,
    RoutingDecision,
    RunOutcome,
    TaskNode,
    DecisionRecord,
    Evidence,
    ExperimentResult,
    Phase,
    Proposal,
    ProposalEvaluation,
    Ranking,
    ResearchContract,
    Rubric,
    StopDecision,
    TestPlan,
    MaturityLevel,
    ValidationResult,
    utcnow,
)


class RunProvenance(Base):
    """What produced this run -- required for reproducibility."""

    provider_id: str = "unknown"
    judge_id: str = "unknown"
    search_tool_ids: list[str] = Field(default_factory=list)
    experiment_adapter_id: str = "unknown"
    harness_version: str = "0.1.0"
    prompt_version: str = "v1"
    seed: int = 0
    python_version: str = ""
    platform: str = ""
    started_at: str = Field(default_factory=utcnow)
    is_simulated_provider: bool = True


class BudgetState(Base):
    iterations_used: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    llm_calls: int = 0
    repair_attempts: int = 0
    consecutive_no_gain: int = 0

    def remaining_text(self, contract: ResearchContract) -> str:
        b = contract.budget
        parts = [
            f"iterations {self.iterations_used}/{b.max_iterations}",
            f"repairs {self.repair_attempts}/{b.max_repair_attempts}",
        ]
        if b.max_tokens is not None:
            parts.append(f"tokens {self.tokens_used}/{b.max_tokens}")
        if b.max_cost_usd is not None:
            parts.append(f"cost ${self.cost_usd:.4f}/${b.max_cost_usd:.4f}")
        if b.max_wall_clock_seconds is not None:
            parts.append(
                f"wall clock {self.elapsed_seconds:.1f}s/{b.max_wall_clock_seconds}s"
            )
        return "; ".join(parts)


class ResearchState(Base):
    """Complete, serialisable state of one research run."""

    run_id: str
    phase: Phase = Phase.INIT
    #: Capability tier (spec section 16). ``goal_only`` is the baseline arm.
    mode: Mode = Mode.ADAPTIVE
    iteration: int = 0
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)

    contract: ResearchContract | None = None
    contract_history: list[ResearchContract] = Field(default_factory=list)

    evidence: list[Evidence] = Field(default_factory=list)
    proposals: list[Proposal] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)

    rubric: Rubric | None = None
    rubric_history: list[Rubric] = Field(default_factory=list)
    evaluations: list[ProposalEvaluation] = Field(default_factory=list)
    ranking: Ranking | None = None

    test_plans: list[TestPlan] = Field(default_factory=list)
    experiments: list[ExperimentResult] = Field(default_factory=list)
    validations: list[ValidationResult] = Field(default_factory=list)

    decisions: list[DecisionRecord] = Field(default_factory=list)
    selected_proposal_id: str | None = None
    tried_proposal_ids: list[str] = Field(default_factory=list)
    rejected_proposal_ids: list[str] = Field(default_factory=list)

    #: Router decisions, in order. The run's control flow is reconstructible
    #: from this list alone -- there is no implicit phase sequence to infer.
    routing_decisions: list[RoutingDecision] = Field(default_factory=list)
    #: Append-only audit log (spec section 12).
    events: list[ResearchEvent] = Field(default_factory=list)
    #: Problem decomposition from the general agent (spec section 4.3).
    task_graph: list[TaskNode] = Field(default_factory=list)
    general_findings: list[str] = Field(default_factory=list)
    critical_unknowns: list[str] = Field(default_factory=list)
    tournament_ran: bool = False
    #: Set once retrieval has demonstrably nothing left to give for this goal.
    #: Deliberately NOT derived from `consecutive_no_gain`: a repair that made
    #: progress would reset that counter and wrongly re-open retrieval.
    evidence_exhausted: bool = False

    stop_decision: StopDecision | None = None
    outcome: RunOutcome | None = None
    final_maturity_level: MaturityLevel | None = None
    awaiting_approval_stage: str | None = None
    approved_stages: list[str] = Field(default_factory=list)

    budget_state: BudgetState = Field(default_factory=BudgetState)
    provenance: RunProvenance = Field(default_factory=RunProvenance)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # -- convenience accessors ------------------------------------------

    @property
    def required_contract(self) -> ResearchContract:
        if self.contract is None:
            raise ValueError(f"run {self.run_id} has no contract yet")
        return self.contract

    @property
    def required_rubric(self) -> Rubric:
        if self.rubric is None:
            raise ValueError(f"run {self.run_id} has no rubric yet")
        return self.rubric

    def proposal(self, proposal_id: str) -> Proposal:
        for p in self.proposals:
            if p.proposal_id == proposal_id:
                return p
        raise KeyError(proposal_id)

    def evaluation(self, proposal_id: str) -> ProposalEvaluation | None:
        for e in self.evaluations:
            if e.proposal_id == proposal_id:
                return e
        return None

    def plan_for(self, proposal_id: str) -> TestPlan | None:
        """Latest-version plan for a candidate."""
        plans = [p for p in self.test_plans if p.proposal_id == proposal_id]
        return max(plans, key=lambda p: p.version) if plans else None

    def latest_validation(self) -> ValidationResult | None:
        return self.validations[-1] if self.validations else None

    def latest_experiment(self) -> ExperimentResult | None:
        return self.experiments[-1] if self.experiments else None

    def critique_for(self, proposal_id: str) -> Critique | None:
        for c in self.critiques:
            if c.proposal_id == proposal_id:
                return c
        return None

    def remaining_candidates(self) -> list[str]:
        """Eligible, ranked candidates not yet tried or rejected."""
        if self.ranking is None:
            return []
        out = []
        for e in sorted(self.ranking.entries, key=lambda x: x.rank):
            if not e.eligible:
                continue
            if e.proposal_id in self.tried_proposal_ids:
                continue
            if e.proposal_id in self.rejected_proposal_ids:
                continue
            out.append(e.proposal_id)
        return out

    def route_actions(self) -> list[str]:
        return [d.action.value for d in self.routing_decisions]

    def last_route(self) -> RoutingDecision | None:
        return self.routing_decisions[-1] if self.routing_decisions else None

    def score_gap_top_two(self) -> float | None:
        """Point-score gap between the two leading eligible candidates.

        None when fewer than two are eligible, which the router reads as
        'nothing to contest' rather than as a zero gap.
        """
        if self.ranking is None:
            return None
        eligible = [
            e for e in sorted(self.ranking.entries, key=lambda x: x.rank) if e.eligible
        ]
        if len(eligible) < 2:
            return None
        return round(abs(eligible[0].point_score - eligible[1].point_score), 4)

    def has_real_evidence(self) -> bool:
        return any(e.is_real for e in self.evidence)

    def next_evidence_id(self) -> str:
        return f"E{len(self.evidence) + 1:03d}"

    # -- mutation helpers ------------------------------------------------

    def touch(self) -> None:
        self.updated_at = utcnow()

    def set_phase(self, phase: Phase) -> None:
        self.phase = phase
        self.touch()

    def add_note(self, note: str) -> None:
        if note and note not in self.notes:
            self.notes.append(note)

    def add_notes(self, notes: list[str]) -> None:
        for n in notes:
            self.add_note(n)

    def add_warning(self, warning: str) -> None:
        if warning and warning not in self.warnings:
            self.warnings.append(warning)

    def record_decision(self, record: DecisionRecord) -> None:
        self.decisions.append(record)
        self.touch()

    def set_rubric(self, rubric: Rubric) -> None:
        """Install a rubric, archiving the previous version."""
        if self.rubric is not None and self.rubric.version != rubric.version:
            if not any(r.version == self.rubric.version for r in self.rubric_history):
                self.rubric_history.append(self.rubric)
        self.rubric = rubric
        self.touch()

    def set_contract(self, contract: ResearchContract) -> None:
        if self.contract is not None and self.contract.version != contract.version:
            self.contract_history.append(self.contract)
        self.contract = contract
        self.touch()

    # -- persistence -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted save cannot leave a truncated state.
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ResearchState":
        data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
