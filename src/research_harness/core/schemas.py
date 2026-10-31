"""Shared structured types for the research harness.

Everything that crosses a module boundary is a validated Pydantic model, never a
free-form dict and never a chat transcript. Deterministic engines (MCDA,
constraint gating, stop policy) consume and emit these; LLM/simulated providers
only ever fill in *semantic* fields.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class Base(BaseModel):
    """Strict base: unknown fields are an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Phase(str, enum.Enum):
    """Run lifecycle state.

    Deliberately not a pipeline. The v2 spec forbids a fixed stage sequence, so
    there is no PROPOSALS or EVALUATION phase to be "in" -- what happens next is
    decided each iteration by the adaptive router. These are the only states a
    run can be resumed from.
    """

    INIT = "INIT"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    DONE = "DONE"
    STOPPED = "STOPPED"


class MaturityLevel(str, enum.Enum):
    """How far verification actually got. Never inflate this."""

    R1_RESEARCHED = "R1_RESEARCHED"
    R2_DESIGNED = "R2_DESIGNED"
    R3_PROTOTYPED = "R3_PROTOTYPED"
    R4_VALIDATED = "R4_VALIDATED"
    R5_GOAL_ACHIEVED = "R5_GOAL_ACHIEVED"
    INCONCLUSIVE = "INCONCLUSIVE"
    REJECTED = "REJECTED"


class ConstraintStatus(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class Stance(str, enum.Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    UNCERTAIN = "uncertain"


class Provenance(str, enum.Enum):
    """Where a claim came from. Keeps 'source said' distinct from 'agent guessed'."""

    SOURCE = "source"  # a real retrieved document said this
    AGENT_INFERENCE = "agent_inference"  # a model reasoned to it
    MEASURED = "measured"  # a real executed experiment produced it
    SIMULATED = "simulated"  # synthetic/demo data, NOT real evidence
    HUMAN = "human"  # a person entered it


class MetricDirection(str, enum.Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Origin(str, enum.Enum):
    """Who put a goal, metric or constraint into the contract.

    Separating these is what lets a report distinguish the requester's actual
    goal from something the harness guessed on their behalf.
    """

    USER_SPECIFIED = "user_specified"
    AGENT_PROPOSED = "agent_proposed"
    ASSUMED = "assumed"


class GoalType(str, enum.Enum):
    """What kind of success the goal is asking for.

    Closed set, because it selects a scoring weight prior. Free text here would
    silently fall through to the default prior -- which is exactly what a real
    model did when this was a plain string.
    """

    TECHNICAL_PERFORMANCE = "technical_performance"
    SCIENTIFIC_VALIDATION = "scientific_validation"
    PRODUCT_UTILITY = "product_utility"
    MARKET_DEMAND = "market_demand"
    COST_REDUCTION = "cost_reduction"


class Mode(str, enum.Enum):
    """Capability tier a run executes under (spec section 16).

    ``GOAL_ONLY`` is the baseline every richer mode must justify itself against:
    one general agent plus tools plus independent validation.
    """

    GOAL_ONLY = "goal_only"
    ADAPTIVE = "adaptive"
    TOURNAMENT = "tournament"


class RouteAction(str, enum.Enum):
    """What the adaptive router decided to do next (spec section 4.2)."""

    CONTINUE_SINGLE_AGENT = "CONTINUE_SINGLE_AGENT"
    SPAWN_SUBAGENT = "SPAWN_SUBAGENT"
    RUN_PROPOSAL_TOURNAMENT = "RUN_PROPOSAL_TOURNAMENT"
    GATHER_EVIDENCE = "GATHER_EVIDENCE"
    RUN_EXPERIMENT = "RUN_EXPERIMENT"
    REQUEST_HUMAN = "REQUEST_HUMAN"
    STOP = "STOP"


class RunOutcome(str, enum.Enum):
    """Why a run ended (spec section 11.2).

    Distinct from :class:`MaturityLevel`, which says how far verification got.
    A run can stop for a budget reason at any maturity.
    """

    GOAL_ACHIEVED = "GOAL_ACHIEVED"
    IMPLEMENTATION_BUG = "IMPLEMENTATION_BUG"
    PROPOSAL_INVALID = "PROPOSAL_INVALID"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    REQUIRES_EXTERNAL_TEST = "REQUIRES_EXTERNAL_TEST"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    STAGNATED = "STAGNATED"
    UNSAFE_OR_UNAUTHORIZED = "UNSAFE_OR_UNAUTHORIZED"
    HUMAN_ACTION_REQUIRED = "HUMAN_ACTION_REQUIRED"
    BLOCKED = "BLOCKED"


class EventType(str, enum.Enum):
    """Immutable audit log event names (spec section 12)."""

    GOAL_DEFINED = "GOAL_DEFINED"
    PLAN_CREATED = "PLAN_CREATED"
    ROUTE_DECIDED = "ROUTE_DECIDED"
    SOURCE_RETRIEVED = "SOURCE_RETRIEVED"
    EVIDENCE_VERIFIED = "EVIDENCE_VERIFIED"
    PROPOSAL_CREATED = "PROPOSAL_CREATED"
    RUBRIC_CREATED = "RUBRIC_CREATED"
    SCORE_RECORDED = "SCORE_RECORDED"
    EXPERIMENT_PLANNED = "EXPERIMENT_PLANNED"
    EXPERIMENT_RUN = "EXPERIMENT_RUN"
    VALIDATION_RECORDED = "VALIDATION_RECORDED"
    PROPOSAL_REVISED = "PROPOSAL_REVISED"
    GOAL_VERIFIED = "GOAL_VERIFIED"
    RUN_STOPPED = "RUN_STOPPED"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"


class SideEffect(str, enum.Enum):
    """Blast radius of a tool or action (spec section 15)."""

    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"
    IRREVERSIBLE = "irreversible"


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


class Evidence(Base):
    evidence_id: str
    claim: str
    source_type: str  # paper | repo | spec | web | local_file | human | simulated
    source_uri: str | None = None
    retrieved_at: str = Field(default_factory=utcnow)
    applicability_conditions: list[str] = Field(default_factory=list)
    stance: Stance = Stance.UNCERTAIN
    provenance: Provenance = Provenance.AGENT_INFERENCE
    verification_status: Literal[
        "unverified", "source_checked", "cross_checked", "contradicted"
    ] = "unverified"
    limitations: list[str] = Field(default_factory=list)
    quality: int = Field(default=1, ge=0, le=5)
    related_metric: str | None = None

    @property
    def is_real(self) -> bool:
        """True only for evidence that is not synthetic."""
        return self.provenance != Provenance.SIMULATED


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


class SuccessMetric(Base):
    key: str
    description: str
    unit: str
    target_threshold: float | None = None
    direction: MetricDirection = MetricDirection.MAXIMIZE
    measurement_method: str = "unspecified"
    origin: Origin = Origin.ASSUMED
    #: A required metric must be met for the goal to count as achieved. An
    #: optional one is informative only -- it cannot block R5 on its own.
    required: bool = True
    #: Set when this stands in for a goal that is not yet directly measurable.
    is_proxy_for: str | None = None

    @property
    def assumed(self) -> bool:
        """True when the harness invented this rather than being told it."""
        return self.origin is Origin.ASSUMED

    def is_met(self, value: float) -> bool:
        """Deterministic threshold comparison. No model involved."""
        if self.target_threshold is None:
            return False
        if self.direction is MetricDirection.MAXIMIZE:
            return value >= self.target_threshold
        return value <= self.target_threshold


class HardConstraint(Base):
    key: str
    description: str
    category: str = "other"  # budget | legal | safety | resource | device | interface
    # A constraint is non-compensatory: no other dimension can buy it off.
    verification_hint: str = ""
    origin: Origin = Origin.ASSUMED
    # Optional machine-checkable form. When ``limit_value`` is set, the constraint
    # can be decided by arithmetic instead of judgement, which is always preferred.
    limit_metric: str | None = None  # e.g. "declared_cost_units"
    limit_value: float | None = None
    limit_direction: MetricDirection = MetricDirection.MINIMIZE

    @property
    def assumed(self) -> bool:
        return self.origin is Origin.ASSUMED

    def check_value(self, value: float) -> bool:
        """Deterministic satisfaction test for a machine-checkable constraint."""
        if self.limit_value is None:
            raise ValueError(f"constraint {self.key} has no numeric limit to check")
        if self.limit_direction is MetricDirection.MINIMIZE:
            return value <= self.limit_value
        return value >= self.limit_value


class SoftPreference(Base):
    key: str
    description: str
    importance: int = Field(default=3, ge=1, le=5)


class Budget(Base):
    max_iterations: int = Field(default=3, ge=1)
    max_tokens: int | None = None
    max_wall_clock_seconds: float | None = None
    max_repair_attempts: int = Field(default=2, ge=0)
    max_consecutive_no_gain: int = Field(default=2, ge=1)
    max_cost_usd: float | None = None


class ApprovalPolicy(Base):
    """Stages that require an explicit human go-ahead before running."""

    require_approval_for: list[str] = Field(default_factory=list)
    # Irreversible / real-world-cost actions are always gated regardless.
    always_gated: list[str] = Field(
        default_factory=lambda: [
            "real_device_control",
            "real_payment",
            "external_publication",
            "real_user_contact",
        ]
    )

    def needs_approval(self, stage: str) -> bool:
        return stage in self.require_approval_for or stage in self.always_gated


class ContractRevision(Base):
    version: int
    at: str = Field(default_factory=utcnow)
    reason: str
    author: str
    changed_fields: list[str] = Field(default_factory=list)


class ResearchContract(Base):
    """The frozen statement of what the user actually asked for.

    ``original_idea`` is never overwritten. Any change to goals/metrics/
    constraints bumps ``version`` and appends a ``ContractRevision``.
    """

    run_id: str
    original_idea: str
    problem_statement: str = ""
    scope_included: list[str] = Field(default_factory=list)
    scope_excluded: list[str] = Field(default_factory=list)
    goal_type: str = "unspecified"
    success_metrics: list[SuccessMetric] = Field(default_factory=list)
    hard_constraints: list[HardConstraint] = Field(default_factory=list)
    soft_preferences: list[SoftPreference] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    available_resources: list[str] = Field(default_factory=list)
    desired_maturity_level: MaturityLevel = MaturityLevel.R3_PROTOTYPED
    budget: Budget = Field(default_factory=Budget)
    approval_policy: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    domain_hint: str = "generic"
    #: ``provisional`` when the stated goal is not yet directly measurable and
    #: the metrics below are proxies. A provisional contract can never support a
    #: claim that the ORIGINAL goal was proven -- only that the proxy was met.
    goal_status: Literal["measurable", "provisional"] = "measurable"
    proxy_limitations: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)
    version: int = 1
    revisions: list[ContractRevision] = Field(default_factory=list)

    @property
    def required_metrics(self) -> list[SuccessMetric]:
        """Metrics that must be met for the goal to count as achieved."""
        return [m for m in self.success_metrics if m.required]

    @field_validator("original_idea")
    @classmethod
    def _idea_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("original_idea must not be empty")
        return v

    def revise(
        self, *, reason: str, author: str, changes: dict[str, Any]
    ) -> "ResearchContract":
        """Return a new contract version. ``original_idea`` is immutable."""
        if "original_idea" in changes:
            raise ValueError(
                "original_idea is immutable; the user's words are not editable by the system"
            )
        data = self.model_dump()
        data.update(changes)
        data["version"] = self.version + 1
        data["revisions"] = [*self.model_dump()["revisions"]] + [
            ContractRevision(
                version=self.version + 1,
                reason=reason,
                author=author,
                changed_fields=sorted(changes.keys()),
            ).model_dump()
        ]
        return ResearchContract.model_validate(data)


# --------------------------------------------------------------------------
# Proposals
# --------------------------------------------------------------------------


class PoCSketch(Base):
    description: str
    expected_result: str
    falsification_condition: str
    estimated_cost_units: float = Field(default=1.0, gt=0)


class Proposal(Base):
    """A candidate route. All candidates share this shape and a similar length
    budget so the judge is not biased by formatting."""

    proposal_id: str
    run_id: str
    title: str
    strategy_archetype: str  # conservative | balanced | exploratory | ...
    core_mechanism: str
    goal_mapping: str  # how it addresses the ORIGINAL goal
    dependencies: list[str] = Field(default_factory=list)
    premises: list[str] = Field(default_factory=list)
    key_steps: list[str] = Field(default_factory=list)
    resource_needs: list[str] = Field(default_factory=list)
    differentiators: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    critical_assumptions: list[str] = Field(default_factory=list)
    poc: PoCSketch
    distinction_from_others: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    declared_cost_units: float = Field(default=1.0, gt=0)
    author_role: str = "proposal_designer"
    generated_by: str = "unknown"  # provider id; SIMULATED providers say so
    version: int = 1
    created_at: str = Field(default_factory=utcnow)

    def card(self, max_items: int = 3) -> str:
        """Compressed, length-normalised, source-anonymised view for pairwise
        comparison. Deliberately omits ``generated_by`` and ``proposal_id``.

        The id is also redacted from the rendered text, so a designer that works
        its own identifier into a title or step cannot defeat the anonymisation.
        """

        def bullets(items: list[str]) -> str:
            sel = items[:max_items] or ["(none stated)"]
            return "; ".join(s.strip() for s in sel)

        text = (
            f"Title: {self.title}\n"
            f"Mechanism: {self.core_mechanism.strip()}\n"
            f"Goal mapping: {self.goal_mapping.strip()}\n"
            f"Key steps: {bullets(self.key_steps)}\n"
            f"Dependencies: {bullets(self.dependencies)}\n"
            f"Failure modes: {bullets(self.failure_modes)}\n"
            f"Critical assumptions: {bullets(self.critical_assumptions)}\n"
            f"Minimal PoC: {self.poc.description.strip()} "
            f"(falsified if: {self.poc.falsification_condition.strip()})\n"
            f"Declared cost units: {self.declared_cost_units}"
        )
        for token in (self.proposal_id, self.generated_by):
            if token and token.strip():
                text = text.replace(token, "[redacted]")
        return text


class Critique(Base):
    proposal_id: str
    counterexamples: list[str] = Field(default_factory=list)
    hidden_costs: list[str] = Field(default_factory=list)
    unstated_dependencies: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    severity: int = Field(default=2, ge=0, le=5)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Rubric / scoring
# --------------------------------------------------------------------------


class Dimension(Base):
    key: str
    name: str
    definition: str
    anchors: dict[int, str]  # 0..5 semantic anchors
    weight: float = Field(ge=0.0)
    acceptable_evidence_types: list[str] = Field(default_factory=list)
    higher_is_better: bool = True
    boundary_note: str = ""  # explicit de-overlap with neighbouring dimensions

    @field_validator("anchors")
    @classmethod
    def _anchor_keys(cls, v: dict[int, str]) -> dict[int, str]:
        missing = {0, 2, 5} - set(v)
        if missing:
            raise ValueError(f"anchors must at least define levels 0,2,5; missing {missing}")
        return v


class Rubric(Base):
    version: str
    run_id: str
    dimensions: list[Dimension]
    rationale: str = ""
    created_at: str = Field(default_factory=utcnow)
    derived_from_contract_version: int = 1

    @field_validator("dimensions")
    @classmethod
    def _nonempty_unique(cls, v: list[Dimension]) -> list[Dimension]:
        if not v:
            raise ValueError("rubric needs at least one dimension")
        keys = [d.key for d in v]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate dimension keys")
        if sum(d.weight for d in v) <= 0:
            raise ValueError("total weight must be positive")
        return v

    def normalized_weights(self) -> dict[str, float]:
        total = sum(d.weight for d in self.dimensions)
        return {d.key: d.weight / total for d in self.dimensions}

    def by_key(self, key: str) -> Dimension:
        for d in self.dimensions:
            if d.key == key:
                return d
        raise KeyError(key)

    def with_weights(self, weights: dict[str, float], new_version: str) -> "Rubric":
        data = self.model_dump()
        for dim in data["dimensions"]:
            if dim["key"] in weights:
                dim["weight"] = weights[dim["key"]]
        data["version"] = new_version
        return Rubric.model_validate(data)


class DimensionScore(Base):
    dimension_key: str
    raw: int | None = Field(default=None, ge=0, le=5)
    status: Literal["SCORED", "UNKNOWN"] = "SCORED"
    justification: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    score_source: Literal["llm_judge", "jev", "deterministic", "measured", "simulated"] = (
        "simulated"
    )

    @field_validator("status")
    @classmethod
    def _consistency(cls, v: str, info: Any) -> str:
        return v

    def model_post_init(self, __context: Any) -> None:
        if self.status == "SCORED" and self.raw is None:
            raise ValueError(f"dimension {self.dimension_key}: SCORED requires a raw value")
        if self.status == "UNKNOWN" and self.raw is not None:
            raise ValueError(
                f"dimension {self.dimension_key}: UNKNOWN must not carry a raw value "
                "(do not invent a number for missing evidence)"
            )


class ConstraintAssessment(Base):
    constraint_key: str
    status: ConstraintStatus
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    unverified_assumption: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.status is ConstraintStatus.UNKNOWN and not self.unverified_assumption:
            raise ValueError(
                f"constraint {self.constraint_key}: UNKNOWN must name the assumption "
                "that needs verification"
            )


class ProposalEvaluation(Base):
    """Deterministically computed from DimensionScores + Rubric. The judge never
    writes the aggregate number itself."""

    proposal_id: str
    run_id: str
    rubric_version: str
    constraint_status: ConstraintStatus
    constraint_assessments: list[ConstraintAssessment] = Field(default_factory=list)
    violated_constraints: list[str] = Field(default_factory=list)
    unverified_constraints: list[str] = Field(default_factory=list)
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    point_score: float = 0.0  # 0..100, over SCORED dims with renormalised weights
    score_lower: float = 0.0  # UNKNOWN dims treated as 0
    score_upper: float = 0.0  # UNKNOWN dims treated as 5
    evidence_sufficiency: float = 0.0  # share of weight actually backed by a score
    evidence_ids: list[str] = Field(default_factory=list)
    judge_disagreement: Literal["none", "low", "moderate", "high", "unknown"] = "unknown"
    unresolved_assumptions: list[str] = Field(default_factory=list)
    recommended_next_test: str = ""
    eligible: bool = True  # False iff a hard constraint FAILed
    computed_at: str = Field(default_factory=utcnow)

    @property
    def score_range(self) -> tuple[float, float]:
        return (self.score_lower, self.score_upper)


# --------------------------------------------------------------------------
# Pairwise
# --------------------------------------------------------------------------


PairVerdict = Literal["A", "B", "tie", "insufficient_evidence"]


class PairwiseJudgement(Base):
    """One raw judge call. ``verdict`` refers to *slot* A/B, not to a proposal id,
    so position bias stays measurable."""

    first_proposal_id: str
    second_proposal_id: str
    verdict: PairVerdict
    rationale: str = ""
    judge_id: str = "unknown"
    prompt_version: str = "v1"
    at: str = Field(default_factory=utcnow)

    @property
    def winner_id(self) -> str | None:
        if self.verdict == "A":
            return self.first_proposal_id
        if self.verdict == "B":
            return self.second_proposal_id
        return None


class PairOutcome(Base):
    """Aggregate of the two position-swapped judgements for one unordered pair."""

    proposal_x: str
    proposal_y: str
    forward: PairwiseJudgement
    reverse: PairwiseJudgement
    consistent: bool
    agreed_winner: str | None = None
    favored_first_slot_both_times: bool = False
    points_x: float = 0.0
    points_y: float = 0.0


class PairwiseReport(Base):
    run_id: str
    rubric_version: str
    outcomes: list[PairOutcome] = Field(default_factory=list)
    win_points: dict[str, float] = Field(default_factory=dict)
    comparisons: int = 0
    consistency_rate: float = 0.0
    position_bias_rate: float = 0.0
    insufficient_evidence_rate: float = 0.0
    note: str = ""


# --------------------------------------------------------------------------
# Ranking / stability
# --------------------------------------------------------------------------


class RankEntry(Base):
    proposal_id: str
    rank: int
    point_score: float
    score_lower: float
    score_upper: float
    constraint_status: ConstraintStatus
    eligible: bool
    pairwise_points: float = 0.0
    evidence_sufficiency: float = 0.0


class SensitivityFinding(Base):
    dimension_key: str
    delta: float
    new_top_proposal: str | None
    kendall_tau_vs_base: float
    top1_changed: bool


class StabilityReport(Base):
    run_id: str
    rubric_version: str
    base_ranking: list[str] = Field(default_factory=list)
    findings: list[SensitivityFinding] = Field(default_factory=list)
    top1_stable: bool = True
    min_kendall_tau: float = 1.0
    unstable_dimensions: list[str] = Field(default_factory=list)
    interpretation: str = ""


class Ranking(Base):
    run_id: str
    rubric_version: str
    entries: list[RankEntry] = Field(default_factory=list)
    excluded_for_constraints: list[str] = Field(default_factory=list)
    top_k: list[str] = Field(default_factory=list)
    stability: StabilityReport | None = None
    pairwise: PairwiseReport | None = None
    scope_caveat: str = (
        "Ranking is 'better among the candidates considered, under the current "
        "evidence, goals and weights'. It is not a claim of global optimality."
    )
    created_at: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Tests / experiments / validation
# --------------------------------------------------------------------------


TestCategory = Literal[
    "contract",
    "correctness",
    "edge_robustness",
    "comparison",
    "goal_acceptance",
    "evidence_quality",
]


class TestCase(Base):
    test_id: str
    category: TestCategory
    description: str
    procedure: str
    acceptance_criterion: str
    measures_metric: str | None = None
    baseline: str | None = None
    executable: bool = False
    requires_external_resource: str | None = None


class TestPlan(Base):
    """Authored by the Test Designer from the CONTRACT, not from the implementation.

    Versioned and hashed; the Executor is structurally forbidden from amending it
    (see :mod:`research_harness.experiments.planner`).
    """

    plan_id: str
    run_id: str
    proposal_id: str
    version: int = 1
    derived_from_contract_version: int = 1
    cases: list[TestCase] = Field(default_factory=list)
    independent_baseline: str = ""
    randomness_controls: str = ""
    author_role: str = "test_designer"
    amendments: list[ContractRevision] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)

    def content_hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            [c.model_dump() for c in self.cases]
            + [self.independent_baseline, self.randomness_controls],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def cases_for(self, category: TestCategory) -> list[TestCase]:
        return [c for c in self.cases if c.category == category]


class MetricObservation(Base):
    metric_key: str
    value: float
    unit: str = ""
    provenance: Provenance = Provenance.SIMULATED
    n_repeats: int = 1
    stddev: float | None = None
    seed: int | None = None


class TestOutcome(Base):
    test_id: str
    passed: bool | None  # None = not run / not determinable
    detail: str = ""
    raw_output_path: str | None = None


class ExperimentResult(Base):
    experiment_id: str
    run_id: str
    proposal_id: str
    plan_id: str
    plan_hash: str
    adapter_id: str
    is_simulated: bool
    requires_human_input: bool = False
    executed: bool = False
    environment: dict[str, str] = Field(default_factory=dict)
    seed: int | None = None
    outcomes: list[TestOutcome] = Field(default_factory=list)
    observations: list[MetricObservation] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    stdout_path: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    started_at: str = Field(default_factory=utcnow)

    @property
    def provenance(self) -> Provenance:
        if self.is_simulated:
            return Provenance.SIMULATED
        return Provenance.MEASURED


class MetricVerdict(Base):
    metric_key: str
    target_threshold: float | None
    direction: MetricDirection
    observed: float | None
    met: bool | None  # None = unmeasured
    provenance: Provenance


class ValidationResult(Base):
    """Produced by the Independent Validator. Never derived from the Executor's
    own claim of success."""

    run_id: str
    proposal_id: str
    experiment_id: str
    # Loop A question: did we build what the proposal said?
    implementation_conforms: bool | None = None
    implementation_notes: str = ""
    failed_test_ids: list[str] = Field(default_factory=list)
    # Loop B question: does the ORIGINAL goal hold?
    metric_verdicts: list[MetricVerdict] = Field(default_factory=list)
    goal_satisfied: bool | None = None
    maturity_level: MaturityLevel = MaturityLevel.INCONCLUSIVE
    evidence_gaps: list[str] = Field(default_factory=list)
    blocking_reason: str = ""
    external_validation_plan: list[str] = Field(default_factory=list)
    validated_at: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------
# Decisions / VoI
# --------------------------------------------------------------------------


class ExperimentCandidate(Base):
    label: str
    targets_assumption: str
    proposal_ids: list[str] = Field(default_factory=list)
    predicted_decision_impact: float = Field(ge=0.0, le=1.0)
    estimated_cost_units: float = Field(gt=0)
    priority: float = 0.0
    impact_is_predicted: bool = True  # never present this as a measured quantity


class DecisionRecord(Base):
    decision_id: str
    run_id: str
    iteration: int
    loop: Literal["implementation", "research", "orchestrator"]
    hypothesis: str
    action: str
    expected_result: str
    actual_result: str = ""
    divergence_cause: str = ""
    continue_reason: str = ""
    should_continue: bool = True
    at: str = Field(default_factory=utcnow)


class StopDecision(Base):
    should_stop: bool
    reason: str
    detail: str = ""


# --------------------------------------------------------------------------
# Adaptive routing (spec section 4.2)
# --------------------------------------------------------------------------


class RoutingDecision(Base):
    """One recorded routing choice, with why it was made and what it cost.

    ``decided_by`` is the part that matters for auditability: a route chosen by
    a deterministic rule (budget exhausted, approval pending, evidence missing)
    is accountable in a way a model's opinion is not. The spec forbids routing
    on an opaque model judgement alone, so a model suggestion is recorded
    alongside the rule that actually fired and is overridden when they differ.
    """

    decision_id: str
    run_id: str
    iteration: int
    action: RouteAction
    decided_by: Literal["rule", "model", "budget", "mode"] = "rule"
    rule_fired: str | None = None
    model_suggestion: RouteAction | None = None
    model_rationale: str = ""
    trigger: str = ""
    rationale: str = ""
    predicted_cost_units: float = 0.0
    #: Filled in after the action ran, so routing quality can be reviewed later.
    realised_new_evidence: int | None = None
    realised_maturity_gain: bool | None = None
    at: str = Field(default_factory=utcnow)

    @property
    def overrode_model(self) -> bool:
        return (
            self.model_suggestion is not None and self.model_suggestion is not self.action
        )


class ResearchEvent(Base):
    """Append-only audit record. Never rewritten, only added to."""

    seq: int
    run_id: str
    event: EventType
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    #: Stable key for actions with side effects, so a resumed run can tell that
    #: something already happened instead of doing it twice.
    idempotency_key: str | None = None
    at: str = Field(default_factory=utcnow)


class TaskNode(Base):
    """One node of a decomposed research task graph (spec section 4.3)."""

    task_id: str
    question: str
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "blocked"] = "pending"
    findings: str = ""
    evidence_ids: list[str] = Field(default_factory=list)

    def is_ready(self, done: set[str]) -> bool:
        return all(dep in done for dep in self.depends_on)


class ModeResult(Base):
    """Outcome of one mode in an ablation comparison (spec section 16)."""

    mode: Mode
    run_id: str
    maturity_level: MaturityLevel
    outcome: RunOutcome
    goal_satisfied: bool | None
    selected_proposal: str | None
    proposal_count: int
    real_evidence_count: int
    hard_constraint_violations: int
    iterations: int
    llm_calls: int
    tokens_used: int
    cost_usd: float
    elapsed_seconds: float
    experiments_run: int
    routing_actions: list[str] = Field(default_factory=list)
    top1_stable: bool | None = None
    position_bias_rate: float | None = None


class AblationReport(Base):
    """Side-by-side comparison of modes on one identical goal."""

    idea: str
    seed: int
    results: list[ModeResult] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    caveat: str = (
        "A single run per mode on one idea cannot establish that a richer mode is "
        "better; it shows what each mode cost and reached on this instance. Drawing "
        "a general conclusion needs repeats across several goals with a fixed model "
        "and budget."
    )
