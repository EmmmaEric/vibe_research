"""Pluggable domain profiles.

The orchestrator, scoring engine and research state know nothing about domains.
All domain-specific knowledge lives here, in data: which goal type an idea maps
to, what conservative default metrics and constraints look like, and which
experiment adapter is plausible. Adding a domain means appending a
:class:`DomainProfile` -- no change to the core.

Every default this module injects into a contract is flagged
``origin=Origin.ASSUMED`` so
a report can distinguish "the user said this" from "we filled in a conservative
placeholder". Profile selection is keyword matching, which is a weak heuristic;
the resulting ``goal_type`` is only a *weight prior*, and the user can override
both it and every injected default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.schemas import HardConstraint, MetricDirection, Origin, SuccessMetric


@dataclass(frozen=True)
class MetricTemplate:
    key: str
    description: str
    unit: str
    direction: MetricDirection
    target_threshold: float | None
    measurement_method: str


@dataclass(frozen=True)
class ConstraintTemplate:
    key: str
    description: str
    category: str
    verification_hint: str = ""
    limit_metric: str | None = None
    limit_value: float | None = None


@dataclass(frozen=True)
class DomainProfile:
    name: str
    goal_type: str
    keywords: tuple[str, ...]
    #: Which experiment adapter is plausible for this domain. The adapter itself
    #: decides whether it can actually run; this is only a preference order.
    preferred_adapters: tuple[str, ...]
    metric_templates: tuple[MetricTemplate, ...] = ()
    constraint_templates: tuple[ConstraintTemplate, ...] = ()
    archetypes: tuple[str, ...] = ("conservative", "balanced", "exploratory")
    validation_notes: tuple[str, ...] = ()
    research_queries: tuple[str, ...] = ()
    mechanisms: tuple[tuple[str, str], ...] = ()
    extra: dict = field(default_factory=dict)

    def score(self, text: str) -> int:
        low = text.lower()
        return sum(1 for k in self.keywords if k in low)


SOFTWARE = DomainProfile(
    name="software_system",
    goal_type="technical_performance",
    keywords=(
        "software", "architecture", "api", "service", "latency", "throughput",
        "refactor", "codebase", "library", "backend", "database", "cache",
        "microservice", "deploy", "scalab", "系统", "架构", "接口", "延迟",
    ),
    preferred_adapters=("pytest", "simulated"),
    metric_templates=(
        MetricTemplate(
            key="p95_latency_ms",
            description="95th-percentile end-to-end request latency",
            unit="ms",
            direction=MetricDirection.MINIMIZE,
            target_threshold=None,
            measurement_method="load test against a fixed request mix, >=3 repeats",
        ),
        MetricTemplate(
            key="correctness_pass_rate",
            description="Fraction of correctness tests passing",
            unit="ratio",
            direction=MetricDirection.MAXIMIZE,
            target_threshold=1.0,
            measurement_method="automated test suite execution",
        ),
    ),
    constraint_templates=(
        ConstraintTemplate(
            key="no_breaking_api_change",
            description="Must not break the existing public interface contract",
            category="interface",
            verification_hint="run the existing contract tests unchanged",
        ),
    ),
    research_queries=(
        "established architectural patterns for {topic}",
        "measured performance comparisons for {topic}",
        "failure modes and operational cost of {topic}",
    ),
    mechanisms=(
        ("conservative", "incremental change to the existing structure, keeping all current interfaces"),
        ("balanced", "introduce one new component with a clear boundary, migrating callers behind a flag"),
        ("exploratory", "restructure the data/control flow around a different primitive"),
    ),
    validation_notes=("unit tests", "integration tests", "load benchmark vs. baseline"),
)

ALGORITHM = DomainProfile(
    name="ai_algorithm",
    goal_type="scientific_validation",
    keywords=(
        "algorithm", "model", "neural", "training", "accuracy", "dataset",
        "inference", "optimiz", "optimis", "kernel", "gradient", "benchmark",
        "machine learning", "convergence", "算法", "模型", "训练", "精度",
    ),
    preferred_adapters=("pytest", "simulated"),
    metric_templates=(
        MetricTemplate(
            key="task_metric",
            description="Primary task quality metric on a held-out split",
            unit="score",
            direction=MetricDirection.MAXIMIZE,
            target_threshold=None,
            measurement_method="held-out test split, fixed seed, >=3 repeats, report variance",
        ),
        MetricTemplate(
            key="speedup_vs_baseline",
            description="Wall-clock speedup relative to the declared baseline",
            unit="x",
            direction=MetricDirection.MAXIMIZE,
            target_threshold=1.0,
            measurement_method="same hardware, same input distribution, >=5 repeats",
        ),
    ),
    constraint_templates=(
        ConstraintTemplate(
            key="numerical_equivalence",
            description="Output must remain within the declared numerical tolerance of the baseline",
            category="safety",
            verification_hint="differential test against the reference implementation",
        ),
    ),
    research_queries=(
        "state of the art methods for {topic}",
        "published ablations and negative results for {topic}",
        "reproducibility reports for {topic}",
    ),
    mechanisms=(
        ("conservative", "tune and correctly configure a well-established method"),
        ("balanced", "combine an established method with one targeted modification"),
        ("exploratory", "replace the core computational primitive with a different formulation"),
    ),
    validation_notes=(
        "public dataset or held-out split",
        "ablation study",
        "statistical comparison against baseline with variance reported",
    ),
)

PRODUCT = DomainProfile(
    name="product_business",
    goal_type="market_demand",
    keywords=(
        "product", "market", "customer", "user", "business", "pricing", "revenue",
        "competitor", "saas", "adoption", "churn", "willingness to pay", "startup",
        "小型企业", "产品", "市场", "客户", "商业", "协作",
    ),
    preferred_adapters=("manual", "simulated"),
    metric_templates=(
        MetricTemplate(
            key="validated_demand_signal",
            description=(
                "Share of interviewed target users who confirm the problem is a top-3 "
                "pain and express willingness to pay"
            ),
            unit="ratio",
            direction=MetricDirection.MAXIMIZE,
            target_threshold=0.4,
            measurement_method="structured interviews with >=12 qualified target users",
        ),
        MetricTemplate(
            key="unit_economics_ratio",
            description="Estimated lifetime value divided by customer acquisition cost",
            unit="ratio",
            direction=MetricDirection.MAXIMIZE,
            target_threshold=3.0,
            measurement_method="bottom-up cost model with sourced input assumptions",
        ),
    ),
    constraint_templates=(
        ConstraintTemplate(
            key="data_privacy_compliance",
            description=(
                "Must comply with applicable data-protection law for the target market"
            ),
            category="legal",
            verification_hint="written review against the specific regulation in scope",
        ),
    ),
    research_queries=(
        "existing competitors and their positioning for {topic}",
        "published user research or survey data on {topic}",
        "regulatory constraints affecting {topic}",
    ),
    mechanisms=(
        ("conservative", "narrow wedge product serving one well-evidenced segment need"),
        ("balanced", "integrated offering covering the two most-cited needs"),
        ("exploratory", "change the delivery or business model rather than the feature set"),
    ),
    validation_notes=(
        "structured user interviews",
        "clickable prototype usability test",
        "pricing / willingness-to-pay probe",
        "A/B test once live traffic exists",
    ),
)

HARDWARE = DomainProfile(
    name="hardware_robotics",
    goal_type="technical_performance",
    keywords=(
        "hardware", "robot", "sensor", "actuator", "pcb", "embedded", "firmware",
        "motor", "power consumption", "device", "mechanical", "硬件", "机器人", "设备",
    ),
    preferred_adapters=("simulated", "manual"),
    metric_templates=(
        MetricTemplate(
            key="power_consumption_w",
            description="Average power draw under the nominal duty cycle",
            unit="W",
            direction=MetricDirection.MINIMIZE,
            target_threshold=None,
            measurement_method="bench measurement on physical hardware",
        ),
        MetricTemplate(
            key="bill_of_materials_usd",
            description="Per-unit bill of materials cost at the target volume",
            unit="USD",
            direction=MetricDirection.MINIMIZE,
            target_threshold=None,
            measurement_method="supplier quotes at the stated volume",
        ),
    ),
    constraint_templates=(
        ConstraintTemplate(
            key="physical_device_available",
            description="Validation requires physical hardware that must be available",
            category="device",
            verification_hint="confirm the specific device and access window exist",
        ),
        ConstraintTemplate(
            key="electrical_safety",
            description="Must meet applicable electrical safety requirements",
            category="safety",
            verification_hint="review against the named standard; certified test if shipping",
        ),
    ),
    research_queries=(
        "component options and datasheets for {topic}",
        "measured power and thermal characteristics for {topic}",
        "safety standards applicable to {topic}",
    ),
    mechanisms=(
        ("conservative", "assemble from qualified off-the-shelf modules"),
        ("balanced", "off-the-shelf core with one custom interface board"),
        ("exploratory", "custom integrated design trading effort for size, cost or power"),
    ),
    validation_notes=(
        "physics simulation / digital twin",
        "bench measurement on real hardware",
        "cost and energy budget review",
    ),
)

SCIENCE = DomainProfile(
    name="science_theory",
    goal_type="scientific_validation",
    keywords=(
        "theorem", "proof", "hypothesis", "physics", "chemistry", "biology",
        "mathematical", "simulation", "equation", "derivation", "experiment",
        "假设", "定理", "证明", "仿真", "科研",
    ),
    preferred_adapters=("pytest", "simulated"),
    metric_templates=(
        MetricTemplate(
            key="hypothesis_support",
            description=(
                "Whether the preregistered prediction is supported at the declared "
                "significance level"
            ),
            unit="boolean_as_ratio",
            direction=MetricDirection.MAXIMIZE,
            target_threshold=1.0,
            measurement_method=(
                "preregistered analysis with the test and threshold fixed in advance"
            ),
        ),
    ),
    constraint_templates=(
        ConstraintTemplate(
            key="no_post_hoc_criteria_change",
            description=(
                "Acceptance criteria and analysis plan must be fixed before observing "
                "results"
            ),
            category="safety",
            verification_hint="compare the executed analysis to the versioned test plan",
        ),
    ),
    research_queries=(
        "prior theoretical treatments of {topic}",
        "existing experimental or numerical evidence on {topic}",
        "known counterexamples or refutations concerning {topic}",
    ),
    mechanisms=(
        ("conservative", "apply the established analytical framework to the stated case"),
        ("balanced", "combine analytical derivation with targeted numerical verification"),
        ("exploratory", "reformulate under weaker assumptions and search for counterexamples"),
    ),
    validation_notes=(
        "numerical verification",
        "counterexample search",
        "formal / symbolic check where tractable",
        "independent reproduction",
    ),
)

GENERIC = DomainProfile(
    name="generic",
    goal_type="technical_performance",
    keywords=(),
    preferred_adapters=("simulated", "manual"),
    metric_templates=(
        MetricTemplate(
            key="primary_objective_measure",
            description=(
                "Primary observable measure of whether the stated objective is reached "
                "(PLACEHOLDER: no measurable target was stated; replace before relying "
                "on any result)"
            ),
            unit="unspecified",
            direction=MetricDirection.MAXIMIZE,
            target_threshold=None,
            measurement_method="undefined -- must be specified by the requester",
        ),
    ),
    research_queries=(
        "existing approaches to {topic}",
        "evidence for and against {topic}",
        "practical constraints and costs of {topic}",
    ),
    mechanisms=(
        ("conservative", "the lowest-risk established route to the stated objective"),
        ("balanced", "a trade-off route accepting moderate novelty for clearer upside"),
        ("exploratory", "a mechanism distinct from current practice with higher variance"),
    ),
    validation_notes=(
        "literature cross-check",
        "explicit assumption list",
        "external validation plan for what cannot be tested here",
    ),
)

REGISTRY: tuple[DomainProfile, ...] = (
    SOFTWARE,
    ALGORITHM,
    PRODUCT,
    HARDWARE,
    SCIENCE,
    GENERIC,
)


def select_profile(idea: str, *, hint: str | None = None) -> DomainProfile:
    """Pick the best-matching profile for an idea.

    ``hint`` (an explicit profile name) always wins. Otherwise the best keyword
    match wins; ties and zero matches fall back to :data:`GENERIC`, which uses an
    explicitly-placeholder metric rather than pretending to know the target.
    """
    if hint:
        for p in REGISTRY:
            if p.name == hint:
                return p
    best = GENERIC
    best_score = 0
    for p in REGISTRY:
        s = p.score(idea)
        if s > best_score:
            best, best_score = p, s
    return best


def to_metrics(profile: DomainProfile) -> list[SuccessMetric]:
    return [
        SuccessMetric(
            key=t.key,
            description=t.description,
            unit=t.unit,
            target_threshold=t.target_threshold,
            direction=t.direction,
            measurement_method=t.measurement_method,
            origin=Origin.ASSUMED,
        )
        for t in profile.metric_templates
    ]


def to_constraints(profile: DomainProfile) -> list[HardConstraint]:
    return [
        HardConstraint(
            key=t.key,
            description=t.description,
            category=t.category,
            verification_hint=t.verification_hint,
            origin=Origin.ASSUMED,
            limit_metric=t.limit_metric,
            limit_value=t.limit_value,
        )
        for t in profile.constraint_templates
    ]
