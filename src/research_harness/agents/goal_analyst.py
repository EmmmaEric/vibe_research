"""Goal Analyst: turns a free-text idea into a versioned Research Contract.

Hard rule enforced structurally: ``original_idea`` is stored verbatim and is
immutable (:meth:`ResearchContract.revise` refuses to change it). Anything the
analyst infers rather than reads is flagged ``origin=ASSUMED`` so the final report
can separate the requester's stated goals from our placeholders.
"""

from __future__ import annotations

import enum
import re

from pydantic import Field

from ..core.schemas import (
    ApprovalPolicy,
    Base,
    Budget,
    HardConstraint,
    GoalType,
    MetricDirection,
    Origin,
    ResearchContract,
    SoftPreference,
    SuccessMetric,
    MaturityLevel,
)
from . import domains


#: Legal ``domain_profile`` values, generated from the registry so a new domain
#: is automatically offerable to the model and the two cannot drift apart.
DomainName = enum.Enum(
    "DomainName",
    {p.name.upper(): p.name for p in domains.REGISTRY},
    type=str,
)


def _enum_value(value) -> str:
    """The string value of an enum member, or the string itself."""
    return value.value if hasattr(value, "value") else str(value)


def _origin(value: Origin | str) -> Origin:
    """Coerce a supplied origin, defaulting to ASSUMED.

    The schema now constrains this to the enum, so this is a safety net for the
    weaker structured-output tiers. An unrecognised value must never become
    USER_SPECIFIED: that would let a malformed response promote a harness guess
    into a stated goal.
    """
    if isinstance(value, Origin):
        return value
    try:
        return Origin(str(value).strip().lower())
    except ValueError:
        return Origin.ASSUMED

#: Universal effort ceiling, expressed in abstract "cost units" so it applies
#: across domains. Machine-checkable, so the constraint gate decides it by
#: arithmetic rather than judgement.
DEFAULT_COST_UNITS_LIMIT = 4.0

_SYSTEM = """\
You are a Goal Analyst. You convert a requester's idea into a precise, testable \
research contract.

Absolute rules:
- Never change, soften or broaden what the requester asked for. If their goal is \
hard, the contract stays hard.
- Distinguish what they actually stated from what you are assuming. Mark every \
inferred metric or constraint with origin="assumed".
- Success metrics must be observable, with a unit and a measurement method. If no \
numeric target can be responsibly inferred, leave target_threshold null and say \
so in unknowns -- do not invent a number.
- Hard constraints are things that disqualify a solution outright (budget, \
legality, safety, required device, interface compatibility). Preferences that can \
be traded off are soft_preferences, not hard constraints.
- List what you do not know in `unknowns`. An honest unknown is more useful than \
a confident guess.
"""


class _MetricOut(Base):
    key: str
    description: str
    unit: str
    target_threshold: float | None = None
    direction: str = "maximize"
    measurement_method: str = "unspecified"
    #: Closed enum, not free text: a real model answered "derived from user
    #: idea", which fell through to ASSUMED and destroyed the very distinction
    #: this field exists to record. The schema now enforces the three values.
    origin: Origin = Origin.ASSUMED
    required: bool = True


class _ConstraintOut(Base):
    key: str
    description: str
    category: str = "other"
    verification_hint: str = ""
    origin: Origin = Origin.ASSUMED


class _PreferenceOut(Base):
    key: str
    description: str
    importance: int = 3


class GoalAnalysis(Base):
    """Schema the LLM fills; also the simulated path's output shape."""

    problem_statement: str
    #: Selects the scoring weight prior, so it must be one of the known kinds.
    goal_type: GoalType = GoalType.TECHNICAL_PERFORMANCE
    #: Selects the domain profile, so it must name one from the registry.
    domain_profile: DomainName = DomainName.GENERIC
    scope_included: list[str] = Field(default_factory=list)
    scope_excluded: list[str] = Field(default_factory=list)
    success_metrics: list[_MetricOut] = Field(default_factory=list)
    hard_constraints: list[_ConstraintOut] = Field(default_factory=list)
    soft_preferences: list[_PreferenceOut] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    available_resources: list[str] = Field(default_factory=list)


class GoalAnalyst:
    def __init__(self, provider) -> None:
        self.provider = provider

    # ------------------------------------------------------------------

    def build_contract(
        self,
        *,
        run_id: str,
        idea: str,
        budget: Budget | None = None,
        approval_policy: ApprovalPolicy | None = None,
        desired_maturity_level: MaturityLevel = MaturityLevel.R3_PROTOTYPED,
        domain_hint: str | None = None,
        cost_units_limit: float = DEFAULT_COST_UNITS_LIMIT,
    ) -> ResearchContract:
        analysis = (
            self._simulate(idea, domain_hint)
            if self.provider.is_simulated
            else self._via_llm(idea, domain_hint)
        )

        metrics = [
            SuccessMetric(
                key=m.key,
                description=m.description,
                unit=m.unit,
                target_threshold=m.target_threshold,
                direction=(
                    MetricDirection.MINIMIZE
                    if str(m.direction).lower().startswith("min")
                    else MetricDirection.MAXIMIZE
                ),
                measurement_method=m.measurement_method,
                origin=_origin(m.origin),
                required=m.required,
            )
            for m in analysis.success_metrics
        ]

        constraints = [
            HardConstraint(
                key=c.key,
                description=c.description,
                category=c.category,
                verification_hint=c.verification_hint,
                origin=_origin(c.origin),
            )
            for c in analysis.hard_constraints
        ]
        # Always append the machine-checkable effort ceiling, so at least one
        # hard constraint can be decided by arithmetic instead of judgement.
        if not any(c.key == "within_cost_budget" for c in constraints):
            constraints.append(
                HardConstraint(
                    key="within_cost_budget",
                    description=(
                        f"Total declared effort must not exceed {cost_units_limit} cost "
                        "units (abstract effort/expenditure units shared across "
                        "candidates)"
                    ),
                    category="budget",
                    verification_hint="compare the candidate's declared cost units to the limit",
                    origin=Origin.ASSUMED,
                    limit_metric="declared_cost_units",
                    limit_value=cost_units_limit,
                    limit_direction=MetricDirection.MINIMIZE,
                )
            )

        unknowns = list(analysis.unknowns)
        for m in metrics:
            if m.target_threshold is None:
                unknowns.append(
                    f"No numeric target was stated or responsibly inferable for metric "
                    f"'{m.key}'; it cannot be used to decide goal attainment until set."
                )

        return ResearchContract(
            run_id=run_id,
            original_idea=idea,
            problem_statement=analysis.problem_statement,
            scope_included=analysis.scope_included,
            scope_excluded=analysis.scope_excluded,
            goal_type=_enum_value(analysis.goal_type),
            success_metrics=metrics,
            hard_constraints=constraints,
            soft_preferences=[
                SoftPreference(
                    key=p.key, description=p.description, importance=p.importance
                )
                for p in analysis.soft_preferences
            ],
            assumptions=analysis.assumptions,
            unknowns=list(dict.fromkeys(unknowns)),
            available_resources=analysis.available_resources,
            desired_maturity_level=desired_maturity_level,
            budget=budget or Budget(),
            approval_policy=approval_policy or ApprovalPolicy(),
            domain_hint=_enum_value(analysis.domain_profile),
        )

    # ------------------------------------------------------------------

    def _via_llm(self, idea: str, domain_hint: str | None) -> GoalAnalysis:
        hint = f"\n\nSuggested domain profile (may be ignored): {domain_hint}" if domain_hint else ""
        known = ", ".join(p.name for p in domains.REGISTRY)
        prompt = (
            f"REQUESTER'S IDEA (verbatim):\n{idea}\n\n"
            f"Available domain_profile values: {known}\n"
            "Produce the research contract fields. Remember: observable metrics with "
            "units and measurement methods; hard constraints only for things that "
            "disqualify a candidate outright; mark inferred items origin=assumed; put "
            "what you cannot determine in unknowns." + hint
        )
        return self.provider.complete_structured(
            system=_SYSTEM,
            prompt=prompt,
            schema=GoalAnalysis,
            purpose="goal_analyst.build_contract",
        )

    def _simulate(self, idea: str, domain_hint: str | None) -> GoalAnalysis:
        """Deterministic contract derivation from the domain registry.

        Keyword-matches a domain profile, adopts its conservative default metrics
        and constraints, and extracts any explicit numeric budget mentioned in the
        idea text. Everything inferred is marked assumed.
        """
        profile = domains.select_profile(idea, hint=domain_hint)
        metrics = [
            _MetricOut(
                key=m.key,
                description=m.description,
                unit=m.unit,
                target_threshold=m.target_threshold,
                direction=m.direction.value,
                measurement_method=m.measurement_method,
                origin=Origin.ASSUMED,
            )
            for m in profile.metric_templates
        ]
        constraints = [
            _ConstraintOut(
                key=c.key,
                description=c.description,
                category=c.category,
                verification_hint=c.verification_hint,
                origin=Origin.ASSUMED,
            )
            for c in profile.constraint_templates
        ]

        # Pull an explicit money constraint out of the idea if one is stated.
        money = re.search(
            r"(?:under|below|less than|within|<=?|不超过|低于)\s*"
            r"[$¥€]?\s*(\d[\d,]*(?:\.\d+)?)\s*"
            r"(usd|dollars?|元|rmb|eur|k|thousand|million|万)?",
            idea,
            re.IGNORECASE,
        )
        assumptions = [
            f"Domain profile '{profile.name}' was selected by keyword match on the "
            "idea text; the resulting goal_type only sets a scoring weight prior.",
            "Default success metrics and hard constraints below are conservative "
            "placeholders, not requirements stated by the requester.",
        ]
        if money:
            amount = money.group(1).replace(",", "")
            unit = money.group(2) or ""
            constraints.append(
                _ConstraintOut(
                    key="stated_budget_ceiling",
                    description=(
                        f"The requester stated a ceiling of {amount} {unit}".strip()
                    ),
                    category="budget",
                    verification_hint="cost model reviewed against this figure",
                    origin=Origin.USER_SPECIFIED,
                )
            )
            assumptions.append(
                f"A numeric ceiling ({amount} {unit}) was detected in the idea text and "
                "recorded as a stated (not assumed) hard constraint."
            )

        prefs: list[_PreferenceOut] = []
        for token, key, desc in (
            ("cheap", "cost_time", "Requester signalled cost sensitivity"),
            ("低成本", "cost_time", "Requester signalled cost sensitivity"),
            ("low-cost", "cost_time", "Requester signalled cost sensitivity"),
            ("fast", "cost_time", "Requester signalled a preference for speed"),
            ("offline", "risk_robustness", "Requester signalled an offline/robustness need"),
            ("离线", "risk_robustness", "Requester signalled an offline/robustness need"),
            ("novel", "novelty", "Requester signalled interest in novelty"),
            ("new ", "novelty", "Requester signalled interest in novelty"),
        ):
            if token in idea.lower() and not any(p.key == key for p in prefs):
                prefs.append(_PreferenceOut(key=key, description=desc, importance=4))

        unknowns = [
            "Whether the requester accepts the placeholder metrics and thresholds above.",
            "What verification depth is acceptable: literature only, PoC, or real-world "
            "measurement.",
        ]
        if profile is domains.GENERIC:
            unknowns.append(
                "No domain could be identified from the idea text, so no domain-specific "
                "metric or validation path was selected. The primary metric is an "
                "explicit placeholder and must be replaced before any result is "
                "meaningful."
            )

        statement = (
            f"[Derived offline from the idea text] Determine whether the requester's "
            f"objective -- \"{idea.strip()[:220]}\" -- can be met, by generating "
            f"competing candidate routes, scoring them against goal-derived criteria, "
            f"and verifying the leading candidate(s) against the contract's success "
            f"metrics. Domain profile: {profile.name}."
        )

        return GoalAnalysis(
            problem_statement=statement,
            goal_type=GoalType(profile.goal_type),
            domain_profile=DomainName(profile.name),
            scope_included=[
                "evidence gathering",
                "competing candidate design",
                "criteria-based evaluation and ranking",
                "test/verification planning",
            ],
            scope_excluded=[
                "any action with irreversible real-world effect",
                "real payments, device control, or contacting real users",
                "claims of global optimality",
            ],
            success_metrics=metrics,
            hard_constraints=constraints,
            soft_preferences=prefs,
            assumptions=assumptions,
            unknowns=unknowns,
            available_resources=["local filesystem", "local Python execution sandbox"],
        )
