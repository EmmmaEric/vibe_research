"""Proposal Designer: generates genuinely competing candidate routes.

Design constraints taken from the spec:

* Candidates are generated **independently** -- one designer invocation per
  archetype, each seeing only the contract and evidence, never a sibling's
  output. Sharing unreviewed conclusions between designers is how a candidate
  set collapses into three rewordings of the same idea.
* Every candidate uses the identical schema and a comparable level of detail, so
  a judge cannot be swayed by formatting or length.
* Differentiation is checked after the fact by
  :func:`assert_substantive_difference`, which compares mechanisms rather than
  trusting that asking for variety produced it.
"""

from __future__ import annotations

import re

from pydantic import Field

from ..core.schemas import Base, Evidence, PoCSketch, Proposal, ResearchContract, Stance
from . import domains

#: Declared effort per archetype, in the abstract cost units the budget
#: constraint checks. Exploratory routes are genuinely more expensive, which is
#: what lets the constraint gate be exercised on a real arithmetic violation.
ARCHETYPE_COST = {"conservative": 1.5, "balanced": 2.5, "exploratory": 5.0}

_SYSTEM = """\
You are a Proposal Designer. You design ONE candidate route to the stated goal, \
following the assigned strategy archetype.

Rules:
- Design a genuinely distinct technical or operational mechanism. Changing \
parameters, names or wording is not a different proposal.
- Address the goal as written in the contract. Do not substitute an easier goal.
- State what must be TRUE for your route to work (critical_assumptions), and the \
condition that would prove it wrong (falsification_condition). A route that \
cannot be wrong cannot be tested.
- Cite only evidence_ids from the supplied pool. Never invent a source.
- Name your own failure modes honestly. Omitting them does not make them absent, \
and a candidate with no stated failure modes scores worse on robustness, not better.
- declared_cost_units is an abstract effort estimate on a 0-6 scale; be realistic, \
as a hard budget constraint is checked against it.
"""


class ProposalDraft(Base):
    """Schema for one designed candidate."""

    title: str
    core_mechanism: str
    goal_mapping: str
    dependencies: list[str] = Field(default_factory=list)
    premises: list[str] = Field(default_factory=list)
    key_steps: list[str] = Field(default_factory=list)
    resource_needs: list[str] = Field(default_factory=list)
    differentiators: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    critical_assumptions: list[str] = Field(default_factory=list)
    poc_description: str = ""
    poc_expected_result: str = ""
    poc_falsification_condition: str = ""
    poc_cost_units: float = 1.0
    distinction_from_others: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    declared_cost_units: float = 2.0


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "under",
        "over", "a", "an", "of", "to", "in", "on", "by", "is", "are", "be",
        "its", "it", "as", "at", "or", "via", "one", "two", "all", "existing",
    }
    return {
        w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in stop
    }


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def assert_substantive_difference(
    proposals: list[Proposal], *, threshold: float = 0.72
) -> list[str]:
    """Flag candidate pairs whose mechanisms are near-duplicates.

    Returns a list of human-readable warnings (empty when all candidates differ).
    Compares ``core_mechanism`` plus ``key_steps``, which is where a real
    difference has to show up -- titles and framing are cheap to vary.
    """
    warnings: list[str] = []
    for i in range(len(proposals)):
        for j in range(i + 1, len(proposals)):
            p, q = proposals[i], proposals[j]
            sim = jaccard(
                p.core_mechanism + " " + " ".join(p.key_steps),
                q.core_mechanism + " " + " ".join(q.key_steps),
            )
            if sim >= threshold:
                warnings.append(
                    f"{p.proposal_id} and {q.proposal_id} have {sim:.0%} mechanism "
                    "overlap -- they may be rewordings rather than competing routes. "
                    "Treat the candidate set as effectively smaller."
                )
    return warnings


class ProposalDesigner:
    def __init__(self, provider) -> None:
        self.provider = provider

    # ------------------------------------------------------------------

    def design_all(
        self,
        *,
        contract: ResearchContract,
        evidence: list[Evidence],
        archetypes: list[str] | None = None,
    ) -> tuple[list[Proposal], list[str]]:
        """Design one candidate per archetype, independently."""
        profile = domains.select_profile(
            contract.original_idea, hint=contract.domain_hint
        )
        arcs = archetypes or list(profile.archetypes)

        proposals: list[Proposal] = []
        for idx, arc in enumerate(arcs):
            draft = (
                self._simulate(contract, evidence, profile, arc)
                if self.provider.is_simulated
                else self._via_llm(contract, evidence, arc)
            )
            valid = {e.evidence_id for e in evidence}
            pid = f"P{chr(ord('A') + idx)}"
            proposals.append(
                Proposal(
                    proposal_id=pid,
                    run_id=contract.run_id,
                    title=draft.title,
                    strategy_archetype=arc,
                    core_mechanism=draft.core_mechanism,
                    goal_mapping=draft.goal_mapping,
                    dependencies=draft.dependencies,
                    premises=draft.premises,
                    key_steps=draft.key_steps,
                    resource_needs=draft.resource_needs,
                    differentiators=draft.differentiators,
                    failure_modes=draft.failure_modes,
                    critical_assumptions=draft.critical_assumptions,
                    poc=PoCSketch(
                        description=draft.poc_description or "No PoC specified.",
                        expected_result=draft.poc_expected_result,
                        falsification_condition=draft.poc_falsification_condition,
                        estimated_cost_units=max(draft.poc_cost_units, 0.1),
                    ),
                    distinction_from_others=draft.distinction_from_others,
                    # Drop invented evidence ids rather than trusting the author.
                    evidence_ids=[e for e in draft.evidence_ids if e in valid],
                    declared_cost_units=max(draft.declared_cost_units, 0.1),
                    generated_by=self.provider.provider_id,
                )
            )

        warnings = assert_substantive_difference(proposals)
        if len(proposals) < 3:
            warnings.append(
                f"Only {len(proposals)} candidate(s) were generated. Fewer than three "
                "requires a stated reason; with a thin candidate set, any 'best' "
                "result is weak."
            )
        return proposals, warnings

    # ------------------------------------------------------------------

    def _via_llm(
        self, contract: ResearchContract, evidence: list[Evidence], archetype: str
    ) -> ProposalDraft:
        ev = (
            "\n".join(
                f"- {e.evidence_id} [{e.stance.value}/{e.provenance.value}] {e.claim}"
                for e in evidence
            )
            or "(no evidence available -- say so in your premises)"
        )
        metrics = (
            "\n".join(
                f"- {m.key}: {m.description} [{m.unit}], target {m.direction.value} "
                f"{m.target_threshold}"
                for m in contract.success_metrics
            )
            or "(none declared)"
        )
        cons = (
            "\n".join(f"- {c.key}: {c.description}" for c in contract.hard_constraints)
            or "(none declared)"
        )
        guidance = {
            "conservative": "Lowest-risk route using well-established means. Prefer "
            "reliability over upside.",
            "balanced": "Trade some risk for materially better expected outcome.",
            "exploratory": "Highest-upside route. Name the key uncertainty that could "
            "sink it, and design the cheapest experiment that would resolve it.",
        }.get(archetype, f"Follow the '{archetype}' strategy.")

        prompt = (
            f"ORIGINAL IDEA (verbatim):\n{contract.original_idea}\n\n"
            f"PROBLEM STATEMENT:\n{contract.problem_statement}\n\n"
            f"SUCCESS METRICS:\n{metrics}\n\nHARD CONSTRAINTS:\n{cons}\n\n"
            f"AVAILABLE RESOURCES: {', '.join(contract.available_resources) or 'none'}\n\n"
            f"EVIDENCE POOL (cite only these ids):\n{ev}\n\n"
            f"YOUR ASSIGNED ARCHETYPE: {archetype}\n{guidance}\n\n"
            "Design exactly one candidate. You cannot see the other candidates; do "
            "not speculate about them -- describe what makes your mechanism distinctive "
            "on its own terms."
        )
        return self.provider.complete_structured(
            system=_SYSTEM,
            prompt=prompt,
            schema=ProposalDraft,
            purpose=f"proposal_designer.{archetype}",
        )

    def _simulate(
        self,
        contract: ResearchContract,
        evidence: list[Evidence],
        profile: domains.DomainProfile,
        archetype: str,
    ) -> ProposalDraft:
        """Deterministic candidate construction from the domain profile.

        The mechanism text comes from the profile's per-archetype mechanism table,
        so the three candidates differ structurally (and measurably, per
        :func:`assert_substantive_difference`) rather than by wording.
        """
        mechanisms = dict(profile.mechanisms)
        mech = mechanisms.get(archetype, f"a route following the {archetype} strategy")

        supporting = [
            e.evidence_id for e in evidence if e.stance is Stance.SUPPORTS
        ][:3]
        refuting = [e.evidence_id for e in evidence if e.stance is Stance.REFUTES][:2]
        cited = supporting + refuting or [e.evidence_id for e in evidence[:2]]

        metric_names = ", ".join(m.key for m in contract.success_metrics) or "the stated objective"
        cost = ARCHETYPE_COST.get(archetype, 2.5)

        shape = {
            "conservative": {
                "title": f"Established-practice route ({profile.name})",
                "deps": ["only capabilities already present in the declared resources"],
                "steps": [
                    "Establish the baseline measurement first, so any change is attributable",
                    f"Apply {mech}",
                    "Re-measure against the same baseline and report the delta with variance",
                ],
                "diff": [
                    "No new unproven component is introduced",
                    "Smallest blast radius if it fails",
                ],
                "fails": [
                    "Established methods may be insufficient to reach the target at all",
                ],
                "assume": [
                    "That the established approach can reach the contract's threshold",
                ],
                "poc": "Measure the baseline, apply the established change, re-measure.",
                "poc_cost": 1.0,
            },
            "balanced": {
                "title": f"Hybrid route with one targeted change ({profile.name})",
                "deps": [
                    "one new component or dependency with a clearly defined boundary",
                    "a migration or fallback path",
                ],
                "steps": [
                    "Define the interface boundary for the new component",
                    f"Apply {mech}",
                    "Run both routes side by side and compare on the contract metrics",
                    "Keep a fallback to the prior behaviour",
                ],
                "diff": [
                    "Captures most of the upside without a full restructure",
                    "Side-by-side comparison makes attribution clean",
                ],
                "fails": [
                    "The new boundary may leak, giving the costs of both routes",
                    "Side-by-side operation doubles maintenance during migration",
                ],
                "assume": [
                    "That the new component integrates without changing the interface contract",
                    "That running both routes concurrently is affordable",
                ],
                "poc": "Build the new component behind a flag and compare both paths.",
                "poc_cost": 2.0,
            },
            "exploratory": {
                "title": f"Restructured route around a different primitive ({profile.name})",
                "deps": [
                    "a mechanism not currently in use",
                    "capability or access that has not been confirmed to exist",
                ],
                "steps": [
                    "Validate the single riskiest assumption in isolation before building",
                    f"Apply {mech}",
                    "If the assumption holds, build the minimal end-to-end path",
                    "Compare against the baseline on the contract metrics",
                ],
                "diff": [
                    "Only route with headroom well beyond the stated target",
                    "Attacks the cause rather than the symptom",
                ],
                "fails": [
                    "The core assumption may simply be false, wasting the whole effort",
                    "Restructuring may break behaviour that currently works",
                    "Effort is likely to exceed the declared budget",
                ],
                "assume": [
                    "That the different primitive is actually applicable here",
                    "That the required capability or access can be obtained",
                    "That the restructure preserves current correctness",
                ],
                "poc": (
                    "Test ONLY the riskiest assumption in isolation, as cheaply as "
                    "possible, before committing any build effort."
                ),
                "poc_cost": 1.5,
            },
        }.get(archetype)

        if shape is None:  # custom archetype
            shape = {
                "title": f"{archetype.title()} route ({profile.name})",
                "deps": ["to be determined"],
                "steps": [f"Apply {mech}", "Measure against the baseline"],
                "diff": [f"Follows the {archetype} strategy"],
                "fails": ["Not characterised"],
                "assume": [f"That the {archetype} strategy applies to this goal"],
                "poc": f"Minimal trial of the {archetype} route.",
                "poc_cost": 1.5,
            }

        validation = ", ".join(profile.validation_notes) or "a defined measurement"
        return ProposalDraft(
            title=f"[SIMULATED] {shape['title']}",
            core_mechanism=(
                f"[SIMULATED] {mech.capitalize()}. Verification route: {validation}."
            ),
            goal_mapping=(
                f"Targets {metric_names} as declared in the research contract, measured "
                f"by the contract's stated measurement methods rather than a "
                f"self-defined criterion."
            ),
            dependencies=list(shape["deps"]),
            premises=[
                f"The contract's success metrics ({metric_names}) are the right measures.",
                "A baseline measurement is obtainable for comparison.",
            ],
            key_steps=list(shape["steps"]),
            resource_needs=list(contract.available_resources),
            differentiators=list(shape["diff"]),
            failure_modes=list(shape["fails"]),
            critical_assumptions=list(shape["assume"]),
            poc_description=f"[SIMULATED] {shape['poc']}",
            poc_expected_result=(
                f"A measured value for {metric_names} comparable to the baseline."
            ),
            poc_falsification_condition=(
                f"The measured value for {metric_names} fails to improve on the "
                "baseline beyond measurement variance."
            ),
            poc_cost_units=float(shape["poc_cost"]),
            distinction_from_others=(
                f"Differs by mechanism class: {mech}. Cost profile {cost} units."
            ),
            evidence_ids=cited,
            declared_cost_units=cost,
        )
