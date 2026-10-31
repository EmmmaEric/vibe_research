"""Skeptic / Critic: finds counterexamples, hidden costs and failure modes.

Constrained in both directions. It must not dismiss a candidate without a
reason, and it must not wave one through either. Its output is advisory: a
critique adds *information* (unstated dependencies, hidden costs) and can raise
a hard-constraint question, but it never sets a score or eliminates a candidate
on its own -- that stays with the deterministic constraint gate.
"""

from __future__ import annotations

from pydantic import Field

from ..core.schemas import (
    Base,
    Critique,
    Evidence,
    Proposal,
    ResearchContract,
    Stance,
)

_SYSTEM = """\
You are a Skeptic. For the given candidate you look for what would make it fail.

Rules:
- Every objection must have a reason: a mechanism, a cited counterexample, or a \
named dependency. "This seems risky" is not an objection.
- Do NOT penalise a candidate for being unfamiliar or ambitious. Novelty is not \
a defect; unexamined novelty is. Say specifically what is unverified.
- Surface costs and dependencies the author did not state, especially operational \
and maintenance burden.
- If you cannot find a substantive problem, say so and set severity low. Inventing \
an objection to look rigorous is a failure.
"""


class CritiqueDraft(Base):
    counterexamples: list[str] = Field(default_factory=list)
    hidden_costs: list[str] = Field(default_factory=list)
    unstated_dependencies: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    severity: int = 2
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class Skeptic:
    def __init__(self, provider) -> None:
        self.provider = provider

    def critique_all(
        self,
        *,
        contract: ResearchContract,
        proposals: list[Proposal],
        evidence: list[Evidence],
    ) -> list[Critique]:
        return [
            self.critique(contract=contract, proposal=p, evidence=evidence)
            for p in proposals
        ]

    def critique(
        self,
        *,
        contract: ResearchContract,
        proposal: Proposal,
        evidence: list[Evidence],
    ) -> Critique:
        draft = (
            self._simulate(contract, proposal, evidence)
            if self.provider.is_simulated
            else self._via_llm(contract, proposal, evidence)
        )
        valid = {e.evidence_id for e in evidence}
        return Critique(
            proposal_id=proposal.proposal_id,
            counterexamples=draft.counterexamples,
            hidden_costs=draft.hidden_costs,
            unstated_dependencies=draft.unstated_dependencies,
            failure_modes=draft.failure_modes,
            severity=max(0, min(5, draft.severity)),
            evidence_ids=[e for e in draft.evidence_ids if e in valid],
            notes=draft.notes,
        )

    # ------------------------------------------------------------------

    def _via_llm(
        self, contract: ResearchContract, proposal: Proposal, evidence: list[Evidence]
    ) -> CritiqueDraft:
        ev = (
            "\n".join(
                f"- {e.evidence_id} [{e.stance.value}] {e.claim}" for e in evidence
            )
            or "(no evidence available)"
        )
        prompt = (
            f"GOAL (verbatim):\n{contract.original_idea}\n\n"
            f"HARD CONSTRAINTS:\n"
            + ("\n".join(f"- {c.key}: {c.description}" for c in contract.hard_constraints)
               or "(none)")
            + f"\n\nEVIDENCE POOL (cite only these ids):\n{ev}\n\n"
            f"CANDIDATE UNDER REVIEW:\n{proposal.card(max_items=8)}\n\n"
            "Find what would make this fail, with reasons."
        )
        return self.provider.complete_structured(
            system=_SYSTEM,
            prompt=prompt,
            schema=CritiqueDraft,
            purpose="skeptic.critique",
        )

    def _simulate(
        self, contract: ResearchContract, proposal: Proposal, evidence: list[Evidence]
    ) -> CritiqueDraft:
        """Deterministic structural critique.

        Finds real gaps by comparison rather than by invention: dependencies not
        covered by declared resources, hard constraints the candidate never
        mentions, refuting evidence it cites anyway, and missing falsifiers.
        """
        unstated: list[str] = []
        for dep in proposal.dependencies:
            if not any(
                tok in " ".join(contract.available_resources).lower()
                for tok in dep.lower().split()
                if len(tok) > 4
            ):
                unstated.append(
                    f"Dependency '{dep}' is not covered by any declared available "
                    "resource; obtaining it is unscheduled work."
                )

        hidden: list[str] = []
        if proposal.declared_cost_units >= 3:
            hidden.append(
                f"Declared effort of {proposal.declared_cost_units} units leaves little "
                "budget for the iteration that a route of this kind usually needs."
            )
        if len(proposal.dependencies) >= 2:
            hidden.append(
                "Each additional dependency carries ongoing maintenance and upgrade "
                "cost that is not reflected in the one-off estimate."
            )
        blob = " ".join(
            proposal.dependencies
            + proposal.premises
            + proposal.resource_needs
            + [proposal.core_mechanism, proposal.goal_mapping]
        ).lower()
        for c in contract.hard_constraints:
            if c.limit_value is not None:
                continue  # decided arithmetically by the constraint gate
            if not any(t in blob for t in c.description.lower().split() if len(t) > 5):
                hidden.append(
                    f"Hard constraint '{c.key}' ({c.description}) is never addressed by "
                    "this candidate, so its satisfaction is unverified."
                )

        counter: list[str] = []
        cited = set(proposal.evidence_ids)
        for e in evidence:
            if e.stance is Stance.REFUTES and e.evidence_id in cited:
                counter.append(
                    f"{e.evidence_id} argues against this direction: {e.claim}"
                )

        extra_modes: list[str] = []
        if not proposal.poc.falsification_condition.strip():
            extra_modes.append(
                "No falsification condition is stated, so the PoC cannot return a "
                "negative result -- it can only fail to finish."
            )
        if not proposal.evidence_ids:
            extra_modes.append(
                "The candidate cites no evidence at all; its premises are unsupported "
                "assertions."
            )
        if len(proposal.critical_assumptions) >= 3:
            extra_modes.append(
                f"{len(proposal.critical_assumptions)} critical assumptions must all "
                "hold simultaneously; the joint probability is lower than any single one."
            )

        severity = min(
            5, len(unstated) + len(counter) + len(extra_modes) // 2 + (
                1 if proposal.declared_cost_units >= 4 else 0
            )
        )
        notes = (
            "[SIMULATED] Structural critique derived by comparing the candidate "
            "against the contract (uncovered dependencies, unaddressed hard "
            "constraints, cited refuting evidence, missing falsifier). It contains no "
            "domain expertise and is not a substitute for expert review."
        )
        if not (unstated or hidden or counter or extra_modes):
            notes += " No structural gap was found; severity is low by absence of "
            notes += "findings, not by positive endorsement."

        return CritiqueDraft(
            counterexamples=counter,
            hidden_costs=hidden,
            unstated_dependencies=unstated,
            failure_modes=extra_modes,
            severity=severity,
            evidence_ids=[
                e.evidence_id
                for e in evidence
                if e.stance is Stance.REFUTES and e.evidence_id in cited
            ],
            notes=notes,
        )
