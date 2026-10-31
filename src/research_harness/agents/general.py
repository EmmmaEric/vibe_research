"""General Research Agent (spec section 4.1).

The default path. One capable agent investigates the goal, builds a problem
tree, forms hypotheses, drafts candidate directions, and — crucially — reports
*signals* rather than deciding what happens next. The
:class:`~research_harness.core.router.AdaptiveRouter` owns the decision; this
agent owns the research.

That split is deliberate. An agent that both assesses its own uncertainty and
chooses whether to keep going will tend to conclude it is nearly finished. Here
it must state how uncertain it is in structured fields, and a rule table reads
them.

The agent also has a standing obligation the spec calls out: look for
counterexamples and failure conditions, not only for support. A report with no
open questions and no disconfirming angle is treated as a warning sign, not as
success.
"""

from __future__ import annotations

from pydantic import Field

from ..core.schemas import (
    Base,
    Evidence,
    ResearchContract,
    RouteAction,
    Stance,
    TaskNode,
)
from . import domains

_SYSTEM = """\
You are a General Research Agent. You investigate a goal end to end: find \
evidence, decompose the problem, form hypotheses, and sketch candidate routes.

Rules:
- Report what you actually found, with the evidence ids you relied on. Never \
invent a source.
- Look for what would make the goal unreachable, not only for support. State \
counterexamples and failure conditions explicitly.
- Report your uncertainty honestly in the structured fields. Do not decide \
whether the run should continue -- that is not your call, and over-claiming \
confidence here directly degrades the routing decision.
- `critical_unknowns` are facts that, if false, would change the conclusion. \
Listing none when some exist is the most damaging error you can make.
- `complexity_score` is 0..1: how much specialist depth or how many \
interacting concerns this goal involves.
"""


class GeneralAgentReport(Base):
    """Structured output of one general-agent step.

    Everything here is either an artefact (findings, hypotheses, task graph) or
    a *signal* the router reads. No field tells the orchestrator what to do.
    """

    findings: str = ""
    hypotheses: list[str] = Field(default_factory=list)
    candidate_directions: list[str] = Field(default_factory=list)
    counterexamples: list[str] = Field(default_factory=list)
    critical_unknowns: list[str] = Field(default_factory=list)
    cheapest_decisive_test: str = ""
    cheap_test_available: bool = False
    complexity_score: float = 0.0
    expert_review_needed: bool = False
    information_sufficient: bool = False
    task_graph: list[TaskNode] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    #: Advisory only; the router records it and may override it.
    suggested_next_action: RouteAction | None = None
    rationale: str = ""


class GeneralResearchAgent:
    """Single capable agent: research, decompose, hypothesise, sketch."""

    def __init__(self, provider) -> None:
        self.provider = provider

    def solve(
        self,
        *,
        contract: ResearchContract,
        evidence: list[Evidence],
        memory_notes: list[str] | None = None,
    ) -> GeneralAgentReport:
        report = (
            self._simulate(contract, evidence)
            if self.provider.is_simulated
            else self._via_llm(contract, evidence, memory_notes or [])
        )
        # Drop any evidence id the agent did not actually have.
        valid = {e.evidence_id for e in evidence}
        report = report.model_copy(
            update={"evidence_ids": [e for e in report.evidence_ids if e in valid]}
        )
        return report

    # ------------------------------------------------------------------

    def _via_llm(
        self,
        contract: ResearchContract,
        evidence: list[Evidence],
        memory_notes: list[str],
    ) -> GeneralAgentReport:
        ev = (
            "\n".join(
                f"- {e.evidence_id} [{e.source_type}/{e.stance.value}/"
                f"{e.provenance.value}] {e.claim}"
                for e in evidence
            )
            or "(no evidence gathered yet)"
        )
        metrics = (
            "\n".join(
                f"- {m.key}: {m.description} [{m.unit}], target {m.direction.value} "
                f"{m.target_threshold}{'' if m.required else ' (optional)'}"
                for m in contract.success_metrics
            )
            or "(none declared)"
        )
        prior = (
            "\n\nPRIOR RELATED WORK FROM MEMORY (similarity is not evidence; check "
            "applicability):\n" + "\n".join(f"- {n}" for n in memory_notes[:5])
            if memory_notes
            else ""
        )
        prompt = (
            f"GOAL (verbatim, authoritative):\n{contract.original_idea}\n\n"
            f"PROBLEM STATEMENT:\n{contract.problem_statement}\n\n"
            f"SUCCESS METRICS:\n{metrics}\n\n"
            f"HARD CONSTRAINTS:\n"
            + ("\n".join(f"- {c.key}: {c.description}" for c in contract.hard_constraints)
               or "(none)")
            + f"\n\nOPEN QUESTIONS FROM THE CONTRACT:\n"
            + ("\n".join(f"- {u}" for u in contract.unknowns) or "(none recorded)")
            + f"\n\nEVIDENCE AVAILABLE (cite only these ids):\n{ev}{prior}\n\n"
            "Investigate and report. Include what would refute the premise."
        )
        return self.provider.complete_structured(
            system=_SYSTEM,
            prompt=prompt,
            schema=GeneralAgentReport,
            purpose="general_agent.solve",
        )

    def _simulate(
        self, contract: ResearchContract, evidence: list[Evidence]
    ) -> GeneralAgentReport:
        """Deterministic signal derivation from the contract and evidence pool.

        Signals come from counting real structure — how much real evidence
        exists, how many metrics lack a threshold, how many hard constraints are
        unaddressed — rather than from a model's self-assessment. That makes the
        offline router path exercise the same rules on comparable numbers.
        """
        profile = domains.select_profile(
            contract.original_idea, hint=contract.domain_hint
        )
        real = [e for e in evidence if e.is_real]
        supporting = [e for e in real if e.stance is Stance.SUPPORTS]
        refuting = [e for e in real if e.stance is Stance.REFUTES]

        # Critical unknowns: anything that would change the conclusion if false.
        unknowns: list[str] = []
        for m in contract.success_metrics:
            if m.target_threshold is None and m.required:
                unknowns.append(
                    f"Required metric '{m.key}' has no numeric target, so goal "
                    "attainment cannot be decided on it."
                )
        if not real:
            unknowns.append(
                "No non-simulated evidence exists; every premise is unverified."
            )
        for c in contract.hard_constraints:
            if c.limit_value is None and c.assumed:
                unknowns.append(
                    f"Hard constraint '{c.key}' is a harness assumption and has not "
                    "been confirmed with the requester."
                )

        # Complexity: interacting concerns, not a model's impression.
        concerns = (
            len(contract.success_metrics)
            + len(contract.hard_constraints)
            + len(contract.unknowns)
        )
        complexity = min(1.0, round(concerns / 12.0, 3))

        # Problem tree: one node per metric plus feasibility and cost, with the
        # integration node depending on the rest.
        tasks: list[TaskNode] = []
        for i, m in enumerate(contract.success_metrics, start=1):
            tasks.append(
                TaskNode(
                    task_id=f"T{i}",
                    question=f"How can '{m.key}' ({m.description}) be moved and measured?",
                    evidence_ids=[e.evidence_id for e in supporting[:2]],
                )
            )
        tasks.append(
            TaskNode(
                task_id="TF",
                question="Is the approach feasible with the declared resources?",
                evidence_ids=[e.evidence_id for e in real[:2]],
            )
        )
        tasks.append(
            TaskNode(
                task_id="TR",
                question="What would refute the premise, and has anyone reported it?",
                evidence_ids=[e.evidence_id for e in refuting[:2]],
            )
        )
        tasks.append(
            TaskNode(
                task_id="TI",
                question="Do the findings jointly support the original goal?",
                depends_on=[t.task_id for t in tasks],
            )
        )

        counter = [
            f"{e.evidence_id}: {e.claim}" for e in refuting[:3]
        ] or [
            "No disconfirming evidence was found. Disconfirming queries were issued, "
            "so this may mean the search is too narrow rather than that no objection "
            "exists."
        ]

        mechanisms = dict(profile.mechanisms)
        directions = [
            f"{archetype}: {mechanisms.get(archetype, 'unspecified mechanism')}"
            for archetype in profile.archetypes
        ]

        cheap_test = ""
        cheap_available = False
        if contract.success_metrics and "pytest" in profile.preferred_adapters:
            cheap_test = (
                f"Measure a baseline for "
                f"{', '.join(m.key for m in contract.success_metrics[:2])} and compare "
                "a minimal implementation against it under identical conditions."
            )
            cheap_available = True
        elif unknowns:
            cheap_test = (
                f"Resolve the cheapest open question first: {unknowns[0]}"
            )

        findings = (
            f"[SIMULATED] Investigated '{contract.original_idea.strip()[:120]}' under "
            f"domain profile '{profile.name}'. Gathered {len(real)} non-simulated "
            f"item(s) ({len(supporting)} supporting, {len(refuting)} refuting) across "
            f"{len(evidence)} total. Decomposed into {len(tasks)} question(s). "
            f"Identified {len(unknowns)} critical unknown(s). Derived offline from "
            "contract structure and the evidence pool; contains no domain expertise."
        )

        # Advisory suggestion only. Deliberately naive, so a test can show the
        # router overriding it when a rule says otherwise.
        if unknowns:
            suggestion = RouteAction.GATHER_EVIDENCE
        elif cheap_available:
            suggestion = RouteAction.RUN_EXPERIMENT
        else:
            suggestion = RouteAction.CONTINUE_SINGLE_AGENT

        return GeneralAgentReport(
            findings=findings,
            hypotheses=[
                f"The goal can be reached by {d}" for d in directions
            ],
            candidate_directions=directions,
            counterexamples=counter,
            critical_unknowns=list(dict.fromkeys(unknowns)),
            cheapest_decisive_test=cheap_test,
            cheap_test_available=cheap_available,
            complexity_score=complexity,
            expert_review_needed=bool(
                contract.hard_constraints
                and any(c.category in ("legal", "safety") for c in contract.hard_constraints)
            ),
            information_sufficient=bool(real) and not unknowns,
            task_graph=tasks,
            evidence_ids=[e.evidence_id for e in real],
            suggested_next_action=suggestion,
            rationale=(
                "[SIMULATED] Signals derived by counting contract structure and "
                "evidence provenance, not by model self-assessment."
            ),
        )


def ready_tasks(graph: list[TaskNode]) -> list[TaskNode]:
    """Tasks whose dependencies are satisfied — the parallelisable frontier.

    Only genuinely independent sub-questions may run concurrently; everything
    must still be integrated against the one original goal (spec section 4.3).
    """
    done = {t.task_id for t in graph if t.status == "done"}
    return [t for t in graph if t.status == "pending" and t.is_ready(done)]
