"""Test Designer: builds the verification plan from the CONTRACT.

Two isolation properties the spec demands, enforced here and in
:mod:`research_harness.experiments.planner`:

* The plan is derived from the contract's success metrics and constraints, **not**
  from the implementation. It is authored before execution.
* Goal-acceptance criteria restate the contract's thresholds verbatim. A
  candidate's own definition of "done" never becomes an acceptance criterion.

All six categories from the spec are covered: contract, correctness,
edge/robustness, comparison, goal acceptance, and evidence quality.
"""

from __future__ import annotations

from pydantic import Field

from ..core.schemas import (
    Base,
    Proposal,
    ResearchContract,
    TestCase,
    TestPlan,
)
from . import domains

_SYSTEM = """\
You are a Test Designer. You write the verification plan for a candidate, BEFORE \
it is implemented, working from the research contract.

Rules:
- Goal-acceptance criteria must restate the contract's own metric thresholds. \
Never substitute the candidate's self-declared success criterion.
- Every test needs a procedure and a concrete acceptance criterion someone else \
could apply without asking you.
- Comparison tests need a named independent baseline.
- Mark tests that need a resource this harness may not have \
(requires_external_resource), rather than quietly assuming they will run.
- You are writing the test that could FAIL the candidate. Design it to be capable \
of that.
"""


class _CaseOut(Base):
    category: str
    description: str
    procedure: str
    acceptance_criterion: str
    measures_metric: str | None = None
    baseline: str | None = None
    executable: bool = False
    requires_external_resource: str | None = None


class TestPlanDraft(Base):
    cases: list[_CaseOut] = Field(default_factory=list)
    independent_baseline: str = ""
    randomness_controls: str = ""


_VALID = {
    "contract",
    "correctness",
    "edge_robustness",
    "comparison",
    "goal_acceptance",
    "evidence_quality",
}


class TestDesigner:
    def __init__(self, provider) -> None:
        self.provider = provider

    def design(
        self, *, contract: ResearchContract, proposal: Proposal, version: int = 1
    ) -> TestPlan:
        draft = (
            self._simulate(contract, proposal)
            if self.provider.is_simulated
            else self._via_llm(contract, proposal)
        )

        cases: list[TestCase] = []
        for i, c in enumerate(draft.cases, start=1):
            cat = c.category if c.category in _VALID else "correctness"
            cases.append(
                TestCase(
                    test_id=f"T{i:03d}",
                    category=cat,  # type: ignore[arg-type]
                    description=c.description,
                    procedure=c.procedure,
                    acceptance_criterion=c.acceptance_criterion,
                    measures_metric=c.measures_metric,
                    baseline=c.baseline,
                    executable=c.executable,
                    requires_external_resource=c.requires_external_resource,
                )
            )

        # Guarantee a goal-acceptance test per contract metric, whatever the
        # author produced. These are the tests that decide goal attainment, so
        # their presence cannot be left to the model's discretion.
        covered = {c.measures_metric for c in cases if c.category == "goal_acceptance"}
        for m in contract.success_metrics:
            if m.key in covered:
                continue
            cases.append(
                TestCase(
                    test_id=f"T{len(cases) + 1:03d}",
                    category="goal_acceptance",
                    description=f"Independent measurement of contract metric '{m.key}'",
                    procedure=m.measurement_method,
                    acceptance_criterion=(
                        f"{m.key} {'>=' if m.direction.value == 'maximize' else '<='} "
                        f"{m.target_threshold} {m.unit}"
                        if m.target_threshold is not None
                        else (
                            f"UNDECIDABLE: contract declares no numeric threshold for "
                            f"'{m.key}', so goal attainment cannot be established on it"
                        )
                    ),
                    measures_metric=m.key,
                    executable=False,
                    requires_external_resource=(
                        None
                        if m.target_threshold is not None
                        else "a numeric target from the requester"
                    ),
                )
            )

        return TestPlan(
            plan_id=f"{contract.run_id}-{proposal.proposal_id}-v{version}",
            run_id=contract.run_id,
            proposal_id=proposal.proposal_id,
            version=version,
            derived_from_contract_version=contract.version,
            cases=cases,
            independent_baseline=draft.independent_baseline,
            randomness_controls=draft.randomness_controls,
        )

    # ------------------------------------------------------------------

    def _via_llm(self, contract: ResearchContract, proposal: Proposal) -> TestPlanDraft:
        metrics = (
            "\n".join(
                f"- {m.key}: {m.description} [{m.unit}], target {m.direction.value} "
                f"{m.target_threshold}, measured by: {m.measurement_method}"
                for m in contract.success_metrics
            )
            or "(none declared)"
        )
        prompt = (
            f"CONTRACT GOAL (verbatim):\n{contract.original_idea}\n\n"
            f"SUCCESS METRICS (acceptance criteria must restate these):\n{metrics}\n\n"
            f"HARD CONSTRAINTS:\n"
            + ("\n".join(f"- {c.key}: {c.description}" for c in contract.hard_constraints)
               or "(none)")
            + f"\n\nCANDIDATE TO BE TESTED:\n{proposal.card(max_items=8)}\n\n"
            f"Categories to cover: {', '.join(sorted(_VALID))}.\n"
            "Write the plan."
        )
        return self.provider.complete_structured(
            system=_SYSTEM,
            prompt=prompt,
            schema=TestPlanDraft,
            purpose="test_designer.design",
        )

    def _simulate(
        self, contract: ResearchContract, proposal: Proposal
    ) -> TestPlanDraft:
        """Deterministic plan covering all six categories."""
        profile = domains.select_profile(
            contract.original_idea, hint=contract.domain_hint
        )
        cases: list[_CaseOut] = []

        cases.append(
            _CaseOut(
                category="contract",
                description="Declared interfaces, inputs and outputs are present and well-formed",
                procedure=(
                    "Instantiate the artefact and assert each interface named in the "
                    "candidate's key steps exists with the declared shape."
                ),
                acceptance_criterion="Every declared interface is present and callable.",
                executable=True,
            )
        )
        cases.append(
            _CaseOut(
                category="correctness",
                description="The mechanism produces correct results on known-answer cases",
                procedure=(
                    "Run the artefact against inputs whose correct output is known "
                    "independently of the implementation."
                ),
                acceptance_criterion="All known-answer cases match the expected output exactly.",
                executable=True,
            )
        )
        cases.append(
            _CaseOut(
                category="edge_robustness",
                description="Behaviour at boundaries and under the candidate's own stated failure modes",
                procedure=(
                    "Exercise empty, minimal, maximal and malformed inputs, plus each "
                    "failure mode the candidate declared: "
                    + ("; ".join(proposal.failure_modes[:3]) or "(none declared)")
                ),
                acceptance_criterion=(
                    "No crash or silent wrong answer; each failure is detected and "
                    "reported rather than absorbed."
                ),
                executable=True,
            )
        )
        baseline = (
            f"The pre-existing behaviour measured before any change, recorded "
            f"independently of the candidate ({profile.name} domain)."
        )
        cases.append(
            _CaseOut(
                category="comparison",
                description="Comparison against an independent baseline",
                procedure=(
                    "Measure baseline and candidate under identical conditions, same "
                    "inputs, same environment, with repeats."
                ),
                acceptance_criterion=(
                    "The candidate's measured value improves on the baseline by more "
                    "than the measurement variance."
                ),
                baseline=baseline,
                executable=True,
            )
        )
        for m in contract.success_metrics:
            cases.append(
                _CaseOut(
                    category="goal_acceptance",
                    description=f"Independent measurement of contract metric '{m.key}'",
                    procedure=m.measurement_method,
                    acceptance_criterion=(
                        f"{m.key} {'>=' if m.direction.value == 'maximize' else '<='} "
                        f"{m.target_threshold} {m.unit}"
                        if m.target_threshold is not None
                        else f"UNDECIDABLE: no numeric threshold declared for '{m.key}'"
                    ),
                    measures_metric=m.key,
                    executable=False,
                    requires_external_resource=(
                        None
                        if m.target_threshold is not None
                        else "a numeric target from the requester"
                    ),
                )
            )
        cases.append(
            _CaseOut(
                category="evidence_quality",
                description="Is the gathered evidence sufficient to support the conclusion?",
                procedure=(
                    "Check that every scored dimension cites non-simulated evidence, "
                    "that repeat counts and variance are recorded, and that no "
                    "conclusion rests solely on simulated data."
                ),
                acceptance_criterion=(
                    "No conclusion depends on simulated evidence; every quantitative "
                    "claim has a recorded provenance and repeat count."
                ),
                executable=True,
            )
        )
        for note in profile.validation_notes:
            if any(k in note for k in ("hardware", "interview", "A/B", "certified", "real")):
                cases.append(
                    _CaseOut(
                        category="goal_acceptance",
                        description=f"Real-world validation step: {note}",
                        procedure=f"Perform {note} with the real subject or device.",
                        acceptance_criterion=(
                            "Result recorded with provenance 'human' or 'measured', "
                            "obtained outside this harness."
                        ),
                        executable=False,
                        requires_external_resource=note,
                    )
                )

        return TestPlanDraft(
            cases=cases,
            independent_baseline=baseline,
            randomness_controls=(
                "Fix and record the random seed for every stochastic step; run at least "
                "3 repeats; report mean and standard deviation. A single run of a "
                "stochastic process is not a measurement."
            ),
        )
