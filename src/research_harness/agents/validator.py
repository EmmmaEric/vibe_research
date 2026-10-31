"""Independent Validator role.

Decides, from raw artefacts and the contract, whether the implementation
conforms and whether the original goal is met. It never reads a success claim
from the Executor -- its inputs are the registered test plan, the recorded test
outcomes, and the measured observations.

All thresholds comparisons and the validation-level decision are deterministic
code in :mod:`research_harness.evaluation.decision_engine`; this role assembles
the inputs and attaches the external validation plan for whatever could not be
settled here.
"""

from __future__ import annotations

from ..core.schemas import (
    ConstraintStatus,
    Evidence,
    ExperimentResult,
    Proposal,
    ResearchContract,
    TestPlan,
    ValidationResult,
)
from ..evaluation.decision_engine import build_validation_result
from ..experiments.adapters import external_plan_steps
from ..experiments.planner import TestPlanRegistry
from ..experiments.validation import (
    check_implementation_conformance,
    evidence_quality_gaps,
)

ROLE = "independent_validator"


class IndependentValidator:
    adapter_id = "independent-validator-v1"

    def __init__(self, registry: TestPlanRegistry | None = None) -> None:
        self.registry = registry
        self.role = ROLE

    def validate(
        self,
        *,
        contract: ResearchContract,
        proposal: Proposal,
        plan: TestPlan,
        experiment: ExperimentResult,
        constraint_status: ConstraintStatus,
        evidence: list[Evidence],
    ) -> ValidationResult:
        # The result is only interpretable against the plan it was run under.
        if self.registry is not None:
            self.registry.verify_unchanged(plan)
        if experiment.plan_hash != plan.content_hash():
            raise ValueError(
                f"experiment {experiment.experiment_id} was run against plan hash "
                f"{experiment.plan_hash}, but the plan now hashes to "
                f"{plan.content_hash()}; the result cannot be attributed to these "
                "acceptance criteria"
            )

        conforms, notes, failed = check_implementation_conformance(
            plan=plan, experiment=experiment
        )
        gaps = evidence_quality_gaps(contract=contract, experiment=experiment)
        has_real_evidence = any(e.is_real for e in evidence)
        if not has_real_evidence:
            gaps.append(
                "No non-simulated evidence was gathered for this run, so even a "
                "passing result rests on an unverified premise base."
            )

        return build_validation_result(
            contract=contract,
            proposal_id=proposal.proposal_id,
            experiment=experiment,
            implementation_conforms=conforms,
            implementation_notes=notes,
            failed_test_ids=failed,
            constraint_status=constraint_status,
            has_real_evidence=has_real_evidence,
            has_test_plan=bool(plan.cases),
            external_plan=external_plan_steps(plan),
            evidence_gaps=gaps,
        )
