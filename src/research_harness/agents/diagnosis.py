"""Diagnosis: failure attribution and next-action selection.

The *choice of next action* is a control-flow gate, so it is deterministic code
rather than a model judgement. The decision tree distinguishes the two loops:

* implementation defect  -> repair (bounded by ``max_repair_attempts``)
* goal shortfall         -> research replanning (gather evidence, switch
                            candidate, or conclude)

The tree also refuses to repeat an action that has already been tried and failed
for the same candidate, which is what keeps a failing route from being retried
forever.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ..core.schemas import MaturityLevel, ValidationResult


class Action(str, enum.Enum):
    REPAIR_IMPLEMENTATION = "repair_implementation"
    GATHER_EVIDENCE = "gather_evidence"
    SWITCH_CANDIDATE = "switch_candidate"
    REQUEST_EXTERNAL_VALIDATION = "request_external_validation"
    CONCLUDE_INFEASIBLE = "conclude_infeasible"
    CONCLUDE_SUCCESS = "conclude_success"
    CONCLUDE_INCONCLUSIVE = "conclude_inconclusive"


@dataclass
class Diagnosis:
    action: Action
    loop: str  # "implementation" | "research" | "orchestrator"
    cause: str
    hypothesis: str
    expected_result: str
    rationale: str
    exhausted: list[str] = field(default_factory=list)


class DiagnosisAgent:
    """Attributes a failure and picks the next most valuable action."""

    def __init__(self) -> None:
        #: ``(proposal_id, action)`` pairs already attempted without success.
        self._tried: set[tuple[str, Action]] = set()

    def record_attempt(self, proposal_id: str, action: Action) -> None:
        self._tried.add((proposal_id, action))

    def tried(self, proposal_id: str, action: Action) -> bool:
        return (proposal_id, action) in self._tried

    # ------------------------------------------------------------------

    def diagnose(
        self,
        *,
        validation: ValidationResult,
        repair_attempts: int,
        max_repair_attempts: int,
        remaining_candidates: list[str],
        has_real_evidence: bool,
        evidence_exhausted: bool,
    ) -> Diagnosis:
        pid = validation.proposal_id
        exhausted = [
            a.value for (p, a) in sorted(self._tried, key=lambda t: t[1].value) if p == pid
        ]

        # 1. Goal genuinely met.
        if validation.goal_satisfied is True:
            return Diagnosis(
                action=Action.CONCLUDE_SUCCESS,
                loop="orchestrator",
                cause="none",
                hypothesis="The selected candidate meets the contract's goal.",
                expected_result="Report the result with its evidence chain.",
                rationale=validation.blocking_reason,
                exhausted=exhausted,
            )

        # 2. Hard-constraint violation -- not repairable, not retryable.
        if validation.maturity_level is MaturityLevel.REJECTED:
            if remaining_candidates:
                return Diagnosis(
                    action=Action.SWITCH_CANDIDATE,
                    loop="research",
                    cause="hard_constraint_violation",
                    hypothesis=(
                        f"{pid} violates a hard constraint; a different candidate may not."
                    ),
                    expected_result=f"Evaluate {remaining_candidates[0]} instead.",
                    rationale=(
                        "Constraint violations are non-compensatory and cannot be "
                        "repaired by iteration on the same candidate."
                    ),
                    exhausted=exhausted,
                )
            return Diagnosis(
                action=Action.CONCLUDE_INFEASIBLE,
                loop="orchestrator",
                cause="hard_constraint_violation",
                hypothesis="Every candidate considered violates a hard constraint.",
                expected_result="Report infeasibility over the candidate set considered.",
                rationale=validation.blocking_reason,
                exhausted=exhausted,
            )

        # 3. Loop A: built the wrong thing. Repair, while budget allows.
        if validation.implementation_conforms is False:
            if repair_attempts < max_repair_attempts:
                return Diagnosis(
                    action=Action.REPAIR_IMPLEMENTATION,
                    loop="implementation",
                    cause="implementation_defect",
                    hypothesis=(
                        f"Conformance case(s) {', '.join(validation.failed_test_ids)} "
                        "fail because of an implementation defect, not because the "
                        "approach is wrong."
                    ),
                    expected_result=(
                        "After the minimal fix, the failing conformance cases pass and "
                        "no previously passing case regresses."
                    ),
                    rationale=(
                        f"Repair attempt {repair_attempts + 1} of "
                        f"{max_repair_attempts}. Only the implementation may change; "
                        "the test plan and acceptance criteria stay fixed."
                    ),
                    exhausted=exhausted,
                )
            if remaining_candidates:
                return Diagnosis(
                    action=Action.SWITCH_CANDIDATE,
                    loop="research",
                    cause="implementation_defect_persistent",
                    hypothesis=(
                        "Repeated conformance failure suggests the approach itself is "
                        "hard to realise, not that one bug remains."
                    ),
                    expected_result=f"Evaluate {remaining_candidates[0]} instead.",
                    rationale=(
                        f"Repair budget of {max_repair_attempts} attempt(s) exhausted "
                        "without conformance."
                    ),
                    exhausted=exhausted,
                )
            return Diagnosis(
                action=Action.CONCLUDE_INCONCLUSIVE,
                loop="orchestrator",
                cause="implementation_defect_persistent",
                hypothesis="No candidate could be implemented to its own specification.",
                expected_result="Report that nothing was validated, with the failures.",
                rationale="Repair budget exhausted and no alternative candidate remains.",
                exhausted=exhausted,
            )

        # 4. Cannot be settled here -- real-world input needed.
        if (
            validation.maturity_level is MaturityLevel.INCONCLUSIVE
            and validation.external_validation_plan
        ):
            return Diagnosis(
                action=Action.REQUEST_EXTERNAL_VALIDATION,
                loop="research",
                cause="requires_real_world_input",
                hypothesis=(
                    "The open question is empirical and needs real users, devices or "
                    "physical measurement."
                ),
                expected_result=(
                    "A human performs the external validation steps and records the "
                    "results as human-provenance evidence."
                ),
                rationale=(
                    "Continuing to iterate inside the harness cannot resolve this; "
                    "more simulation would add cost and no information."
                ),
                exhausted=exhausted,
            )

        # 5. Loop B: implementation fine, goal not demonstrated.
        if not evidence_exhausted and not self.tried(pid, Action.GATHER_EVIDENCE):
            missing = ", ".join(validation.evidence_gaps[:2]) or "unspecified gaps"
            return Diagnosis(
                action=Action.GATHER_EVIDENCE,
                loop="research",
                cause=(
                    "insufficient_evidence"
                    if not has_real_evidence
                    else "goal_not_demonstrated"
                ),
                hypothesis=(
                    "The goal is not demonstrated because of missing evidence or "
                    f"measurement rather than a wrong approach. Gaps: {missing}"
                ),
                expected_result=(
                    "Closing the named gap either demonstrates the goal or refutes it, "
                    "both of which are progress."
                ),
                rationale=(
                    "Resolving the cheapest decision-relevant uncertainty is worth more "
                    "than re-running the same experiment."
                ),
                exhausted=exhausted,
            )

        if remaining_candidates:
            return Diagnosis(
                action=Action.SWITCH_CANDIDATE,
                loop="research",
                cause="goal_not_demonstrated",
                hypothesis=(
                    f"{pid} is correctly built but does not reach the goal; a "
                    "structurally different candidate may."
                ),
                expected_result=f"Evaluate {remaining_candidates[0]} against the same plan.",
                rationale=(
                    "Evidence gathering for this candidate is exhausted; switching "
                    "mechanism class is the remaining source of new information."
                ),
                exhausted=exhausted,
            )

        return Diagnosis(
            action=Action.CONCLUDE_INCONCLUSIVE,
            loop="orchestrator",
            cause="evidence_exhausted",
            hypothesis=(
                "No remaining action would change the conclusion within budget."
            ),
            expected_result=(
                "Report the current evidence level honestly, with the gaps that remain."
            ),
            rationale=(
                "Every candidate and evidence avenue available has been tried. "
                "Reporting an unresolved question is the correct outcome; "
                "manufacturing a conclusion is not."
            ),
            exhausted=exhausted,
        )
