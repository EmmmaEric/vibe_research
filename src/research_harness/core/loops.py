"""The two feedback loops, as separate classes.

The spec requires that "did we build it right?" and "does the goal hold?" be
distinguishable *in code*, not just in prose. They are two classes with
different inputs, different permissions and different exit conditions:

:class:`ImplementationRepairLoop`
    Triggered only by conformance failure. May propose changes to the artefact.
    **Cannot touch the test plan or acceptance criteria** -- it has no reference
    to the registry's ``amend``, and the registry would refuse it anyway because
    the author role is an execution role. Bounded by ``max_repair_attempts``.

:class:`ResearchReplanningLoop`
    Triggered when the implementation is fine but the goal is not demonstrated.
    May gather more evidence, switch candidate, request external validation, or
    conclude. **Cannot lower the bar** either: changing the contract requires an
    explicit versioned revision with a reason, and ``original_idea`` is immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..agents.diagnosis import Action, Diagnosis
from ..agents.executor import Executor
from ..core.schemas import (
    DecisionRecord,
    ExperimentResult,
    TestPlan,
    ValidationResult,
    utcnow,
)
from ..experiments.adapters import ExperimentAdapter
from .state import ResearchState


@dataclass
class LoopOutcome:
    """What a loop iteration did and whether the run should keep going."""

    loop: str
    action: str
    progressed: bool
    detail: str
    experiment: ExperimentResult | None = None
    validation: ValidationResult | None = None
    should_continue: bool = True


class ImplementationRepairLoop:
    """Loop A: make the artefact match its own specification.

    Does not evaluate the research goal, and cannot weaken the criteria it is
    judged by. Its only success condition is: previously-failing conformance
    cases now pass, and nothing that passed before has regressed.
    """

    loop_name = "implementation"

    def __init__(self, executor: Executor) -> None:
        self.executor = executor

    def attempt(
        self,
        *,
        state: ResearchState,
        diagnosis: Diagnosis,
        adapter: ExperimentAdapter,
        plan: TestPlan,
        workdir: Path,
        experiment_index: int,
    ) -> LoopOutcome:
        contract = state.required_contract
        proposal = state.proposal(diagnosis_proposal(state, diagnosis))
        before = _conformance_pass_ids(state, plan)

        state.budget_state.repair_attempts += 1
        attempt_no = state.budget_state.repair_attempts

        # Re-run against the *same registered plan*. The executor verifies the
        # plan hash on both sides of the run, so a changed criterion is an error.
        experiment = self.executor.run(
            adapter=adapter,
            contract=contract,
            proposal=proposal,
            plan=plan,
            workdir=workdir,
            experiment_id=f"{state.run_id}-exp{experiment_index:03d}-repair{attempt_no}",
        )

        after = {
            o.test_id for o in experiment.outcomes if o.passed is True
        }
        regressed = sorted(before - after)
        newly_fixed = sorted(after - before)
        still_failing = [
            o.test_id for o in experiment.outcomes if o.passed is False
        ]

        if regressed:
            detail = (
                f"Repair attempt {attempt_no} fixed {len(newly_fixed)} case(s) but "
                f"regressed {len(regressed)} previously-passing case(s) "
                f"({', '.join(regressed)}). A net-negative repair is not progress."
            )
            progressed = False
        elif newly_fixed and not still_failing:
            detail = (
                f"Repair attempt {attempt_no} fixed {', '.join(newly_fixed)}; all "
                "conformance cases now pass with no regression."
            )
            progressed = True
        elif newly_fixed:
            detail = (
                f"Repair attempt {attempt_no} fixed {', '.join(newly_fixed)}, but "
                f"{len(still_failing)} case(s) still fail ({', '.join(still_failing)})."
            )
            progressed = True
        else:
            detail = (
                f"Repair attempt {attempt_no} changed nothing: {len(still_failing)} "
                "case(s) still fail. The defect is not where the diagnosis supposed."
            )
            progressed = False

        state.record_decision(
            DecisionRecord(
                decision_id=f"{state.run_id}-d{len(state.decisions) + 1:03d}",
                run_id=state.run_id,
                iteration=state.iteration,
                loop="implementation",
                hypothesis=diagnosis.hypothesis,
                action=f"{Action.REPAIR_IMPLEMENTATION.value} (attempt {attempt_no})",
                expected_result=diagnosis.expected_result,
                actual_result=detail,
                divergence_cause="" if progressed else "repair did not address the cause",
                continue_reason=diagnosis.rationale,
                should_continue=True,
                at=utcnow(),
            )
        )
        state.experiments.append(experiment)
        return LoopOutcome(
            loop=self.loop_name,
            action=Action.REPAIR_IMPLEMENTATION.value,
            progressed=progressed,
            detail=detail,
            experiment=experiment,
        )


class ResearchReplanningLoop:
    """Loop B: the artefact is right, but the goal is not demonstrated.

    Chooses among: gather more evidence, switch to the next candidate, hand off
    for external validation, or conclude. It never redefines success -- the
    contract's metrics and thresholds are the only bar, and altering them
    requires an explicit, reasoned, versioned revision.
    """

    loop_name = "research"

    def apply(
        self, *, state: ResearchState, diagnosis: Diagnosis
    ) -> LoopOutcome:
        action = diagnosis.action
        progressed = False
        should_continue = True

        if action is Action.GATHER_EVIDENCE:
            detail = (
                "Queued additional evidence gathering targeted at the named gap, "
                "rather than re-running the same experiment."
            )
            progressed = True

        elif action is Action.SWITCH_CANDIDATE:
            current = state.selected_proposal_id
            if current and current not in state.tried_proposal_ids:
                state.tried_proposal_ids.append(current)
            remaining = state.remaining_candidates()
            if remaining:
                state.selected_proposal_id = remaining[0]
                detail = (
                    f"Switched from {current} to {remaining[0]}: a structurally "
                    "different candidate is the remaining source of new information. "
                    f"{current} is retained in the record, not deleted."
                )
                progressed = True
            else:
                detail = (
                    f"No alternative candidate remains after {current}; the candidate "
                    "set considered is exhausted."
                )
                should_continue = False

        elif action is Action.REQUEST_EXTERNAL_VALIDATION:
            detail = (
                "Handed off to external validation: the open question needs real "
                "users, devices or physical measurement. Further internal iteration "
                "would spend budget without producing information."
            )
            should_continue = False

        elif action in (
            Action.CONCLUDE_INFEASIBLE,
            Action.CONCLUDE_INCONCLUSIVE,
            Action.CONCLUDE_SUCCESS,
        ):
            detail = diagnosis.rationale
            should_continue = False

        else:
            detail = f"No research action is defined for {action.value}."
            should_continue = False

        state.record_decision(
            DecisionRecord(
                decision_id=f"{state.run_id}-d{len(state.decisions) + 1:03d}",
                run_id=state.run_id,
                iteration=state.iteration,
                loop="research",
                hypothesis=diagnosis.hypothesis,
                action=action.value,
                expected_result=diagnosis.expected_result,
                actual_result=detail,
                divergence_cause=diagnosis.cause,
                continue_reason=diagnosis.rationale,
                should_continue=should_continue,
                at=utcnow(),
            )
        )
        return LoopOutcome(
            loop=self.loop_name,
            action=action.value,
            progressed=progressed,
            detail=detail,
            should_continue=should_continue,
        )


# --------------------------------------------------------------------------


def diagnosis_proposal(state: ResearchState, diagnosis: Diagnosis) -> str:
    """The candidate a diagnosis refers to, defaulting to the selected one."""
    v = state.latest_validation()
    if v is not None:
        return v.proposal_id
    if state.selected_proposal_id:
        return state.selected_proposal_id
    raise ValueError("no candidate is selected")


def _conformance_pass_ids(state: ResearchState, plan: TestPlan) -> set[str]:
    """Conformance cases that passed in the most recent experiment for this plan."""
    from ..experiments.validation import CONFORMANCE_CATEGORIES

    cats = {c.test_id: c.category for c in plan.cases}
    for exp in reversed(state.experiments):
        if exp.plan_id != plan.plan_id:
            continue
        return {
            o.test_id
            for o in exp.outcomes
            if o.passed is True and cats.get(o.test_id) in CONFORMANCE_CATEGORIES
        }
    return set()
