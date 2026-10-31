"""Independent validation.

The two questions the spec insists on keeping apart are separated here by *which
test categories answer them*:

* **Loop A -- "did we build what the proposal specified?"** is answered by the
  ``contract``, ``correctness`` and ``edge_robustness`` cases. These are about
  conformance of the artefact to its own specification.
* **Loop B -- "does the original goal hold?"** is answered only by
  ``goal_acceptance`` cases and the contract's metric thresholds.

Keeping them disjoint is what stops "the tests pass" from being mistaken for
"the research goal is met". A candidate can be perfectly implemented and still
fail the goal; that is a research result, not a bug to repair.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.schemas import (
    ExperimentResult,
    Proposal,
    ResearchContract,
    TestPlan,
    ValidationResult,
)

#: Categories that speak to implementation conformance (Loop A).
CONFORMANCE_CATEGORIES = ("contract", "correctness", "edge_robustness")
#: Categories that speak to goal attainment (Loop B).
GOAL_CATEGORIES = ("goal_acceptance",)


@runtime_checkable
class ValidationAdapter(Protocol):
    adapter_id: str

    def validate(
        self,
        *,
        contract: ResearchContract,
        proposal: Proposal,
        plan: TestPlan,
        experiment: ExperimentResult,
        **kwargs: object,
    ) -> ValidationResult: ...


def check_implementation_conformance(
    *, plan: TestPlan, experiment: ExperimentResult
) -> tuple[bool | None, str, list[str]]:
    """Answer Loop A from conformance-category outcomes only.

    Returns ``(conforms, notes, failed_test_ids)``. ``conforms`` is ``None`` when
    no conformance test actually ran -- which is different from conforming, and
    is reported as such rather than defaulted either way.
    """
    cats = {c.test_id: c.category for c in plan.cases}
    relevant = [
        o
        for o in experiment.outcomes
        if cats.get(o.test_id) in CONFORMANCE_CATEGORIES
    ]
    ran = [o for o in relevant if o.passed is not None]
    failed = [o.test_id for o in ran if o.passed is False]

    if not relevant:
        return (
            None,
            "The test plan contains no implementation-conformance cases, so "
            "conformance was not assessed.",
            [],
        )
    if not ran:
        return (
            None,
            f"None of the {len(relevant)} conformance case(s) were executed, so "
            "whether the implementation matches the proposal is unknown.",
            [],
        )
    if failed:
        return (
            False,
            f"{len(failed)} of {len(ran)} executed conformance case(s) failed "
            f"({', '.join(failed)}): the artefact does not match the candidate's own "
            "specification. This is an implementation defect, to be handled by the "
            "implementation-repair loop -- it says nothing yet about the research goal.",
            failed,
        )
    skipped = len(relevant) - len(ran)
    note = f"All {len(ran)} executed conformance case(s) passed."
    if skipped:
        note += (
            f" {skipped} further conformance case(s) did not run, so conformance is "
            "demonstrated only over what was executed."
        )
    if experiment.is_simulated:
        note += (
            " NOTE: these verdicts come from a SIMULATED adapter and are not evidence "
            "that any real artefact conforms."
        )
    return True, note, []


def goal_test_summary(
    *, plan: TestPlan, experiment: ExperimentResult
) -> tuple[list[str], list[str], list[str]]:
    """Split goal-acceptance cases into ``(passed, failed, not_run)``."""
    cats = {c.test_id: c.category for c in plan.cases}
    passed: list[str] = []
    failed: list[str] = []
    not_run: list[str] = []
    for o in experiment.outcomes:
        if cats.get(o.test_id) not in GOAL_CATEGORIES:
            continue
        if o.passed is True:
            passed.append(o.test_id)
        elif o.passed is False:
            failed.append(o.test_id)
        else:
            not_run.append(o.test_id)
    return passed, failed, not_run


def evidence_quality_gaps(
    *, contract: ResearchContract, experiment: ExperimentResult
) -> list[str]:
    """Collect reasons the evidence is insufficient to support a conclusion."""
    gaps: list[str] = []
    if experiment.is_simulated:
        gaps.append(
            "All experimental results come from a simulated adapter; no real "
            "measurement of the goal exists."
        )
    if not experiment.observations:
        gaps.append("No metric observation was recorded at all.")
    for obs in experiment.observations:
        if obs.n_repeats < 3 and obs.stddev is None:
            gaps.append(
                f"Metric '{obs.metric_key}' was measured {obs.n_repeats} time(s) with no "
                "variance recorded; a single reading of a variable quantity is not a "
                "measurement."
            )
    measured = {o.metric_key for o in experiment.observations}
    for m in contract.success_metrics:
        if m.key not in measured:
            gaps.append(f"Contract metric '{m.key}' was never measured.")
        elif m.target_threshold is None:
            gaps.append(
                f"Contract metric '{m.key}' has no numeric threshold, so no observed "
                "value can establish whether the goal was met."
            )
    return list(dict.fromkeys(gaps))
