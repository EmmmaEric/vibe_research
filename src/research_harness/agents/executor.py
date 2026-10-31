"""Executor role: runs experiments inside the authorised environment.

Deliberately thin and deliberately powerless over the criteria it is judged by.
The Executor:

* may run an :class:`~research_harness.experiments.adapters.ExperimentAdapter`;
* must register-and-verify the test plan hash around every run, so a plan that
  changed during execution is an error rather than a silent re-baselining;
* may NOT amend the test plan -- :class:`TestPlanRegistry` refuses amendments
  from execution roles;
* may NOT declare success. It reports raw outcomes; the Validator decides.

Approval gating for anything with real-world cost happens before this role is
reached, in the orchestrator.
"""

from __future__ import annotations

from pathlib import Path

from ..core.schemas import (
    ExperimentResult,
    Proposal,
    ResearchContract,
    TestPlan,
)
from ..experiments.adapters import ExperimentAdapter
from ..experiments.planner import TestPlanRegistry

ROLE = "executor"


class Executor:
    def __init__(self, registry: TestPlanRegistry) -> None:
        self.registry = registry
        self.role = ROLE

    def run(
        self,
        *,
        adapter: ExperimentAdapter,
        contract: ResearchContract,
        proposal: Proposal,
        plan: TestPlan,
        workdir: Path,
        experiment_id: str,
    ) -> ExperimentResult:
        """Execute one experiment against a registered, unmodified plan."""
        if self.registry.registered_hash(plan.plan_id) is None:
            self.registry.register(plan)
        # Guard before and after: the plan the run was judged against must be the
        # plan that was registered.
        self.registry.verify_unchanged(plan)

        result = adapter.execute(
            contract=contract,
            proposal=proposal,
            plan=plan,
            workdir=workdir,
            experiment_id=experiment_id,
        )

        self.registry.verify_unchanged(plan)
        return result

    def amend_plan(self, *_: object, **__: object):
        """Always refuses. Present so the prohibition is discoverable in code."""
        raise PermissionError(
            "the Executor may not amend a test plan. Changing acceptance criteria to "
            "make a run pass is precisely the failure mode this separation prevents."
        )
