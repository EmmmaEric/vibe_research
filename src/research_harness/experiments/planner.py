"""Test plan custody and experiment adapter selection.

The spec requires that the Executor cannot quietly relax acceptance criteria to
make a run pass. That is enforced structurally rather than by instruction:

* :class:`TestPlanRegistry` records the content hash of every plan it is given.
* :meth:`TestPlanRegistry.amend` refuses any amendment whose author is an
  execution role, and otherwise produces a **new version** carrying a reason and
  an audit record. Plans are never mutated in place.
* :meth:`TestPlanRegistry.verify_unchanged` is called after execution; a changed
  hash is a hard error, not a warning.
"""

from __future__ import annotations

from ..core.schemas import ContractRevision, TestCase, TestPlan, utcnow

#: Roles that run experiments, and therefore have an interest in the result.
#: None of them may alter acceptance criteria.
EXECUTION_ROLES = frozenset({"executor", "implementation_loop", "experiment_runner"})


class TestPlanTampering(RuntimeError):
    """Raised when a plan is altered by a party that must not alter it."""


class TestPlanRegistry:
    """Custodian of versioned, hash-verified test plans."""

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}
        self._plans: dict[str, TestPlan] = {}

    def register(self, plan: TestPlan) -> str:
        h = plan.content_hash()
        self._hashes[plan.plan_id] = h
        self._plans[plan.plan_id] = plan
        return h

    def registered_hash(self, plan_id: str) -> str | None:
        return self._hashes.get(plan_id)

    def verify_unchanged(self, plan: TestPlan) -> None:
        """Raise if the plan's content no longer matches what was registered."""
        known = self._hashes.get(plan.plan_id)
        if known is None:
            raise TestPlanTampering(
                f"plan {plan.plan_id} was never registered; an unregistered plan "
                "cannot be used to judge a result"
            )
        current = plan.content_hash()
        if current != known:
            raise TestPlanTampering(
                f"plan {plan.plan_id} content changed after registration "
                f"({known} -> {current}). Acceptance criteria must not change during "
                "a run; amend the plan explicitly to create a new version."
            )

    def amend(
        self,
        plan: TestPlan,
        *,
        reason: str,
        author: str,
        cases: list[TestCase] | None = None,
        independent_baseline: str | None = None,
    ) -> TestPlan:
        """Create a new plan version. Execution roles are refused outright."""
        if author in EXECUTION_ROLES:
            raise TestPlanTampering(
                f"role '{author}' executes experiments and may not amend the test "
                "plan. Relaxing acceptance criteria to make a run pass is the "
                "failure this check exists to prevent; route the change through the "
                "test designer with a stated reason."
            )
        if not reason.strip():
            raise TestPlanTampering("an amendment requires a stated reason")

        data = plan.model_dump()
        if cases is not None:
            data["cases"] = [c.model_dump() for c in cases]
        if independent_baseline is not None:
            data["independent_baseline"] = independent_baseline
        data["version"] = plan.version + 1
        data["plan_id"] = f"{plan.run_id}-{plan.proposal_id}-v{plan.version + 1}"
        data["created_at"] = utcnow()
        data["amendments"] = [*data["amendments"]] + [
            ContractRevision(
                version=plan.version + 1,
                reason=reason,
                author=author,
                changed_fields=(
                    [f for f, v in (("cases", cases),
                                    ("independent_baseline", independent_baseline))
                     if v is not None]
                ),
            ).model_dump()
        ]
        amended = TestPlan.model_validate(data)
        self.register(amended)
        return amended


def select_adapter_id(
    *, preferred: tuple[str, ...] | list[str], available: dict[str, object]
) -> str:
    """Pick the first preferred adapter that is actually available.

    Falls back to ``"simulated"`` -- which is honest about what it is -- rather
    than to whatever happens to be first in the registry.
    """
    for name in preferred:
        if name in available:
            return name
    return "simulated"
