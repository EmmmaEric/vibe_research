"""Hard-constraint gating.

Hard constraints are *non-compensatory*: a FAIL cannot be bought off by a high
score anywhere else. This module turns per-constraint assessments into an
overall status with a fixed precedence:

    any FAIL      -> FAIL      (ineligible, cannot be selected)
    else any UNKNOWN -> UNKNOWN (retained, but must be verified first)
    else          -> PASS

The precedence is deterministic code so that no amount of judge enthusiasm can
promote a violating candidate.
"""

from __future__ import annotations

from ..core.schemas import (
    ConstraintAssessment,
    ConstraintStatus,
    HardConstraint,
    ResearchContract,
)


def combine(assessments: list[ConstraintAssessment]) -> ConstraintStatus:
    """Fold per-constraint statuses into one overall status."""
    statuses = {a.status for a in assessments}
    if ConstraintStatus.FAIL in statuses:
        return ConstraintStatus.FAIL
    if ConstraintStatus.UNKNOWN in statuses:
        return ConstraintStatus.UNKNOWN
    if not assessments:
        # No hard constraints declared -> nothing to violate.
        return ConstraintStatus.PASS
    return ConstraintStatus.PASS


def ensure_complete(
    contract: ResearchContract, assessments: list[ConstraintAssessment]
) -> list[ConstraintAssessment]:
    """Every declared hard constraint must be assessed.

    A constraint the judge failed to address becomes UNKNOWN (with an explicit
    assumption), never an implicit PASS.
    """
    seen = {a.constraint_key for a in assessments}
    out = list(assessments)
    for c in contract.hard_constraints:
        if c.key not in seen:
            out.append(
                ConstraintAssessment(
                    constraint_key=c.key,
                    status=ConstraintStatus.UNKNOWN,
                    rationale="Not addressed by the evaluator; treated as unverified.",
                    unverified_assumption=(
                        f"Whether the proposal satisfies hard constraint '{c.key}': "
                        f"{c.description}"
                    ),
                )
            )
    # Drop assessments for constraints that are not in the contract.
    declared = {c.key for c in contract.hard_constraints}
    return [a for a in out if a.constraint_key in declared]


def filter_eligible(
    evaluations: list,
) -> tuple[list, list[str]]:
    """Split evaluations into eligible and constraint-excluded.

    Returns ``(eligible, excluded_proposal_ids)``.
    """
    eligible = [e for e in evaluations if e.constraint_status is not ConstraintStatus.FAIL]
    excluded = [
        e.proposal_id for e in evaluations if e.constraint_status is ConstraintStatus.FAIL
    ]
    return eligible, excluded


def describe(contract: ResearchContract) -> str:
    """Human-readable constraint list for prompts and reports."""
    if not contract.hard_constraints:
        return "(no hard constraints declared)"
    lines = []
    for c in contract.hard_constraints:
        flag = " [ASSUMED DEFAULT]" if c.assumed else ""
        lines.append(f"- {c.key} ({c.category}): {c.description}{flag}")
    return "\n".join(lines)


def constraint_by_key(contract: ResearchContract, key: str) -> HardConstraint | None:
    for c in contract.hard_constraints:
        if c.key == key:
            return c
    return None
