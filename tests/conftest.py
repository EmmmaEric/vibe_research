"""Shared fixtures. Everything here is offline and deterministic."""

from __future__ import annotations

import pytest

from research_harness.agents.goal_analyst import GoalAnalyst
from research_harness.agents.proposal_designer import ProposalDesigner
from research_harness.agents.researcher import Researcher
from research_harness.core.schemas import (
    Budget,
    ConstraintAssessment,
    ConstraintStatus,
    DimensionScore,
    Evidence,
    MetricDirection,
    PoCSketch,
    Proposal,
    Provenance,
    ResearchContract,
    Stance,
    SuccessMetric,
    MaturityLevel,
)
from research_harness.evaluation.judge_adapter import MockJudgeAdapter
from research_harness.evaluation.rubric import build_rubric
from research_harness.providers.simulated import SimulatedProvider
from research_harness.tools.search import FixtureSearchTool

FIXTURE_DIR = None  # resolved lazily from the installed package


@pytest.fixture
def provider() -> SimulatedProvider:
    return SimulatedProvider(seed=42)


@pytest.fixture
def judge() -> MockJudgeAdapter:
    return MockJudgeAdapter()


@pytest.fixture
def bundled_fixture_dir():
    from research_harness import cli

    return cli.BUNDLED_FIXTURES


@pytest.fixture
def contract(provider) -> ResearchContract:
    """A contract built the way a real run builds it, from a bare idea."""
    return GoalAnalyst(provider).build_contract(
        run_id="test_run",
        idea="Make sparse matrix multiplication faster under a fixed memory budget",
        budget=Budget(max_iterations=2),
        desired_maturity_level=MaturityLevel.R3_PROTOTYPED,
    )


@pytest.fixture
def evidence(provider, contract, bundled_fixture_dir) -> list[Evidence]:
    researcher = Researcher(provider, [FixtureSearchTool(bundled_fixture_dir)])
    items, _ = researcher.gather(contract, limit=8)
    return items


@pytest.fixture
def proposals(provider, contract, evidence) -> list[Proposal]:
    items, _ = ProposalDesigner(provider).design_all(
        contract=contract, evidence=evidence
    )
    return items


@pytest.fixture
def rubric(contract):
    return build_rubric(contract, version="v1")


def make_contract(
    *,
    run_id: str = "t",
    idea: str = "test idea",
    metrics: list[SuccessMetric] | None = None,
    constraints: list | None = None,
    budget: Budget | None = None,
) -> ResearchContract:
    """Minimal hand-built contract for unit tests."""
    return ResearchContract(
        run_id=run_id,
        original_idea=idea,
        problem_statement="statement",
        success_metrics=metrics
        or [
            SuccessMetric(
                key="speed",
                description="speedup",
                unit="x",
                target_threshold=2.0,
                direction=MetricDirection.MAXIMIZE,
            )
        ],
        hard_constraints=constraints or [],
        budget=budget or Budget(),
    )


def make_proposal(
    pid: str,
    *,
    archetype: str = "balanced",
    cost: float = 2.0,
    mechanism: str = "a mechanism",
    steps: list[str] | None = None,
    failure_modes: list[str] | None = None,
    assumptions: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    falsifier: str = "it does not improve on the baseline",
    poc_cost: float = 1.0,
) -> Proposal:
    return Proposal(
        proposal_id=pid,
        run_id="t",
        title=f"candidate {pid}",
        strategy_archetype=archetype,
        core_mechanism=mechanism,
        goal_mapping="targets speed",
        key_steps=steps or ["step one", "step two"],
        failure_modes=failure_modes or [],
        critical_assumptions=assumptions or [],
        evidence_ids=evidence_ids or [],
        declared_cost_units=cost,
        poc=PoCSketch(
            description="minimal trial",
            expected_result="a measurement",
            falsification_condition=falsifier,
            estimated_cost_units=poc_cost,
        ),
    )


def make_evidence(eid: str, *, real: bool = True, stance: Stance = Stance.SUPPORTS,
                  quality: int = 3) -> Evidence:
    return Evidence(
        evidence_id=eid,
        claim=f"claim {eid}",
        source_type="paper" if real else "simulated",
        provenance=Provenance.SOURCE if real else Provenance.SIMULATED,
        stance=stance,
        quality=quality,
        verification_status="source_checked" if real else "unverified",
    )


def scores_for(rubric, value: int, *, unknown: tuple[str, ...] = ()) -> list[DimensionScore]:
    """Flat score across every rubric dimension, with chosen dims UNKNOWN."""
    out = []
    for d in rubric.dimensions:
        if d.key in unknown:
            out.append(
                DimensionScore(
                    dimension_key=d.key, raw=None, status="UNKNOWN", justification="n/a"
                )
            )
        else:
            out.append(
                DimensionScore(dimension_key=d.key, raw=value, justification="flat")
            )
    return out


def passing(keys: list[str]) -> list[ConstraintAssessment]:
    return [
        ConstraintAssessment(constraint_key=k, status=ConstraintStatus.PASS)
        for k in keys
    ]
