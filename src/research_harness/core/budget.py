"""Budget tracking and the artifact store layout.

The budget tracker is the thing that makes "the system can stop and say why"
true. It samples wall clock and provider usage, writes them into
:class:`~research_harness.core.state.BudgetState`, and
:func:`~research_harness.evaluation.decision_engine.should_stop` reads them.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

from .state import BudgetState, ResearchState


class BudgetTracker:
    """Samples elapsed time and provider usage into the run state."""

    def __init__(self, provider, *, started_monotonic: float | None = None) -> None:
        self.provider = provider
        self.start = started_monotonic if started_monotonic is not None else time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def sync(self, state: ResearchState) -> BudgetState:
        usage = getattr(self.provider, "usage", None)
        bs = state.budget_state
        bs.elapsed_seconds = round(self.elapsed(), 4)
        if usage is not None:
            bs.tokens_used = usage.total_tokens
            bs.cost_usd = round(usage.cost_usd, 6)
            bs.llm_calls = usage.calls
        state.touch()
        return bs


def environment_fingerprint() -> dict[str, str]:
    """Environment details recorded with every run, for reproducibility."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


class ArtifactStore:
    """Filesystem layout for one run, matching the spec's output contract.

        research_run_<id>/
          research_contract.json
          evidence.jsonl
          proposals/proposal_*.json
          evaluation/{rubric,scores,pairwise}.json, ranking_report.md
          experiments/test_plan.json, run_001/, run_002/
          decisions.jsonl
          state.json
          final_report.md
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def ensure(self) -> "ArtifactStore":
        for sub in ("", "proposals", "evaluation", "experiments"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

    # -- paths -----------------------------------------------------------

    @property
    def contract(self) -> Path:
        return self.root / "research_contract.json"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence.jsonl"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions.jsonl"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def final_report(self) -> Path:
        return self.root / "final_report.md"

    @property
    def research_report(self) -> Path:
        return self.root / "research_report.md"

    @property
    def ranking_report(self) -> Path:
        return self.root / "evaluation" / "ranking_report.md"

    def proposal(self, proposal_id: str) -> Path:
        return self.root / "proposals" / f"proposal_{proposal_id}.json"

    def evaluation_file(self, name: str) -> Path:
        return self.root / "evaluation" / name

    def test_plan(self, proposal_id: str, version: int) -> Path:
        return self.root / "experiments" / f"test_plan_{proposal_id}_v{version}.json"

    def experiment_dir(self, index: int) -> Path:
        p = self.root / "experiments" / f"run_{index:03d}"
        p.mkdir(parents=True, exist_ok=True)
        return p
