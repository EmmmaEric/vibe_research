"""Repository: persistence for the research memory.

Every ``save_*`` is idempotent (``INSERT OR REPLACE``) so re-running a phase
after a resume does not duplicate rows. The database is a queryable index over
the run; the artifact store remains the source of truth for large artefacts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..core.schemas import (
    DecisionRecord,
    Evidence,
    ExperimentResult,
    Proposal,
    ProposalEvaluation,
    Ranking,
    ResearchContract,
    Rubric,
    TestPlan,
    ValidationResult,
)
from .models import DDL, FTS_DDL, SCHEMA_VERSION


def _b(value: bool | None) -> int | None:
    """SQLite has no boolean; preserve the None/True/False distinction."""
    return None if value is None else int(value)


class Repository:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.fts_available = False
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(DDL)
        try:
            self.conn.executescript(FTS_DDL)
            self.fts_available = True
        except sqlite3.OperationalError:
            # No FTS5 in this build; retrieval falls back to LIKE.
            self.fts_available = False
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Repository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------

    def _index(self, run_id: str, kind: str, ref_id: str, text: str) -> None:
        if not self.fts_available or not text.strip():
            return
        self.conn.execute(
            "DELETE FROM search_index WHERE run_id=? AND kind=? AND ref_id=?",
            (run_id, kind, ref_id),
        )
        self.conn.execute(
            "INSERT INTO search_index(run_id, kind, ref_id, text) VALUES(?,?,?,?)",
            (run_id, kind, ref_id, text),
        )

    # -- writes ----------------------------------------------------------

    def save_run(self, state: Any) -> None:
        bs = state.budget_state
        contract = state.contract
        self.conn.execute(
            """INSERT OR REPLACE INTO runs(
                run_id, original_idea, phase, iteration, maturity_level,
                stop_reason, stop_detail, provider_id, judge_id, adapter_id,
                is_simulated, seed, python_version, platform, tokens_used,
                cost_usd, elapsed_seconds, created_at, updated_at, payload
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                state.run_id,
                contract.original_idea if contract else "",
                state.phase.value,
                state.iteration,
                state.final_maturity_level.value if state.final_maturity_level else None,
                state.stop_decision.reason if state.stop_decision else None,
                state.stop_decision.detail if state.stop_decision else None,
                state.provenance.provider_id,
                state.provenance.judge_id,
                state.provenance.experiment_adapter_id,
                int(state.provenance.is_simulated_provider),
                state.provenance.seed,
                state.provenance.python_version,
                state.provenance.platform,
                bs.tokens_used,
                bs.cost_usd,
                bs.elapsed_seconds,
                state.created_at,
                state.updated_at,
                state.to_json(),
            ),
        )
        self.conn.commit()

    def save_contract(self, contract: ResearchContract) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO contracts(
                run_id, version, goal_type, domain_hint, created_at, payload
            ) VALUES(?,?,?,?,?,?)""",
            (
                contract.run_id,
                contract.version,
                contract.goal_type,
                contract.domain_hint,
                contract.created_at,
                contract.model_dump_json(),
            ),
        )
        self._index(
            contract.run_id,
            "contract",
            f"v{contract.version}",
            f"{contract.original_idea}\n{contract.problem_statement}",
        )
        self.conn.commit()

    def save_evidence(self, run_id: str, evidence: list[Evidence]) -> None:
        for e in evidence:
            self.conn.execute(
                """INSERT OR REPLACE INTO evidence(
                    run_id, evidence_id, claim, source_type, source_uri, stance,
                    provenance, verification_status, quality, retrieved_at, payload
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    e.evidence_id,
                    e.claim,
                    e.source_type,
                    e.source_uri,
                    e.stance.value,
                    e.provenance.value,
                    e.verification_status,
                    e.quality,
                    e.retrieved_at,
                    e.model_dump_json(),
                ),
            )
            self._index(run_id, "evidence", e.evidence_id, e.claim)
        self.conn.commit()

    def save_proposals(self, proposals: list[Proposal]) -> None:
        for p in proposals:
            self.conn.execute(
                """INSERT OR REPLACE INTO proposals(
                    run_id, proposal_id, version, title, archetype, mechanism,
                    declared_cost, generated_by, created_at, payload
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    p.run_id,
                    p.proposal_id,
                    p.version,
                    p.title,
                    p.strategy_archetype,
                    p.core_mechanism,
                    p.declared_cost_units,
                    p.generated_by,
                    p.created_at,
                    p.model_dump_json(),
                ),
            )
            self._index(
                p.run_id,
                "proposal",
                p.proposal_id,
                f"{p.title}\n{p.core_mechanism}\n{p.goal_mapping}\n"
                + "\n".join(p.key_steps + p.failure_modes),
            )
        self.conn.commit()

    def save_rubric(self, rubric: Rubric) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO rubrics(run_id, version, created_at, payload) "
            "VALUES(?,?,?,?)",
            (rubric.run_id, rubric.version, rubric.created_at, rubric.model_dump_json()),
        )
        self.conn.commit()

    def save_evaluations(self, evaluations: list[ProposalEvaluation]) -> None:
        for ev in evaluations:
            self.conn.execute(
                """INSERT OR REPLACE INTO scores(
                    run_id, proposal_id, rubric_version, constraint_status,
                    point_score, score_lower, score_upper, evidence_suff, eligible,
                    computed_at, payload
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ev.run_id,
                    ev.proposal_id,
                    ev.rubric_version,
                    ev.constraint_status.value,
                    ev.point_score,
                    ev.score_lower,
                    ev.score_upper,
                    ev.evidence_sufficiency,
                    int(ev.eligible),
                    ev.computed_at,
                    ev.model_dump_json(),
                ),
            )
        self.conn.commit()

    def save_ranking(self, ranking: Ranking) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO rankings(run_id, rubric_version, created_at, payload) "
            "VALUES(?,?,?,?)",
            (
                ranking.run_id,
                ranking.rubric_version,
                ranking.created_at,
                ranking.model_dump_json(),
            ),
        )
        self.conn.commit()

    def save_test_plan(self, plan: TestPlan) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO test_plans(
                run_id, plan_id, proposal_id, version, plan_hash, created_at, payload
            ) VALUES(?,?,?,?,?,?,?)""",
            (
                plan.run_id,
                plan.plan_id,
                plan.proposal_id,
                plan.version,
                plan.content_hash(),
                plan.created_at,
                plan.model_dump_json(),
            ),
        )
        self.conn.commit()

    def save_experiment(self, exp: ExperimentResult) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO experiments(
                run_id, experiment_id, proposal_id, plan_id, plan_hash, adapter_id,
                is_simulated, executed, requires_human, seed, duration_s, error,
                started_at, payload
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                exp.run_id,
                exp.experiment_id,
                exp.proposal_id,
                exp.plan_id,
                exp.plan_hash,
                exp.adapter_id,
                int(exp.is_simulated),
                int(exp.executed),
                int(exp.requires_human_input),
                exp.seed,
                exp.duration_seconds,
                exp.error,
                exp.started_at,
                exp.model_dump_json(),
            ),
        )
        for obs in exp.observations:
            self.conn.execute(
                """INSERT OR REPLACE INTO observations(
                    run_id, experiment_id, metric_key, value, unit, provenance,
                    n_repeats, stddev, seed
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    exp.run_id,
                    exp.experiment_id,
                    obs.metric_key,
                    obs.value,
                    obs.unit,
                    obs.provenance.value,
                    obs.n_repeats,
                    obs.stddev,
                    obs.seed,
                ),
            )
        self.conn.commit()

    def save_validation(self, v: ValidationResult) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO validations(
                run_id, experiment_id, proposal_id, conforms, goal_satisfied,
                maturity_level, blocking_reason, validated_at, payload
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                v.run_id,
                v.experiment_id,
                v.proposal_id,
                _b(v.implementation_conforms),
                _b(v.goal_satisfied),
                v.maturity_level.value,
                v.blocking_reason,
                v.validated_at,
                v.model_dump_json(),
            ),
        )
        self.conn.commit()

    def save_decisions(self, decisions: list[DecisionRecord]) -> None:
        for d in decisions:
            self.conn.execute(
                """INSERT OR REPLACE INTO decisions(
                    decision_id, run_id, iteration, loop, action, cause, continued,
                    at, payload
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    d.decision_id,
                    d.run_id,
                    d.iteration,
                    d.loop,
                    d.action,
                    d.divergence_cause,
                    int(d.should_continue),
                    d.at,
                    d.model_dump_json(),
                ),
            )
        self.conn.commit()

    # -- reads -----------------------------------------------------------

    def list_runs(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT run_id, original_idea, phase, maturity_level, stop_reason, "
                "is_simulated, updated_at FROM runs ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        )

    def get_run_payload(self, run_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT payload FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return row["payload"] if row else None

    def causal_chain(self, run_id: str) -> dict[str, list[dict]]:
        """The run's causal chain, in the spec's order, for inspection and reuse."""
        out: dict[str, list[dict]] = {}
        for label, sql in (
            ("contract", "SELECT version, goal_type, payload FROM contracts WHERE run_id=? ORDER BY version"),
            ("evidence", "SELECT evidence_id, claim, stance, provenance, verification_status FROM evidence WHERE run_id=? ORDER BY evidence_id"),
            ("proposals", "SELECT proposal_id, title, archetype, declared_cost FROM proposals WHERE run_id=? ORDER BY proposal_id"),
            ("scores", "SELECT proposal_id, rubric_version, point_score, score_lower, score_upper, constraint_status, eligible FROM scores WHERE run_id=? ORDER BY rubric_version, proposal_id"),
            ("test_plans", "SELECT plan_id, proposal_id, version, plan_hash FROM test_plans WHERE run_id=? ORDER BY plan_id"),
            ("experiments", "SELECT experiment_id, adapter_id, is_simulated, executed, error FROM experiments WHERE run_id=? ORDER BY experiment_id"),
            ("observations", "SELECT experiment_id, metric_key, value, unit, provenance, n_repeats FROM observations WHERE run_id=? ORDER BY experiment_id, metric_key"),
            ("validations", "SELECT experiment_id, proposal_id, conforms, goal_satisfied, maturity_level FROM validations WHERE run_id=? ORDER BY experiment_id"),
            ("decisions", "SELECT decision_id, iteration, loop, action, cause, continued FROM decisions WHERE run_id=? ORDER BY decision_id"),
        ):
            out[label] = [dict(r) for r in self.conn.execute(sql, (run_id,))]
        return out
