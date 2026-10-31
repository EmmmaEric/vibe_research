"""SQLite schema for the research memory.

Stores the research *causal chain* -- goal, evidence, proposal, evaluation, test
plan, experiment, observation, decision, conclusion -- as queryable rows, not as
a chat log. Plain ``sqlite3`` from the standard library; an ORM would add a
dependency without buying anything at this size.

Full artefacts stay in the filesystem artifact store; the database holds the
structured spine plus a JSON payload column per row, so nothing is lost and
cross-run queries stay cheap.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    original_idea     TEXT NOT NULL,
    phase             TEXT NOT NULL,
    iteration         INTEGER NOT NULL DEFAULT 0,
    maturity_level  TEXT,
    stop_reason       TEXT,
    stop_detail       TEXT,
    provider_id       TEXT,
    judge_id          TEXT,
    adapter_id        TEXT,
    is_simulated      INTEGER NOT NULL DEFAULT 1,
    seed              INTEGER,
    python_version    TEXT,
    platform          TEXT,
    tokens_used       INTEGER DEFAULT 0,
    cost_usd          REAL DEFAULT 0.0,
    elapsed_seconds   REAL DEFAULT 0.0,
    created_at        TEXT,
    updated_at        TEXT,
    payload           TEXT
);

CREATE TABLE IF NOT EXISTS contracts (
    run_id      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    goal_type   TEXT,
    domain_hint TEXT,
    created_at  TEXT,
    payload     TEXT NOT NULL,
    PRIMARY KEY (run_id, version)
);

CREATE TABLE IF NOT EXISTS evidence (
    run_id              TEXT NOT NULL,
    evidence_id         TEXT NOT NULL,
    claim               TEXT NOT NULL,
    source_type         TEXT,
    source_uri          TEXT,
    stance              TEXT,
    provenance          TEXT,
    verification_status TEXT,
    quality             INTEGER,
    retrieved_at        TEXT,
    payload             TEXT NOT NULL,
    PRIMARY KEY (run_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS proposals (
    run_id        TEXT NOT NULL,
    proposal_id   TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    title         TEXT,
    archetype     TEXT,
    mechanism     TEXT,
    declared_cost REAL,
    generated_by  TEXT,
    created_at    TEXT,
    payload       TEXT NOT NULL,
    PRIMARY KEY (run_id, proposal_id, version)
);

CREATE TABLE IF NOT EXISTS rubrics (
    run_id     TEXT NOT NULL,
    version    TEXT NOT NULL,
    created_at TEXT,
    payload    TEXT NOT NULL,
    PRIMARY KEY (run_id, version)
);

CREATE TABLE IF NOT EXISTS scores (
    run_id            TEXT NOT NULL,
    proposal_id       TEXT NOT NULL,
    rubric_version    TEXT NOT NULL,
    constraint_status TEXT,
    point_score       REAL,
    score_lower       REAL,
    score_upper       REAL,
    evidence_suff     REAL,
    eligible          INTEGER,
    computed_at       TEXT,
    payload           TEXT NOT NULL,
    PRIMARY KEY (run_id, proposal_id, rubric_version)
);

CREATE TABLE IF NOT EXISTS rankings (
    run_id         TEXT NOT NULL,
    rubric_version TEXT NOT NULL,
    created_at     TEXT,
    payload        TEXT NOT NULL,
    PRIMARY KEY (run_id, rubric_version)
);

CREATE TABLE IF NOT EXISTS test_plans (
    run_id      TEXT NOT NULL,
    plan_id     TEXT NOT NULL,
    proposal_id TEXT,
    version     INTEGER,
    plan_hash   TEXT,
    created_at  TEXT,
    payload     TEXT NOT NULL,
    PRIMARY KEY (run_id, plan_id)
);

CREATE TABLE IF NOT EXISTS experiments (
    run_id        TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    proposal_id   TEXT,
    plan_id       TEXT,
    plan_hash     TEXT,
    adapter_id    TEXT,
    is_simulated  INTEGER,
    executed      INTEGER,
    requires_human INTEGER,
    seed          INTEGER,
    duration_s    REAL,
    error         TEXT,
    started_at    TEXT,
    payload       TEXT NOT NULL,
    PRIMARY KEY (run_id, experiment_id)
);

CREATE TABLE IF NOT EXISTS observations (
    run_id        TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    metric_key    TEXT NOT NULL,
    value         REAL,
    unit          TEXT,
    provenance    TEXT,
    n_repeats     INTEGER,
    stddev        REAL,
    seed          INTEGER,
    PRIMARY KEY (run_id, experiment_id, metric_key)
);

CREATE TABLE IF NOT EXISTS validations (
    run_id           TEXT NOT NULL,
    experiment_id    TEXT NOT NULL,
    proposal_id      TEXT,
    conforms         INTEGER,
    goal_satisfied   INTEGER,
    maturity_level TEXT,
    blocking_reason  TEXT,
    validated_at     TEXT,
    payload          TEXT NOT NULL,
    PRIMARY KEY (run_id, experiment_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    iteration   INTEGER,
    loop        TEXT,
    action      TEXT,
    cause       TEXT,
    continued   INTEGER,
    at          TEXT,
    payload     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);
CREATE INDEX IF NOT EXISTS idx_proposals_run ON proposals(run_id);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_experiments_run ON experiments(run_id);
"""

#: FTS5 is compiled into most CPython builds but not guaranteed; the repository
#: degrades to LIKE-based search when it is unavailable rather than failing.
FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
    run_id UNINDEXED,
    kind   UNINDEXED,
    ref_id UNINDEXED,
    text
);
"""
