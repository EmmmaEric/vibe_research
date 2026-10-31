"""Layered TOML config and the append-only event log (spec sections 14, 12)."""

from __future__ import annotations

import argparse

import pytest

from research_harness.core.schemas import EventType
from research_harness.infrastructure.config import (
    ConfigError,
    apply_defaults,
    deep_merge,
    effective_config,
    load_layered,
    redact,
    to_cli_defaults,
    validate,
)
from research_harness.memory.events import EventLog, idempotency_key

REPO_GLOBAL = "config/global_config"


def write(tmp_path, name: str, text: str):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Shipped config
# --------------------------------------------------------------------------


def test_bundled_global_config_is_valid_and_complete():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    merged = load_layered(global_dir=root / REPO_GLOBAL)
    assert merged["agent"]["provider"] == "simulated"
    assert merged["strategy"]["mode"] == "adaptive"
    assert merged["target"]["maturity_level"].startswith("R")
    assert merged["budget"]["max_iterations"] >= 1
    # The optional Teacher-Student mode ships disabled, as the spec requires.
    assert merged["agent"]["teacher"]["enabled"] is False


def test_bundled_task_overlays_override_the_globals():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    base = load_layered(global_dir=root / REPO_GLOBAL)
    sci = load_layered(
        global_dir=root / REPO_GLOBAL,
        task_file=root / "config/tasks/demo_scientific.toml",
    )
    prod = load_layered(
        global_dir=root / REPO_GLOBAL,
        task_file=root / "config/tasks/demo_product.toml",
    )
    assert sci["experiment"]["adapter"] == "pytest"
    assert prod["experiment"]["adapter"] == "manual"
    assert prod["strategy"]["mode"] == "tournament"
    # Untouched sections survive the overlay (deep merge, not replacement).
    assert sci["agent"]["provider"] == base["agent"]["provider"]
    assert sci["evaluation"] == base["evaluation"]


# --------------------------------------------------------------------------
# Merge semantics
# --------------------------------------------------------------------------


def test_tables_merge_but_arrays_replace():
    base = {"a": {"x": 1, "y": 2}, "list": [1, 2, 3]}
    over = {"a": {"y": 20, "z": 30}, "list": [9]}
    merged = deep_merge(base, over)
    assert merged["a"] == {"x": 1, "y": 20, "z": 30}
    assert merged["list"] == [9], "arrays replace; silent appending is unpredictable"


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


# --------------------------------------------------------------------------
# Validation: unknown keys are errors
# --------------------------------------------------------------------------


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown section"):
        validate({"nonsense": {"a": 1}}, source="t")


def test_unknown_key_is_rejected_not_ignored():
    with pytest.raises(ConfigError, match="unknown key"):
        validate({"budget": {"max_iterationz": 3}}, source="t")


def test_section_must_be_a_table():
    with pytest.raises(ConfigError, match="must be a table"):
        validate({"budget": 5}, source="t")


def test_subtables_are_allowed_through():
    """[agent.teacher] must not trip the flat-key check."""
    validate({"agent": {"provider": "simulated", "teacher": {"enabled": False}}},
             source="t")


def test_malformed_toml_is_reported(tmp_path):
    write(tmp_path, "g/bad.toml", "this is not = = toml")
    with pytest.raises(ConfigError, match="could not parse"):
        load_layered(global_dir=tmp_path / "g")


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_layered(global_dir=tmp_path / "nope")


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------


def test_explicit_flag_beats_config():
    args = argparse.Namespace(provider="anthropic", mode="adaptive", seed=0)
    apply_defaults(
        args,
        {"provider": "simulated", "mode": "goal_only", "seed": 99},
        explicit={"provider"},  # the user typed --provider
    )
    assert args.provider == "anthropic"
    assert args.mode == "goal_only"
    assert args.seed == 99


def test_wall_time_minutes_become_seconds():
    defaults = to_cli_defaults({"budget": {"max_wall_time_minutes": 2}})
    assert defaults["max_seconds"] == 120.0


def test_cli_projection_only_emits_known_destinations():
    defaults = to_cli_defaults(
        {"agent": {"provider": "simulated", "timeout_s": 10}, "strategy": {"mode": "tournament"}}
    )
    assert defaults["provider"] == "simulated"
    assert defaults["mode"] == "tournament"
    assert "timeout_s" not in defaults


# --------------------------------------------------------------------------
# Redaction  (spec section 14: keys never reach logs or reports)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["api_key", "ANTHROPIC_API_KEY", "token", "secret", "password",
            "credential", "auth_header"]
)
def test_secret_shaped_keys_are_redacted(key):
    assert redact({key: "sk-ant-real-value"})[key] == "***REDACTED***"


def test_env_variable_names_are_not_secrets():
    """`api_key_env` holds the NAME of a variable, which is safe to show."""
    assert redact({"api_key_env": "ANTHROPIC_API_KEY"})["api_key_env"] == (
        "ANTHROPIC_API_KEY"
    )


def test_redaction_recurses_into_tables_and_lists():
    out = redact({"outer": {"api_key": "x", "items": [{"token": "y"}]}})
    assert out["outer"]["api_key"] == "***REDACTED***"
    assert out["outer"]["items"][0]["token"] == "***REDACTED***"


def test_effective_config_snapshot_carries_no_secret_values(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-appear")
    snapshot = effective_config(
        {"backends": {"api_key": "sk-ant-inline", "base_url": "https://x"}},
        cli_overrides={"model": "claude-opus-5-5", "api_key": "sk-ant-flag"},
    )
    blob = str(snapshot)
    assert "sk-ant" not in blob
    assert snapshot["config"]["backends"]["base_url"] == "https://x"
    # Presence is shown by name only, so you can tell a key was available.
    # Asserted as a subset: the developer's real environment may hold other
    # credential variables, and the test must not depend on their absence.
    assert snapshot["credentials_present"].get("ANTHROPIC_API_KEY") is True
    assert all(v is True for v in snapshot["credentials_present"].values())


# --------------------------------------------------------------------------
# Event log
# --------------------------------------------------------------------------


def test_events_are_appended_in_order_with_dense_sequence():
    log = EventLog("r1")
    log.append(EventType.GOAL_DEFINED, "goal")
    log.append(EventType.ROUTE_DECIDED, "route")
    log.append(EventType.RUN_STOPPED, "done")
    assert [e.seq for e in log.events] == [1, 2, 3]
    assert [e.event for e in log.events] == [
        EventType.GOAL_DEFINED, EventType.ROUTE_DECIDED, EventType.RUN_STOPPED
    ]
    assert log.summary_counts()["ROUTE_DECIDED"] == 1
    assert log.last(EventType.RUN_STOPPED).summary == "done"


def test_log_mirrors_to_jsonl_as_it_goes(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog("r1", path)
    log.append(EventType.GOAL_DEFINED, "goal")
    log.append(EventType.ROUTE_DECIDED, "route")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"GOAL_DEFINED"' in lines[0]


# --------------------------------------------------------------------------
# Idempotency  (spec section 12: no replayed side effects)
# --------------------------------------------------------------------------


def test_key_is_derived_from_identity_not_from_time():
    """A key built from a timestamp would never match on replay."""
    a = idempotency_key("experiment", "run1", "PA", "hash123", 1)
    b = idempotency_key("experiment", "run1", "PA", "hash123", 1)
    assert a == b
    assert a != idempotency_key("experiment", "run1", "PA", "hash123", 2)
    assert a != idempotency_key("experiment", "run1", "PB", "hash123", 1)
    # Empty and None parts are skipped rather than producing a different key.
    assert idempotency_key("a", None, "", "b") == idempotency_key("a", "b")


def test_guard_reports_a_skip_once_an_action_has_completed():
    log = EventLog("r1")
    key = idempotency_key("experiment", "r1", "PA", "h", 1)

    assert log.guard(key, action="experiment 1") is False, "first attempt must run"
    log.append(EventType.EXPERIMENT_RUN, "ran it", key=key)

    assert log.already_done(key) is True
    assert log.guard(key, action="experiment 1") is True, "second attempt must skip"
    skip = log.events[-1]
    assert "Skipped" in skip.summary
    assert skip.payload["key"] == key


def test_restore_rebuilds_the_key_set_so_resume_stays_safe():
    log = EventLog("r1")
    key = idempotency_key("experiment", "r1", "PA", "h", 1)
    log.append(EventType.EXPERIMENT_RUN, "ran", key=key)

    restored = EventLog.restore("r1", log.events)
    assert restored.already_done(key), (
        "without the restored key set, a resumed run would repeat the side effect"
    )
    assert [e.seq for e in restored.events] == [1]


def test_load_from_disk_round_trips_keys(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog("r1", path)
    key = idempotency_key("experiment", "r1", "PA", "h", 1)
    log.append(EventType.EXPERIMENT_RUN, "ran", key=key)

    loaded = EventLog.load("r1", path)
    assert loaded.already_done(key)
    assert loaded.events[0].event is EventType.EXPERIMENT_RUN


def test_an_unkeyed_event_never_blocks_anything():
    log = EventLog("r1")
    log.append(EventType.ROUTE_DECIDED, "no side effect")
    assert log.guard("some-key", action="x") is False
