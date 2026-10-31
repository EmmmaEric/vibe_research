"""End-to-end CLI tests for the v2 surface: modes, offline, compare, events.

Shares the helpers and fixtures in :mod:`tests.e2e.test_cli`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .test_cli import (  # noqa: F401  (ws is a fixture used by injection)
    ALGORITHM_IDEA,
    load_state,
    run_cli,
    run_dir,
    ws,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Modes  (spec 16)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["goal_only", "adaptive", "tournament"])
def test_every_mode_runs_end_to_end_offline(ws, capsys, mode):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--mode", mode, "--offline", "--max-iterations", "10",
    ) == 0
    out = capsys.readouterr().out
    assert f"Mode:       {mode}" in out
    assert "Route:" in out, "the routing trace must be visible"

    state = load_state(ws)
    assert state["mode"] == mode
    assert state["routing_decisions"], "a run with no routing decision has no audit"
    assert state["outcome"]


def test_goal_only_is_a_real_baseline_not_a_flag(ws, capsys):
    """One candidate, no tournament, no subagent -- structurally."""
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--mode", "goal_only", "--offline", "--max-iterations", "10",
    ) == 0
    capsys.readouterr()
    state = load_state(ws)
    actions = [d["action"] for d in state["routing_decisions"]]
    assert "RUN_PROPOSAL_TOURNAMENT" not in actions
    assert "SPAWN_SUBAGENT" not in actions
    assert len(state["proposals"]) == 1
    assert state["tournament_ran"] is False


# --------------------------------------------------------------------------
# Offline guarantees
# --------------------------------------------------------------------------


def test_offline_refuses_a_network_provider(ws):
    with pytest.raises(SystemExit, match="offline forbids a network provider"):
        run_cli(
            "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--offline", "--provider", "anthropic",
        )


def test_offline_downgrades_web_search_and_records_it(ws, capsys):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--offline", "--search", "web", "--max-iterations", "6",
    ) == 0
    capsys.readouterr()
    assert any("offline" in n for n in load_state(ws)["notes"])


# --------------------------------------------------------------------------
# Idea input
# --------------------------------------------------------------------------


def test_idea_can_come_from_a_file(ws, tmp_path, capsys):
    idea_file = tmp_path / "idea.md"
    idea_file.write_text(
        "# Goal\n\nReduce p95 latency of a read-only API using a cache.\n",
        encoding="utf-8",
    )
    assert run_cli(
        "run", "--idea-file", str(idea_file), "--workspace", str(ws),
        "--offline", "--max-iterations", "8",
    ) == 0
    capsys.readouterr()
    assert "Reduce p95 latency" in load_state(ws)["contract"]["original_idea"]


def test_missing_idea_file_fails_cleanly(ws, tmp_path):
    with pytest.raises(SystemExit, match="idea file not found"):
        run_cli("run", "--idea-file", str(tmp_path / "nope.md"), "--workspace", str(ws))


def test_empty_idea_file_is_rejected(ws, tmp_path):
    path = tmp_path / "blank.md"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(SystemExit, match="idea file is empty"):
        run_cli("run", "--idea-file", str(path), "--workspace", str(ws))


# --------------------------------------------------------------------------
# Config snapshot and audit log
# --------------------------------------------------------------------------


def test_run_writes_a_redacted_effective_config(ws, capsys):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--offline", "--max-iterations", "6",
    ) == 0
    capsys.readouterr()
    snapshot = json.loads(
        (run_dir(ws) / "redacted_effective_config.json").read_text("utf-8")
    )
    assert "config" in snapshot and "cli_overrides" in snapshot
    assert "sk-ant" not in json.dumps(snapshot), "a secret reached a run artifact"


def test_events_command_shows_the_audit_log(ws, capsys):
    run_cli("run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--offline", "--max-iterations", "8")
    run_id = load_state(ws)["run_id"]
    capsys.readouterr()

    assert run_cli("events", run_id, "--workspace", str(ws)) == 0
    out = capsys.readouterr().out
    for expected in ("GOAL_DEFINED", "ROUTE_DECIDED", "RUN_STOPPED"):
        assert expected in out

    assert run_cli("events", run_id, "--workspace", str(ws), "--json") == 0
    events = json.loads(capsys.readouterr().out)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


# --------------------------------------------------------------------------
# Pareto and summary views
# --------------------------------------------------------------------------


def test_evaluate_includes_the_pareto_view(ws, capsys):
    run_cli("run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--mode", "tournament", "--offline", "--max-iterations", "10")
    run_id = load_state(ws)["run_id"]
    capsys.readouterr()
    assert run_cli("evaluate", run_id, "--workspace", str(ws)) == 0
    out = capsys.readouterr().out
    assert "Pareto analysis" in out
    assert "Non-dominated:" in out


def test_report_summary_format(ws, capsys):
    run_cli("run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--offline", "--max-iterations", "8")
    run_id = load_state(ws)["run_id"]
    capsys.readouterr()
    assert run_cli(
        "report", run_id, "--workspace", str(ws), "--format", "summary"
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["run_id"] == run_id
    assert summary["routing_actions"]
    assert summary["maturity_level"]


# --------------------------------------------------------------------------
# Ablation  (spec 16)
# --------------------------------------------------------------------------


def test_compare_runs_every_arm_and_refuses_to_declare_a_winner(ws, capsys):
    assert run_cli(
        "compare", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--offline", "--max-iterations", "8", "--run-prefix", "cmp",
    ) == 0
    out = capsys.readouterr().out

    for mode in ("goal_only", "adaptive", "tournament"):
        assert mode in out
    assert "Route taken per arm" in out
    assert "Observations:" in out
    assert "Candidates generated per arm" in out
    # The caveat is the point: one instance establishes nothing general.
    assert "cannot establish that a richer mode is better" in out

    report = json.loads((ws / "ablation_cmp.json").read_text("utf-8"))
    assert len(report["results"]) == 3
    assert {r["mode"] for r in report["results"]} == {
        "goal_only", "adaptive", "tournament"
    }
    for r in report["results"]:
        assert (ws / f"research_run_{r['run_id']}" / "state.json").is_file()


def test_compare_can_select_specific_arms(ws, capsys):
    assert run_cli(
        "compare", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--modes", "goal_only", "adaptive",
        "--offline", "--max-iterations", "6", "--run-prefix", "two", "--json",
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert {r["mode"] for r in report["results"]} == {"goal_only", "adaptive"}


def test_baseline_arm_generates_exactly_one_candidate(ws, capsys):
    """If the mode gate leaked, the whole ablation would be meaningless."""
    assert run_cli(
        "compare", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--offline", "--max-iterations", "8", "--run-prefix", "gate", "--json",
    ) == 0
    report = json.loads(capsys.readouterr().out)
    by_mode = {r["mode"]: r for r in report["results"]}
    assert by_mode["goal_only"]["proposal_count"] == 1
    assert by_mode["tournament"]["proposal_count"] >= 2
    assert not any("mode gate may be leaking" in f for f in report["findings"])


def test_arms_differ_in_the_route_they_take(ws, capsys):
    assert run_cli(
        "compare", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--offline", "--max-iterations", "10", "--run-prefix", "route", "--json",
    ) == 0
    report = json.loads(capsys.readouterr().out)
    routes = {r["mode"]: tuple(r["routing_actions"]) for r in report["results"]}
    assert len(set(routes.values())) > 1, (
        "every mode took an identical route; the modes are not actually different"
    )


# --------------------------------------------------------------------------
# Layered config
# --------------------------------------------------------------------------


def test_config_and_task_layers_are_applied(ws, capsys):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--config", str(REPO_ROOT / "config/global_config"),
        "--task", str(REPO_ROOT / "config/tasks/demo_product.toml"),
        "--offline",
    ) == 0
    capsys.readouterr()
    state = load_state(ws)
    # demo_product.toml selects tournament mode and the manual adapter.
    assert state["mode"] == "tournament"
    assert state["provenance"]["experiment_adapter_id"] == "manual"


def test_an_explicit_flag_still_beats_the_task_config(ws, capsys):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--config", str(REPO_ROOT / "config/global_config"),
        "--task", str(REPO_ROOT / "config/tasks/demo_product.toml"),
        "--mode", "goal_only", "--offline",
    ) == 0
    capsys.readouterr()
    assert load_state(ws)["mode"] == "goal_only"


def test_an_unknown_config_key_fails_loudly(ws, tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("[budget]\nmax_iterationz = 3\n", encoding="utf-8")
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--task", str(bad), "--offline",
    ) == 2
    assert "unknown key" in capsys.readouterr().err
