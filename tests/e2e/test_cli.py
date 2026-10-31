"""End-to-end CLI tests.

Driven through ``cli.main`` with real argv, a real workspace on disk and real
SQLite persistence -- the same path a user takes. One test runs the bundled
Example A through the **real** pytest adapter, so the harness's claim that it
can reach L5 on measured data is verified by actually measuring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_harness import cli

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
ALGORITHM_IDEA = (
    "Explore a new algorithm to make sparse matrix-vector multiplication faster "
    "on commodity CPUs while keeping memory below the dense representation"
)
PRODUCT_IDEA = (
    "Design an offline document collaboration product for small businesses and "
    "assess its commercial viability"
)


def run_cli(*argv: str) -> int:
    return cli.main(list(argv))


def run_dir(ws: Path) -> Path:
    dirs = sorted(ws.glob("research_run_*"))
    assert dirs, f"no run directory under {ws}"
    return dirs[-1]


def load_state(ws: Path) -> dict:
    return json.loads((run_dir(ws) / "state.json").read_text(encoding="utf-8"))


@pytest.fixture
def ws(tmp_path) -> Path:
    return tmp_path / "workspace"


# --------------------------------------------------------------------------
# init / run / status / report
# --------------------------------------------------------------------------


def test_init_creates_a_workspace(ws, capsys):
    assert run_cli("init", "--workspace", str(ws)) == 0
    assert (ws / "research_memory.sqlite3").is_file()
    out = capsys.readouterr().out
    assert "Workspace" in out and "Domain profiles" in out


def test_run_produces_artifacts_and_a_report(ws, capsys):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--max-iterations", "10", "--offline",
    ) == 0
    out = capsys.readouterr().out
    assert "Phase:" in out
    # The ranking header now names its own sort key, because the displayed
    # point score is NOT what the order is based on (ADR-014).
    assert "Ranking (ordered by LOWER bound" in out

    d = run_dir(ws)
    assert (d / "final_report.md").is_file()
    report = (d / "final_report.md").read_text(encoding="utf-8")
    assert ALGORITHM_IDEA in report
    assert "## 10. Conclusion level" in report


def test_status_proposals_and_evaluate_read_back(ws, capsys):
    run_cli("run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--mode", "tournament", "--max-iterations", "10", "--offline")
    run_id = load_state(ws)["run_id"]
    capsys.readouterr()

    assert run_cli("status", run_id, "--workspace", str(ws)) == 0
    assert "Phase:" in capsys.readouterr().out

    assert run_cli("proposals", run_id, "--workspace", str(ws)) == 0
    proposals_out = capsys.readouterr().out
    assert "score:" in proposals_out and "constraints:" in proposals_out
    # The constraint-violating candidate is shown as ineligible, not hidden.
    assert "eligible: NO" in proposals_out

    assert run_cli("evaluate", run_id, "--workspace", str(ws)) == 0
    evaluate_out = capsys.readouterr().out
    assert "Rubric" in evaluate_out and "Weight" in evaluate_out

    assert run_cli("report", run_id, "--workspace", str(ws)) == 0
    assert "## 1. Original idea" in capsys.readouterr().out

    assert run_cli("report", run_id, "--workspace", str(ws), "--format", "json") == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == run_id


def test_proposals_json_is_machine_readable(ws, capsys):
    run_cli("run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--mode", "tournament", "--max-iterations", "10", "--offline")
    run_id = load_state(ws)["run_id"]
    capsys.readouterr()
    assert run_cli("proposals", run_id, "--workspace", str(ws), "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) >= 2
    assert {"proposal_id", "core_mechanism", "declared_cost_units"} <= set(data[0])


def test_unknown_run_id_fails_cleanly(ws):
    (ws).mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit, match="no run found"):
        run_cli("status", "run_doesnotexist", "--workspace", str(ws))


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------


def test_weights_view_then_change_reranks(ws, capsys):
    run_cli("run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--mode", "tournament", "--max-iterations", "10", "--offline")
    run_id = load_state(ws)["run_id"]
    capsys.readouterr()

    assert run_cli("weights", run_id, "--workspace", str(ws)) == 0
    view = capsys.readouterr().out
    assert "normalised" in view and "cost_time" in view

    assert run_cli(
        "weights", run_id, "--workspace", str(ws),
        "--set", "cost_time=30.0", "--reason", "cost dominates now",
    ) == 0
    out = capsys.readouterr().out
    assert "Ranking before:" in out and "Ranking after:" in out

    state = load_state(ws)
    assert state["rubric"]["version"] != "v1"
    assert state["rubric_history"], "the prior rubric version must be retained"
    assert any("Weights changed" in n for n in state["notes"])


def test_weights_rejects_a_bad_dimension_and_a_bad_number(ws):
    run_cli("run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--mode", "tournament", "--max-iterations", "10", "--offline")
    run_id = load_state(ws)["run_id"]
    with pytest.raises(SystemExit):
        run_cli("weights", run_id, "--workspace", str(ws), "--set", "cost_time=abc")
    with pytest.raises(SystemExit):
        run_cli("weights", run_id, "--workspace", str(ws), "--set", "noSuchDim=1.0")


# --------------------------------------------------------------------------
# approval gate and resume
# --------------------------------------------------------------------------


def test_approval_gate_pauses_then_resumes_via_cli(ws, capsys):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--max-iterations", "10", "--offline", "--require-approval", "experiment",
    ) == 0
    out = capsys.readouterr().out
    assert "awaiting approval" in out.lower()

    state = load_state(ws)
    run_id = state["run_id"]
    assert state["phase"] == "AWAITING_APPROVAL"
    assert not state["experiments"]

    # resume alone must not bypass the gate
    assert run_cli("resume", run_id, "--workspace", str(ws)) == 1
    assert "awaiting approval" in capsys.readouterr().out.lower()

    assert run_cli(
        "approve", run_id, "--workspace", str(ws), "--stage", "experiment"
    ) == 0
    capsys.readouterr()

    after = load_state(ws)
    assert after["phase"] in ("DONE", "STOPPED")
    assert after["experiments"]
    assert "experiment" in after["approved_stages"]


# --------------------------------------------------------------------------
# memory commands
# --------------------------------------------------------------------------


def test_runs_chain_and_search(ws, capsys):
    run_cli("run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--mode", "tournament", "--max-iterations", "10", "--offline")
    run_id = load_state(ws)["run_id"]
    capsys.readouterr()

    assert run_cli("runs", "--workspace", str(ws)) == 0
    assert run_id in capsys.readouterr().out

    assert run_cli("chain", run_id, "--workspace", str(ws)) == 0
    chain_out = capsys.readouterr().out
    for stage in ("contract", "evidence", "proposals", "scores", "decisions"):
        assert f"--- {stage}" in chain_out

    assert run_cli("chain", run_id, "--workspace", str(ws), "--json") == 0
    assert "proposals" in json.loads(capsys.readouterr().out)

    assert run_cli("search", "sparse matrix", "--workspace", str(ws)) == 0
    assert capsys.readouterr().out.strip()


# --------------------------------------------------------------------------
# contract editing
# --------------------------------------------------------------------------


def test_set_metric_is_recorded_as_a_versioned_revision(ws):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--domain", "ai_algorithm", "--max-iterations", "10", "--offline",
        "--set-metric", "speedup_vs_baseline=4.0",
    ) == 0
    contract = json.loads(
        (run_dir(ws) / "research_contract.json").read_text(encoding="utf-8")
    )
    assert contract["version"] == 2
    assert contract["revisions"][-1]["author"] == "user"
    assert "4.0" in contract["revisions"][-1]["reason"]
    metric = next(
        m for m in contract["success_metrics"] if m["key"] == "speedup_vs_baseline"
    )
    assert metric["target_threshold"] == 4.0
    # Supplying a threshold promotes the metric from a harness guess to a
    # requester-stated goal, which is what protects it from --remove-metric.
    assert metric["origin"] == "user_specified"
    # The original idea survives the revision untouched.
    assert contract["original_idea"] == ALGORITHM_IDEA


def test_setting_and_removing_the_same_metric_is_refused(ws, capsys):
    """Contradictory input must be rejected, not silently resolved one way."""
    rc = run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--domain", "ai_algorithm", "--max-iterations", "10", "--offline",
        "--set-metric", "speedup_vs_baseline=4.0",
        "--remove-metric", "speedup_vs_baseline",
    )
    assert rc == 2
    assert "opposite intentions" in capsys.readouterr().err


def test_removing_an_assumed_placeholder_metric_is_allowed(ws):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--domain", "ai_algorithm", "--max-iterations", "10", "--offline",
        "--remove-metric", "task_metric",
    ) == 0
    contract = json.loads(
        (run_dir(ws) / "research_contract.json").read_text(encoding="utf-8")
    )
    assert "task_metric" not in {m["key"] for m in contract["success_metrics"]}
    assert contract["success_metrics"], "cannot remove every metric"
    assert "removed assumed placeholder" in contract["revisions"][-1]["reason"]


def test_unknown_metric_key_is_rejected(ws, capsys):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--max-iterations", "10", "--offline", "--set-metric", "nonexistent=1.0",
    ) == 2
    assert "unknown metric" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Real measurement through the pytest adapter  (Example A)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (EXAMPLES / "algorithm_idea" / "tests").is_dir(),
    reason="bundled example A is not present",
)
def test_real_experiment_reaches_l5_on_measured_data(ws, capsys):
    """Example A runs genuinely: real code, real timing, real threshold check."""
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--domain", "ai_algorithm",
        "--adapter", "pytest",
        "--project-dir", str(EXAMPLES / "algorithm_idea"),
        "--test-target", "tests",
        "--set-metric", "speedup_vs_baseline=5.0",
        "--remove-metric", "task_metric",
        "--depth", "R5_GOAL_ACHIEVED",
        "--max-iterations", "10", "--offline",
    ) == 0
    capsys.readouterr()

    state = load_state(ws)
    assert state["final_maturity_level"] == "R5_GOAL_ACHIEVED"
    assert state["stop_decision"]["reason"] == "goal_satisfied"

    experiment = state["experiments"][-1]
    assert experiment["is_simulated"] is False
    assert experiment["executed"] is True
    assert experiment["error"] is None

    speedup = next(
        o for o in experiment["observations"] if o["metric_key"] == "speedup_vs_baseline"
    )
    assert speedup["provenance"] == "measured", "this must be a real measurement"
    assert speedup["value"] > 5.0
    assert speedup["n_repeats"] >= 3
    assert speedup["stddev"] is not None

    validation = state["validations"][-1]
    assert validation["implementation_conforms"] is True
    assert validation["goal_satisfied"] is True

    report = (run_dir(ws) / "final_report.md").read_text(encoding="utf-8")
    assert "R5_GOAL_ACHIEVED" in report
    assert "| measured |" in report
    # Raw output from the real run is retained for inspection.
    assert list((run_dir(ws) / "experiments").rglob("pytest_output.txt"))


@pytest.mark.skipif(
    not (EXAMPLES / "algorithm_idea" / "tests").is_dir(),
    reason="bundled example A is not present",
)
def test_unreachable_threshold_is_reported_as_not_met_not_as_success(ws, capsys):
    """Same real experiment, impossible bar: the honest answer is 'not met'."""
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--domain", "ai_algorithm",
        "--adapter", "pytest",
        "--project-dir", str(EXAMPLES / "algorithm_idea"),
        "--test-target", "tests",
        "--set-metric", "speedup_vs_baseline=100000.0",
        "--remove-metric", "task_metric",
        "--depth", "R5_GOAL_ACHIEVED",
        "--max-iterations", "10", "--offline",
    ) == 0
    capsys.readouterr()

    state = load_state(ws)
    assert state["final_maturity_level"] != "R5_GOAL_ACHIEVED"
    validation = state["validations"][-1]
    assert validation["goal_satisfied"] is False
    # Correctly built, but the goal was not reached -- the two are distinguished.
    assert validation["implementation_conforms"] is True
    assert "did not reach" in validation["blocking_reason"]


# --------------------------------------------------------------------------
# Product domain, same core
# --------------------------------------------------------------------------


def test_product_idea_is_inconclusive_with_an_external_plan(ws, capsys):
    assert run_cli(
        "run", "--idea", PRODUCT_IDEA, "--workspace", str(ws),
        "--adapter", "manual", "--depth", "R4_VALIDATED", "--max-iterations", "10", "--offline",
    ) == 0
    capsys.readouterr()

    state = load_state(ws)
    assert state["final_maturity_level"] == "INCONCLUSIVE"
    assert state["validations"][-1]["external_validation_plan"]
    assert state["contract"]["domain_hint"] == "product_business"

    report = (run_dir(ws) / "final_report.md").read_text(encoding="utf-8")
    assert "Requires real-world validation (not performed here)" in report
    assert "INCONCLUSIVE" in report
    # An external validation plan file was actually written for a human to use.
    assert list((run_dir(ws) / "experiments").rglob("EXTERNAL_VALIDATION_PLAN.md"))


def test_jev_judge_refuses_rather_than_guessing_an_api(ws):
    with pytest.raises(SystemExit, match="not implemented"):
        run_cli(
            "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--judge", "jev", "--max-iterations", "10", "--offline",
        )


def test_llm_judge_requires_a_real_provider(ws):
    with pytest.raises(SystemExit, match="needs a real provider"):
        run_cli(
            "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--judge", "llm", "--max-iterations", "10", "--offline",
        )


def test_no_search_backend_is_reported_rather_than_filled_in(ws, capsys):
    assert run_cli(
        "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
        "--search", "none", "--max-iterations", "10", "--offline",
    ) == 0
    capsys.readouterr()
    state = load_state(ws)
    assert not state["evidence"], "no backend must mean no evidence, not invented evidence"
    assert any("No evidence retrieval was possible" in n for n in state["notes"])

    report = (run_dir(ws) / "final_report.md").read_text(encoding="utf-8")
    assert "No real evidence was gathered" in report
    # With nothing to go on, evidence-dependent dimensions must be UNKNOWN.
    scores = json.loads(
        (run_dir(ws) / "evaluation" / "scores.json").read_text(encoding="utf-8")
    )
    for ev in scores:
        strength = next(
            s for s in ev["dimension_scores"] if s["dimension_key"] == "evidence_strength"
        )
        assert strength["status"] == "UNKNOWN"
        assert strength["raw"] is None


# --------------------------------------------------------------------------
# benchmark / calibration commands
# --------------------------------------------------------------------------


def test_benchmark_command_passes_and_exits_zero(ws, capsys):
    """Usable as a CI gate: exit 0 on pass, non-zero on failure."""
    assert run_cli("benchmark", "--workspace", str(ws)) == 0
    out = capsys.readouterr().out
    assert "Result: PASS" in out
    assert "pairwise agreement (dominance)" in out
    assert "dominance by construction" in out  # scope caveat is shown


def test_benchmark_json_and_file_output(ws, tmp_path, capsys):
    out_path = tmp_path / "bench.json"
    assert run_cli(
        "benchmark", "--workspace", str(ws), "--json", "--out", str(out_path)
    ) == 0
    capsys.readouterr()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert len(data["cases"]) == 3
    assert data["pairwise_agreement"] == 1.0


def test_benchmark_fails_loudly_for_a_biased_judge(ws, capsys, monkeypatch):
    """A broken judge must make the gate fail, not warn."""
    from research_harness.evaluation.judge_adapter import MockJudgeAdapter

    monkeypatch.setattr(
        cli, "_build_judge", lambda a, p: MockJudgeAdapter(always_prefer_first_slot=True)
    )
    assert run_cli("benchmark", "--workspace", str(ws)) == 1
    assert "Result: FAIL" in capsys.readouterr().out


def test_calibration_refuses_to_report_on_simulated_runs(ws, capsys):
    run_cli("init", "--workspace", str(ws))
    run_cli("run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--mode", "tournament", "--max-iterations", "10", "--offline")
    capsys.readouterr()

    assert run_cli("calibration", "--workspace", str(ws)) == 0
    out = capsys.readouterr().out
    assert "excluded, simulated run     : 1" in out
    assert "sufficient data             : False" in out
    assert "Insufficient data" in out
    assert "not a probability" in out


@pytest.mark.skipif(
    not (EXAMPLES / "algorithm_idea" / "tests").is_dir(),
    reason="bundled example A is not present",
)
def test_calibration_uses_real_runs_once_enough_accumulate(ws, capsys):
    """Accumulate real measured runs with mixed outcomes, then calibrate."""
    run_cli("init", "--workspace", str(ws))
    # Reachable thresholds -> achieved; absurd ones -> genuinely not achieved.
    thresholds = [2.0, 3.0, 4.0, 5.0, 6.0, 1e5, 2e5, 3e5, 4e5, 5e5]
    for i, t in enumerate(thresholds):
        run_cli(
            "run", "--idea", ALGORITHM_IDEA, "--workspace", str(ws),
            "--run-id", f"cal{i:02d}", "--domain", "ai_algorithm",
            "--adapter", "pytest",
            "--project-dir", str(EXAMPLES / "algorithm_idea"),
            "--test-target", "tests",
            "--set-metric", f"speedup_vs_baseline={t}",
            "--remove-metric", "task_metric",
            "--depth", "R5_GOAL_ACHIEVED", "--max-iterations", "10", "--offline",
        )
    capsys.readouterr()

    assert run_cli("calibration", "--workspace", str(ws), "--json") == 0
    data = json.loads(capsys.readouterr().out)

    assert data["n_excluded_simulated"] == 0, "these were real pytest runs"
    assert data["n_labelled"] == len(thresholds)
    assert data["sufficient_data"] is True
    assert data["discrimination_auc"] is not None
    # Both outcomes really occurred, so the labels are informative.
    achieved = [r for r in data["records"] if r["goal_achieved"]]
    assert 0 < len(achieved) < len(thresholds)
    assert all(r["was_simulated"] is False for r in data["records"])
