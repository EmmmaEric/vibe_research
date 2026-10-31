"""Real experiment for Example A, executed by the harness's pytest adapter.

This suite performs actual computation and actual timing. Nothing here is
simulated, so the measurements it reports carry ``Provenance.MEASURED`` and can
legitimately support validation level L4/L5.

Adapter contract: write ``results.json`` (path from ``$RESULTS_PATH``) mapping
test-plan case ids to pass/fail, plus observations for contract metrics. The
ids below correspond to the plan the Test Designer generates for the
``ai_algorithm`` profile:

    T001 contract         -- declared interfaces exist
    T002 correctness      -- differential vs. the baseline on known inputs
    T003 edge_robustness  -- boundaries, malformed input, the density limit
    T004 comparison       -- timed against the independent baseline
    T005 goal_acceptance  -- speedup_vs_baseline meets the contract threshold
    T006 evidence_quality -- repeats and variance actually recorded

If the generated plan ever uses different ids, the adapter ignores unknown ones
and records the plan's own cases as "not executed" -- it does not credit a pass
to a case that was never run.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from impl.baseline import dense_matvec  # noqa: E402
from impl.candidate import CSRMatrix, csr_matvec, to_csr  # noqa: E402

SEED = 20251030
N_ROWS = 320
N_COLS = 320
DENSITY = 0.02
REPEATS = 5
TOLERANCE = 1e-9

_RESULTS: dict[str, bool] = {}
_OBSERVATIONS: dict[str, dict] = {}


def _make_sparse(seed: int = SEED) -> tuple[list[list[float]], list[float]]:
    """Deterministic sparse test matrix and vector."""
    rng = random.Random(seed)
    matrix = [
        [rng.uniform(-1.0, 1.0) if rng.random() < DENSITY else 0.0 for _ in range(N_COLS)]
        for _ in range(N_ROWS)
    ]
    vector = [rng.uniform(-1.0, 1.0) for _ in range(N_COLS)]
    return matrix, vector


def _record(test_id: str, passed: bool) -> None:
    _RESULTS[test_id] = passed


# --------------------------------------------------------------------------
# T001 -- contract: the declared interfaces exist and are callable
# --------------------------------------------------------------------------


def test_t001_interfaces_present():
    ok = False
    try:
        csr = to_csr([[1.0, 0.0], [0.0, 2.0]])
        assert isinstance(csr, CSRMatrix)
        assert csr.n_rows == 2 and csr.n_cols == 2
        assert csr.nnz == 2
        assert csr.row_pointers == [0, 1, 2]
        y = csr_matvec(csr, [1.0, 1.0])
        assert y == [1.0, 2.0]
        assert callable(dense_matvec)
        assert csr.storage_units() > 0
        ok = True
    finally:
        _record("T001", ok)


# --------------------------------------------------------------------------
# T002 -- correctness: differential test against the baseline
# --------------------------------------------------------------------------


def test_t002_matches_baseline():
    """The candidate must agree with the baseline within numerical tolerance.

    This is the hard constraint `numerical_equivalence` made executable: a
    faster wrong answer is not an improvement.
    """
    ok = False
    try:
        max_abs_err = 0.0
        for seed in (SEED, SEED + 1, SEED + 2):
            matrix, vector = _make_sparse(seed)
            expected = dense_matvec(matrix, vector)
            actual = csr_matvec(to_csr(matrix), vector)
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                max_abs_err = max(max_abs_err, abs(a - b))
        assert max_abs_err <= TOLERANCE, (
            f"max absolute divergence {max_abs_err} exceeds tolerance {TOLERANCE}"
        )
        _OBSERVATIONS["max_abs_divergence"] = {
            "value": max_abs_err,
            "unit": "absolute",
            "n_repeats": 3,
            "seed": SEED,
        }
        ok = True
    finally:
        _record("T002", ok)


# --------------------------------------------------------------------------
# T003 -- edge / robustness
# --------------------------------------------------------------------------


def test_t003_edges_and_known_limit():
    ok = False
    try:
        # Empty matrix.
        assert csr_matvec(to_csr([]), []) == []
        # All-zero matrix: no stored entries, result is zeros.
        zeros = to_csr([[0.0, 0.0], [0.0, 0.0]])
        assert zeros.nnz == 0
        assert csr_matvec(zeros, [3.0, 4.0]) == [0.0, 0.0]
        # Single element.
        assert csr_matvec(to_csr([[2.0]]), [3.0]) == [6.0]

        # Dimension mismatch must raise, not silently truncate.
        for bad in ([1.0], [1.0, 2.0, 3.0]):
            try:
                csr_matvec(to_csr([[1.0, 0.0]]), bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"dimension mismatch {bad} was not rejected")

        # Ragged input must raise.
        try:
            to_csr([[1.0, 0.0], [1.0]])
        except ValueError:
            pass
        else:
            raise AssertionError("ragged matrix was not rejected")

        # The declared memory limit: CSR is NOT smaller at high density.
        dense_full = [[1.0] * 16 for _ in range(16)]
        csr_full = to_csr(dense_full)
        assert csr_full.storage_units() > 16 * 16, (
            "expected CSR to be larger than dense at full density; the candidate's "
            "memory advantage is conditional on sparsity"
        )
        ok = True
    finally:
        _record("T003", ok)


# --------------------------------------------------------------------------
# T004 / T005 -- timed comparison against the independent baseline
# --------------------------------------------------------------------------


def _time(fn, repeats: int = REPEATS) -> tuple[float, float, list[float]]:
    """Best-of timing with all samples retained.

    Reports the *median* rather than the minimum: the minimum flatters whichever
    implementation got the luckiest scheduling slice.
    """
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return median, stdev, samples


def test_t004_t005_speedup_vs_baseline():
    ok_cmp = False
    ok_goal = False
    try:
        matrix, vector = _make_sparse()
        csr = to_csr(matrix)

        # Warm up both paths so neither pays first-call costs in the measurement.
        dense_matvec(matrix, vector)
        csr_matvec(csr, vector)

        base_med, base_sd, base_all = _time(lambda: dense_matvec(matrix, vector))
        cand_med, cand_sd, cand_all = _time(lambda: csr_matvec(csr, vector))

        assert cand_med > 0, "timer resolution too coarse to measure"
        speedup = base_med / cand_med

        _OBSERVATIONS["speedup_vs_baseline"] = {
            "value": round(speedup, 4),
            "unit": "x",
            "n_repeats": REPEATS,
            # Propagated relative error of the ratio, not the stddev of one side.
            "stddev": round(
                speedup
                * ((base_sd / base_med) ** 2 + (cand_sd / cand_med) ** 2) ** 0.5,
                4,
            ),
            "seed": SEED,
        }
        _OBSERVATIONS["baseline_median_seconds"] = {
            "value": round(base_med, 6),
            "unit": "s",
            "n_repeats": REPEATS,
            "stddev": round(base_sd, 6),
            "seed": SEED,
        }
        _OBSERVATIONS["candidate_median_seconds"] = {
            "value": round(cand_med, 6),
            "unit": "s",
            "n_repeats": REPEATS,
            "stddev": round(cand_sd, 6),
            "seed": SEED,
        }

        # T004: measurably faster than the baseline, beyond run-to-run spread.
        ok_cmp = speedup > 1.0 and (base_med - cand_med) > (base_sd + cand_sd)
        # T005: the harness re-checks the contract threshold itself; this records
        # whether the measurement was obtained cleanly enough to be judged.
        ok_goal = ok_cmp

        print(
            f"\nmeasured: baseline {base_med:.6f}s (sd {base_sd:.6f}), "
            f"candidate {cand_med:.6f}s (sd {cand_sd:.6f}), "
            f"speedup {speedup:.2f}x over {REPEATS} repeats, "
            f"density {csr.density:.4f}, nnz {csr.nnz}"
        )
        assert ok_cmp, (
            f"speedup {speedup:.2f}x is not separable from measurement noise "
            f"(baseline sd {base_sd:.6f}, candidate sd {cand_sd:.6f})"
        )
    finally:
        _record("T004", ok_cmp)
        _record("T005", ok_goal)


# --------------------------------------------------------------------------
# T006 -- evidence quality
# --------------------------------------------------------------------------


def test_t006_evidence_quality():
    """Were the measurements recorded well enough to support a conclusion?"""
    ok = False
    try:
        assert "speedup_vs_baseline" in _OBSERVATIONS, "no speedup was recorded"
        for key, obs in _OBSERVATIONS.items():
            assert obs["n_repeats"] >= 3, f"{key} has too few repeats"
            assert obs.get("seed") is not None, f"{key} has no recorded seed"
        timed = ("speedup_vs_baseline", "baseline_median_seconds")
        for key in timed:
            assert _OBSERVATIONS[key].get("stddev") is not None, (
                f"{key} has no variance recorded; a timing without spread is not a "
                "measurement"
            )
        ok = True
    finally:
        _record("T006", ok)


# --------------------------------------------------------------------------
# Emit results for the adapter
# --------------------------------------------------------------------------


def test_zz_write_results():
    """Runs last (alphabetical ordering) and writes the adapter's results file."""
    path = Path(os.environ.get("RESULTS_PATH", "results.json"))
    payload = {
        "seed": SEED,
        "tests": _RESULTS,
        "observations": _OBSERVATIONS,
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "n_rows": N_ROWS,
            "n_cols": N_COLS,
            "density_target": DENSITY,
            "repeats": REPEATS,
        },
        "note": (
            "Real measurements from executed code. Timings are wall-clock on a "
            "shared machine and will vary between hosts; the speedup ratio is the "
            "portable figure, and even it depends on the matrix density above."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert path.is_file()
