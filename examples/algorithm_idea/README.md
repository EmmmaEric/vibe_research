# Example A — a real computational experiment

Demonstrates the harness reaching **R5_GOAL_ACHIEVED** on genuinely measured
data: real code executes, real timings are taken, and the contract's threshold
is checked against them by deterministic code.

## What is measured

A CSR (compressed sparse row) matrix-vector product against a dense baseline on
a 320×320 matrix at ~2% density.

- `impl/baseline.py` — dense matvec, visiting every entry including zeros.
  Written in the same pure-Python idiom as the candidate: an under-tuned
  baseline is the classic way to manufacture a speedup, so it is deliberately
  not handicapped.
- `impl/candidate.py` — CSR, visiting only stored non-zeros.
- `tests/test_experiment.py` — the experiment.

Measurements use the **median** of 5 repeats (the minimum flatters whichever
side got the luckiest scheduling slice), report standard deviation, and fix the
seed. The comparison only passes if the gap exceeds combined run-to-run spread.

Correctness is a differential test against the baseline — this is the
`numerical_equivalence` hard constraint made executable. A faster wrong answer
is not an improvement.

The edge-case test asserts the candidate's **own limit**: at full density CSR
stores *more* than dense. The memory advantage is conditional on sparsity, and
the suite proves it rather than footnoting it.

## Run it

Through the harness:

```bash
research-harness run \
  --idea "Make sparse matrix-vector multiplication faster on commodity CPUs
          while keeping memory below the dense representation" \
  --domain ai_algorithm \
  --adapter pytest --project-dir examples/algorithm_idea --test-target tests \
  --set-metric speedup_vs_baseline=5.0 --remove-metric task_metric \
  --depth R5_GOAL_ACHIEVED
```

Standalone:

```bash
cd examples/algorithm_idea && RESULTS_PATH=results.json python -m pytest tests -q -s
```

## Adapter contract

The suite writes `results.json` at `$RESULTS_PATH`:

```json
{
  "seed": 20251030,
  "tests": {"T001": true, "T002": true},
  "observations": {
    "speedup_vs_baseline": {"value": 33.4, "unit": "x",
                            "n_repeats": 5, "stddev": 3.8, "seed": 20251030}
  }
}
```

Test ids key into the generated plan. The adapter **ignores ids not in the
plan** and records plan cases absent from the file as "not executed" — it never
credits a pass to a case that did not run. Only contract metrics count toward
goal attainment.

## Try making it fail honestly

```bash
  --set-metric speedup_vs_baseline=100000.0
```

Same real measurement, unreachable bar. The harness reports
`goal_satisfied: false` while `implementation_conforms` stays `true` — built
correctly, goal not met. That separation is the point of the two loops.

## Caveats

Wall-clock timings are host-specific; the speedup *ratio* is the portable
figure, and even it depends on the density above. Both implementations are pure
Python — this measures an algorithmic effect, not production performance, where
a BLAS-backed library would dominate both.
