# Examples

Two ideas from unrelated domains, run through the **same** core orchestrator,
scoring engine and research state. Only the tools, metrics and adapters differ —
which is what makes the architecture general rather than merely configurable.

| | [`algorithm_idea/`](algorithm_idea/) | [`product_idea/`](product_idea/) |
|---|---|---|
| Domain profile | `ai_algorithm` | `product_business` |
| Experiment adapter | `pytest` (real execution) | `manual` (human required) |
| Outcome | **R5_GOAL_ACHIEVED** on measured data | **INCONCLUSIVE** + external plan |
| Why | real code ran and met the threshold | viability needs real customers |

Both outcomes are correct. A harness that could only produce the first one would
be lying about the second.

Run them with the commands in each directory's README, or see the end-to-end
tests that assert these outcomes in `tests/e2e/test_cli.py`.
