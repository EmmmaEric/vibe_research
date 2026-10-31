# Example B — when the honest answer is INCONCLUSIVE

Demonstrates a domain where the harness **cannot** settle the question, and says
so instead of simulating an answer.

This directory has no code on purpose. Commercial viability is decided by real
customers, and no amount of local computation substitutes for talking to them.

## Run it

```bash
research-harness run \
  --idea "Design an offline document collaboration product for small businesses
          and assess its commercial viability" \
  --adapter manual --depth R4_VALIDATED
```

## What happens

```
Level:    INCONCLUSIVE
Stopped:  request_external_validation
Reason:   Validation requires real-world input (users, devices, or physical
          measurement) that this harness cannot obtain autonomously.
```

The run still does real work: it builds a contract with market-appropriate
metrics (`validated_demand_signal`, `unit_economics_ratio`), gathers evidence
including counter-evidence, designs three competing product routes, scores and
ranks them with constraint gating, and writes a full test plan.

It then writes `experiments/run_001/EXTERNAL_VALIDATION_PLAN.md` listing what a
person must do — structured interviews, usability testing, A/B tests — and
stops. It does **not** burn its iteration budget on simulations that cannot
answer the question.

## Feeding real findings back

Once interviews are done, record them as JSONL:

```json
{"claim": "9 of 14 interviewed firms named offline access a top-3 blocker",
 "source_type": "interview", "stance": "supports", "quality": 4,
 "related_metric": "validated_demand_signal"}
```

```bash
research-harness run --idea "..." --human-evidence findings.jsonl --search all
```

These enter the evidence pool with `provenance: human` — first-class, and never
conflated with something the harness measured itself.

## Why this matters

Example A and Example B use the **identical** `Orchestrator`, scoring engine and
`ResearchState`. Only the tools, metrics and adapters differ. One reaches L5 on
measured data; the other reaches INCONCLUSIVE and explains what would resolve
it. Both are correct outcomes, and the same core produced them —
`test_two_unrelated_domains_run_on_the_same_core` asserts it.
