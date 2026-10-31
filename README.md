# Adaptive Universal Research Harness

A domain-agnostic autonomous research harness. You give it a goal; it decides how
much machinery that goal actually needs, gathers evidence, designs competing
candidates when they're warranted, scores them with a deterministic decision
engine, plans independent verification, runs what it can actually run, and
reports **how far verification really got** — including when the answer is "not
far enough to say".

**There is no fixed pipeline.** Each iteration an adaptive router chooses one
action from explicit signals:

```
CONTINUE_SINGLE_AGENT · GATHER_EVIDENCE · SPAWN_SUBAGENT
RUN_PROPOSAL_TOURNAMENT · RUN_EXPERIMENT · REQUEST_HUMAN · STOP
```

The organising principle: **goals outrank candidates, evidence outranks agent
consensus, independent verification outranks self-assessment, and "we cannot
tell yet" is a valid result.**

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Linux/macOS: .venv/bin/python

research-harness init
research-harness run --idea "your goal here" --offline
```

No API key, no network. Completes in a couple of seconds.

```bash
.venv/Scripts/python -m pytest          # 336 tests, fully offline
```

---

## The three modes, and why the baseline is structural

Spec §16 asks for a Goal-Only baseline that richer modes must justify themselves
against. A baseline implemented as "the full system with a flag" proves nothing,
so `goal_only` is restricted **structurally**: it cannot reach the multi-agent
actions at all.

| Mode | Capability | Candidates |
|---|---|---|
| `goal_only` | one general agent + tools + independent validation | exactly 1 |
| `adaptive` | router may escalate to subagents / tournaments on demand | 1–N |
| `tournament` | candidates always compete head to head | N |

```bash
research-harness compare --idea "your goal" --offline
```

```
mode        maturity          outcome              cand  iter  exp
goal_only   R3_PROTOTYPED     STAGNATED               1     7    3
adaptive    R3_PROTOTYPED     STAGNATED               3     6    1
tournament  R3_PROTOTYPED     STAGNATED               3     6    1

Route taken per arm:
  goal_only   GATHER_EVIDENCE -> CONTINUE_SINGLE_AGENT -> RUN_EXPERIMENT -> ...
  adaptive    GATHER_EVIDENCE -> CONTINUE_SINGLE_AGENT -> RUN_EXPERIMENT -> RUN_PROPOSAL_TOURNAMENT -> ...
  tournament  GATHER_EVIDENCE -> RUN_PROPOSAL_TOURNAMENT -> RUN_EXPERIMENT -> ...

Observations:
  - 'adaptive' reached the same maturity as the baseline while spending
    no extra model cost. On this instance the extra machinery bought no
    measurable gain.
```

That last line is the point. One run per mode cannot establish that a richer
mode is better, and the report says so rather than rounding it up.

---

## What the default run is, and is not

The offline demo exercises **real machinery on labelled placeholder content**:

| Component | In the offline demo | Real? |
|---|---|---|
| Router, orchestrator, state, resume | the actual implementation | **yes** |
| Scoring, constraint gate, ranking, Pareto, sensitivity | the actual implementation | **yes** |
| Test-plan authoring and custody | the actual implementation | **yes** |
| Budget, stop policy, approval gates, event log | the actual implementation | **yes** |
| Research / proposal / critique / judge **content** | deterministic local heuristics | **no — `[SIMULATED]`** |
| Evidence | bundled fixtures | **no — `provenance: simulated`** |
| Experiment results | synthetic from a mock adapter | **no — capped at R3 in code** |

Simulated content is marked in the *data*, not just in prose, so the decision
engine acts on it: a run touching the mock adapter cannot report above
`R3_PROTOTYPED`, whatever the numbers say.

Make each part real independently: `--provider openai|anthropic` (model),
`--search web` (evidence), `--adapter pytest` (measurement).

### Using a real model

```bash
# Any OpenAI-compatible endpoint, incl. proxies. Reads baseURL/apiKey from an
# opencode config; the key is used in-process and never written to an artifact.
research-harness run --idea "..."   --provider openai --provider-key local --model gpt-6.1-sol --judge llm

# Or from the environment: OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
research-harness run --idea "..." --provider openai --judge llm

# Claude via the official SDK (needs `pip install anthropic`)
research-harness run --idea "..." --provider anthropic --judge llm
```

Structured output is tiered automatically — `json_schema` (strict, with the
schema rewritten so every property is `required`, which real endpoints demand),
falling back to `json_object`, then to schema-in-prompt. The working tier is
discovered once per run.

**Metric names are emergent with a real model**, and differ between runs — five
real runs on one idea produced seven namings. So real measurement is a two-phase
flow:

```bash
# 1. Establish the contract and read the metric keys it actually declared.
research-harness run --idea "..." --provider openai --depth R1_RESEARCHED --run-id r1
cat runs/research_run_r1/research_contract.json

# 2. Deepen the SAME contract, binding the suite's output to those keys.
research-harness resume r1 --depth R5_GOAL_ACHIEVED   --provider openai --judge llm   --adapter pytest --project-dir examples/algorithm_idea   --map-metric spmv_speedup=speedup_vs_baseline
```

`resume --depth` continues on the already-fixed, versioned contract, so keys
read a moment ago stay valid; starting a fresh run would regenerate them.
Lowering the requested maturity is refused — asking for less verification after
seeing results is how a goal gets quietly weakened.

The binding is a research judgement, so it is recorded on the experiment and
never guessed by fuzzy name matching. A key the contract does not declare is a
hard error listing the real ones, not a silent no-op.

---

## A real experiment, end to end

Example A is a genuine computational experiment — CSR sparse matrix-vector
product against a dense baseline, median of 5 repeats, variance, fixed seed, and
a differential correctness check.

```bash
research-harness run \
  --idea "Make sparse matrix-vector multiplication faster on commodity CPUs
          while keeping memory below the dense representation" \
  --domain ai_algorithm \
  --adapter pytest --project-dir examples/algorithm_idea \
  --set-metric speedup_vs_baseline=5.0 --remove-metric task_metric \
  --depth R5_GOAL_ACHIEVED --offline
```

```
Level:    R5_GOAL_ACHIEVED
Outcome:  GOAL_ACHIEVED
Route:    GATHER_EVIDENCE -> SPAWN_SUBAGENT -> RUN_EXPERIMENT -> STOP
Goal met: yes

| Contract metric     | Target       | Observed | Met | Provenance |
| speedup_vs_baseline | maximize 5.0 | 33.42    | yes | measured   |
```

`measured`, not `simulated`. Set the bar where it cannot be reached and the
harness says so instead — while keeping `implementation_conforms: true`, because
"built right" and "goal met" are different findings.

### Example B: when honesty means INCONCLUSIVE

```bash
research-harness run \
  --idea "Design an offline document collaboration product for small businesses
          and assess its commercial viability" \
  --mode tournament --adapter manual --depth R4_VALIDATED --offline
```

```
Level:    INCONCLUSIVE
Outcome:  REQUIRES_EXTERNAL_TEST
Route:    GATHER_EVIDENCE -> RUN_PROPOSAL_TOURNAMENT -> RUN_EXPERIMENT -> REQUEST_HUMAN
```

It writes `EXTERNAL_VALIDATION_PLAN.md` with the interviews and A/B tests a
person must run, and stops rather than spending budget on simulations that
cannot answer the question.

---

## The routing rules

Ordered; first match wins. Order encodes precedence.

| # | Rule | Action |
|---|---|---|
| 1 | unauthorised action pending | `REQUEST_HUMAN` |
| 2 | gated stage awaiting approval | `REQUEST_HUMAN` (pauses, resumable) |
| 3 | budget exhausted | `STOP` |
| 4 | goal independently validated | `STOP` |
| 5 | requested depth reached | `STOP` |
| 6 | needs real users / devices | `REQUEST_HUMAN` |
| 7 | no gain for N iterations | `STOP` (stagnated) |
| 8 | conformance failed, repairs remain | `RUN_EXPERIMENT` (Loop A) |
| 9 | no real evidence, or a critical unknown | `GATHER_EVIDENCE` |
| 10 | a cheap decisive test exists | `RUN_EXPERIMENT` |
| 11 | tournament mode requested | `RUN_PROPOSAL_TOURNAMENT` |
| 12 | ranking unstable, or top two within 8 pts | `RUN_PROPOSAL_TOURNAMENT` |
| 13 | built right but goal missed, alternatives remain | `RUN_PROPOSAL_TOURNAMENT` |
| 14 | complexity ≥ 0.7, or specialist review needed | `SPAWN_SUBAGENT` |
| 15 | plan ready, never executed | `RUN_EXPERIMENT` |
| 16 | nothing left to try | `STOP` |
| — | default | `CONTINUE_SINGLE_AGENT` |

**Rules decide; the model advises.** A model suggestion is recorded alongside
each decision — including when it disagreed and was overridden
(`overrode_model`). Every decision carries the rule that fired, the signal
snapshot, a predicted cost, and the realised effect afterwards.

---

## The scoring engine

Entirely deterministic code.

**1. Hard constraints first, and they do not trade.** `any FAIL → FAIL`,
`else any UNKNOWN → UNKNOWN`, `else PASS`. A `FAIL` makes a candidate ineligible
regardless of score — one scoring 100.0 with a violation loses to one scoring
40.0 without. An unassessed constraint becomes `UNKNOWN`, never an implicit pass.

**2. Anchored dimensions.** Eight catalogue dimensions, each with 0–5 semantic
anchors and an explicit boundary note against its neighbours. Cost is scored as
*favourability* so every dimension is higher-is-better and no inversion can
hide. Novelty enters only when the goal rewards it.

**3. Weighted aggregation with honest uncertainty.** A dimension lacking evidence
is `UNKNOWN` and carries **no number** — the schema rejects one. You get a point
estimate, a `[lower, upper]` band bounding the unknowns, and
`evidence_sufficiency`. There is deliberately no "confidence penalty" multiplier.

Ranking uses `score_lower`, not `point_score`: the point estimate renormalises
over scored dimensions, which makes it incomparable between candidates with
different evidence coverage — dropping a below-average dimension *raises* it, so
citing nothing could outrank citing a real source. That was a live bug, caught by
the benchmark (ADR-014). Missing evidence can now only hurt.

**4. Pairwise, with bias measured.** Every pair is judged twice with positions
swapped over length-normalised, id-redacted cards. `insufficient_evidence` is a
permitted verdict. A judge that picks slot A both times is detected and the run
warns that the ordering is unreliable.

**5. Pareto, beside the ranking.** When dimensions genuinely cannot trade off,
the non-dominated set is the honest answer. Reported alongside, never instead —
so the weights' contribution stays visible.

**6. Sensitivity and invariance.** Each weight perturbed ±20%; separately,
candidates relabelled and reordered, where the ranking must be identical.

**7. Scope.** Every recommendation is phrased as *better among the candidates
considered, under current evidence and weights*. Never "optimal".

---

## Maturity levels

| Level | Meaning |
|---|---|
| `R1_RESEARCHED` | sourced research completed |
| `R2_DESIGNED` | design + verification plan; nothing executed |
| `R3_PROTOTYPED` | a PoC ran — **the ceiling for simulated adapters** |
| `R4_VALIDATED` | independently and repeatably validated |
| `R5_GOAL_ACHIEVED` | every required metric measured, met, constraints satisfied |
| `INCONCLUSIVE` | evidence insufficient to assert success or failure |
| `REJECTED` | violates a hard constraint |

Separately, `RunOutcome` says *why the run ended*: `GOAL_ACHIEVED`,
`IMPLEMENTATION_BUG`, `PROPOSAL_INVALID`, `EVIDENCE_INSUFFICIENT`,
`REQUIRES_EXTERNAL_TEST`, `BUDGET_EXHAUSTED`, `STAGNATED`,
`UNSAFE_OR_UNAUTHORIZED`, `HUMAN_ACTION_REQUIRED`, `BLOCKED`.

Three rules enforced in code, not prose:

- **Simulated results cannot exceed R3.** `goal_satisfied` is `True` only at R5.
- **Unmeasured ≠ failed.** A metric with no observation gets `met=None`.
- **Attempted-but-blocked ≠ never-run** (ADR-012 — this was a real bug).

---

## The two loops

Kept apart by *which test categories answer them*:

**Loop A** — did we build what the candidate specified? Answered by
`contract` / `correctness` / `edge_robustness` cases. May propose fixes; cannot
touch acceptance criteria. Bounded by `max_repair_attempts`.

**Loop B** — does the original goal hold? Answered only by `goal_acceptance`
cases against contract thresholds. May gather evidence, re-compete candidates,
request external validation, or conclude.

A correct implementation that misses the goal is a *research finding*, so it
never enters Loop A.

---

## Guardrails

- **`original_idea` is immutable.** Revisions are versioned with a reason and
  author.
- **Three-way provenance.** Every metric and constraint is `user_specified`,
  `agent_proposed` or `assumed`. `--remove-metric` drops harness placeholders
  but **refuses** requester-stated ones.
- **Test plans authored at candidate selection**, from the contract, hashed, and
  verified before and after each run. `TestPlanRegistry.amend` raises for any
  execution role; `Executor.amend_plan` exists only to refuse.
- **Resume cannot replay a side effect.** Idempotency keys derived from action
  identity (never from a timestamp), restored with the event log.
- **Approval gates pause, not end.** `resume` refuses to pass an unapproved
  gate; only `approve` does, and it records who granted it. Device control,
  payments, publication and contacting real users are *always* gated.
- **No invented APIs.** `JevJudgeAdapter` raises with implementation
  instructions rather than guessing a third-party interface.
- **No fabricated sources.** With no backend, evidence is empty and
  `evidence_strength` is `UNKNOWN` for every candidate.
- **Every run terminates** for a stated reason.
- **Secrets never reach an artifact.** Keys come from the environment only;
  `redacted_effective_config.json` is redacted by key shape, so a *new* secret
  field is covered by default.

`tools/execution.py` is the only place a subprocess runs: argument lists (never
a shell string), executable denylist, root confinement, environment allowlist
that strips API keys, hard timeout. **Defence in depth, not a security
boundary** — for untrusted code, use a container.

---

## CLI

```bash
research-harness init
research-harness run --idea "..." [--idea-file F] [--mode M] [--offline]
research-harness compare --idea "..." [--modes goal_only adaptive]
research-harness status | resume | proposals | evaluate | events <run_id>
research-harness weights <run_id> [--set dim=w]
research-harness approve <run_id> --stage experiment
research-harness report  <run_id> [--format markdown|json|summary]
research-harness benchmark [--json] [--out F]   # grade the scoring system
research-harness calibration [--json]           # did scores predict outcomes?
research-harness chain <run_id> | runs | search "query"
```

Key options: `--mode goal_only|adaptive|tournament` ·
`--provider simulated|anthropic` · `--judge mock|llm|jev` ·
`--adapter simulated|pytest|manual` · `--search fixture|local|web|all|none` ·
`--depth R1..R5` · `--offline` · `--config DIR` · `--task FILE` ·
`--set-metric KEY=N` · `--map-metric K=OBSERVED` · `--provider-key NAME` ·
`--require-approval experiment` · `--seed`

`benchmark` exits non-zero on failure, so it works as a CI gate.

### Configuration

```
config/global_config/{models,loop,evaluation,backends,security,search}.toml
config/tasks/{demo_scientific,demo_product}.toml
```

Precedence: global → task → explicit flag. Tables deep-merge; **arrays
replace**. Unknown keys are an error, not ignored.

---

## Output

```
research_run_<id>/
├── research_contract.json          # immutable idea + versioned revisions
├── evidence.jsonl                  # every claim with provenance and limits
├── proposals/proposal_*.json
├── evaluation/
│   ├── rubric.json  scores.json  pairwise.json  ranking.json
│   ├── next_action.json            # value-of-information shortlist
│   └── ranking_report.md
├── experiments/
│   ├── test_plan_*.json            # versioned, hashed
│   └── run_001/ ...                # raw output, artefacts
├── routing_decisions.jsonl         # every decision, rule, cost, realised effect
├── events.jsonl                    # append-only audit log
├── decisions.jsonl                 # failure attributions
├── run_summary.json                # machine-readable; nulls, never placeholders
├── redacted_effective_config.json
├── state.json                      # resume from here
└── final_report.md                 # the ten required sections
```

---

## Architecture

```
src/research_harness/
├── cli.py
├── core/        schemas · state · router · orchestrator · loops · budget · ablation
├── agents/      general · goal_analyst · researcher · proposal_designer · skeptic
│                test_designer · executor · validator · diagnosis · domains
├── evaluation/  rubric · constraints · mcda · ranking (Pareto) · pairwise
│                stability · judge_adapter · next_action · decision_engine
│                benchmark · calibration
├── experiments/ planner (plan custody) · adapters · validation
├── infrastructure/ config (layered TOML + redaction)
├── memory/      models · repository · retrieval · events
├── providers/   base · simulated · openai_compatible · anthropic_provider
├── reports/     generator
└── tools/       search · files · execution
```

Pluggable at every boundary: `AgentProvider`, `JudgeAdapter`, `ResearchTool`,
`ExperimentAdapter`, `ValidationAdapter`, `RankingStrategy`, and the
`DomainProfile` registry. Adding a domain means appending a record.

No LangGraph: resume is a core feature, so we want the state object to *be* the
contract rather than owning a framework's persistence model. 28 architecture
decisions, including six bugs found during implementation:
**[`docs/ADR.md`](docs/ADR.md)**.

---

## Testing

```bash
.venv/Scripts/python -m pytest              # 336 tests, offline
```

- **unit** — router rules and mode gating; scoring invariants; constraint
  precedence; pairwise bias detection; Pareto; next-best-action; contract
  immutability; plan custody; maturity levels; config layering and redaction;
  event-log idempotency; both real providers' logic via injected stubs
  (tiering, strict-schema rewriting, retries, refusal and truncation handling)
- **integration** — the router-driven orchestrator; cross-domain generality;
  reweighting; approval pause/resume; resume without replaying side effects;
  determinism; persistence
- **e2e** — the CLI with real argv, workspaces and SQLite, including a run that
  **actually measures** Example A and reaches R5, and a three-arm ablation

Tests needing credentials are marked `requires_network` and deselected.

---

## Known limitations

Stated plainly, because a harness about honest reporting should be honest about
itself.

1. **`AnthropicProvider` has never made a real call.** Its logic is covered by
   22 tests with an injected stub client, but the actual HTTP round trip and the
   real SDK's object shapes are untested, and `anthropic` is not installed here.
   `AnthropicWebSearchTool` is wholly unexercised.
   `OpenAICompatibleProvider`, by contrast, **has** been exercised against a
   live endpoint end to end (see "Using a real model").
2. **`JevJudgeAdapter` is deliberately not implemented** (ADR-010).
3. **Teacher–Student review is not implemented.** The spec marks it optional;
   `config/global_config/models.toml` ships it disabled so enabling it would be
   a visible change rather than a silent default.
4. **AHP / TOPSIS / ELECTRE / Bradley–Terry are not implemented.** The spec says
   the MVP does not need them; `RankingStrategy` is the extension point, and an
   unused ranking algorithm would be dead weight.
5. **Simulated candidates are templated, not reasoned.** Structurally distinct
   (verified by a Jaccard check) but containing no insight. Only
   `--provider anthropic` produces genuine reasoning.
6. **Domain detection is keyword matching.** Weak by construction, which is why
   it only sets a weight prior and `assumed` placeholders, and why `--domain`
   exists.
7. **The benchmark covers dominance cases only** (ADR-015). It detects a broken
   judge; it does not establish good taste on contested trade-offs.
8. **Calibration needs accumulated real outcomes** — it excludes simulated runs
   and refuses a statistic below 10 real labels, so a fresh install correctly
   reports "insufficient data".
9. **Value-of-information impact is a flagged prediction, not calibrated.**
10. **The routing rule table is an unvalidated heuristic.** It records predicted
    cost and realised effect so it *can* be assessed, but it has not been.
11. **The execution sandbox is defence in depth, not isolation.**
12. **Retrieval is keyword/FTS5 only.** No vector store.
13. **Single-user, local, no concurrency.** The router decides one action at a
    time; the task graph exposes an independent frontier but nothing executes it
    in parallel yet.

### Next

- Execute the `ready_tasks` frontier concurrently (the graph already marks it).
- Validate the routing heuristic against accumulated `realised_*` data.
- Benchmark cases with real expert labels for contested trade-offs.

---

## Licence

A reference implementation. No licence file is included; add one before
distributing.
