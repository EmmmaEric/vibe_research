# Architecture Decision Record

Decisions that shape the harness, and what each one buys. Written before the
implementation and revised where the code taught us something (ADR-012).

---

## ADR-001: Deterministic core, semantic periphery

**Decision.** Every gate that changes the course of a run is ordinary Python:
score aggregation (`evaluation/mcda.py`), hard-constraint folding
(`evaluation/constraints.py`), ranking and Top-K (`evaluation/decision_engine.py`),
validation-level assignment, stop policy, and next-action selection
(`agents/diagnosis.py`). Models are used only where judgement is irreducible:
interpreting an idea, designing candidates, criticising them, and assigning
anchored per-dimension scores.

**Why.** Two reasons, one practical and one about trust.

Practically: an LLM asked to compute a weighted mean will usually get close and
occasionally not, and the failure is silent. Re-ranking on a weight change is
only meaningful if the aggregate is a *pure function* of scores and weights —
which also makes the sensitivity analysis real rather than decorative.

For trust: if a model both scores a candidate and decides whether it wins, there
is no independent step anywhere in the pipeline. Splitting them means a model can
be wrong about a dimension without being able to promote a candidate that
violates a constraint.

**Cost.** The deterministic parts cannot exercise judgement. A hard constraint
that needs real interpretation comes back `UNKNOWN` rather than being resolved
cleverly. That is the intended trade.

---

## ADR-002: Simulated provider as a first-class, labelled path

**Decision.** `SimulatedProvider` sets `is_simulated = True`; agents branch on it
and take an explicit, typed local path. It is not a mock that parses rendered
prompts back into structure.

**Why.** The alternative — a mock LLM that reads a prompt string and returns a
plausible object — couples the offline path to prompt *formatting*, so a harmless
prompt edit breaks the offline demo in a way that looks like a logic bug. Branching
on a flag gives the simulated path the same typed inputs the real path gets, so
the two cannot silently diverge in what they are given.

**Consequence.** Every agent has two code paths and both must be maintained. We
accepted this because the offline path is what makes the whole suite runnable
with no credentials, and 140 tests depend on it.

**Honesty requirement.** Simulated content is labelled at the *data* level
(`Provenance.SIMULATED`, `is_simulated`), not only in prose, so the decision
engine can act on it (ADR-005) rather than relying on a reader noticing a
disclaimer.

---

## ADR-003: Missing evidence widens a range; it is never imputed

**Decision.** A dimension with insufficient evidence is `UNKNOWN` and carries no
number — the schema rejects a `raw` value on an `UNKNOWN` score. Aggregation
reports a point estimate over scored dimensions (weights renormalised), plus
`[score_lower, score_upper]` bounding the unknowns at 0 and 5, plus
`evidence_sufficiency` (the share of weight actually backed by a score).

**Why.** The tempting alternatives are both wrong. Imputing a neutral 2.5 invents
evidence. Multiplying by a "confidence factor" invents a second unvalidated
quantity and buries it inside the first — the spec explicitly warns against this,
and it is unfalsifiable: nobody can say whether 0.8 was the right penalty.

A wide band is self-explanatory: it says *we do not know enough to rank these*,
which is a conclusion rather than a hedge.

**Rejected.** Interval arithmetic throughout, Bayesian score distributions. Both
are better; both need calibration data this system does not yet have.

---

## ADR-004: Hard constraints are non-compensatory, and arithmetic where possible

**Decision.** Constraint status folds with fixed precedence —
`any FAIL -> FAIL`, `else any UNKNOWN -> UNKNOWN`, `else PASS` — and a `FAIL`
sets `eligible = False`, which excludes a candidate from Top-K and from selection
regardless of score. A constraint carrying `limit_value` is decided by
comparison, not judgement.

**Why.** Weighted sums are compensatory by construction: enough novelty will
eventually outweigh "exceeds the budget" or "is unsafe". For constraints that
genuinely disqualify, that arithmetic is simply the wrong model. Keeping the gate
outside the sum is the only way to guarantee it holds, and an unassessed
constraint becomes `UNKNOWN` rather than an implicit pass.

**Verified by** `test_constraint_fail_beats_a_perfect_score`: a candidate scoring
100.0 with a violation loses to one scoring 40.0 without.

---

## ADR-005: Simulated results are capped at R3 in code

**Decision.** `determine_maturity_level` returns at most `R3_PROTOTYPED` when
`experiment.is_simulated`, and `INCONCLUSIVE` when
`requires_human_input`. `goal_satisfied` is `True` only at `L5`, which requires
real execution, every contract metric measured and met, and all hard constraints
satisfied.

**Why.** This is the failure this project exists to prevent. A mock adapter can
produce beautiful numbers; without a structural cap, a run would report success
on synthetic data and the disclaimer would be a paragraph nobody reads. Encoding
the cap in the function that assigns the level means no prompt, phrasing or
reviewer attention is load-bearing.

**Also encoded:** unmeasured is distinct from failed. A metric with no
observation gets `met=None`, never `False` — "we did not check" and "it does not
work" are different findings and must not collapse.

---

## ADR-006: Two loops as two classes

**Decision.** `ImplementationRepairLoop` (Loop A) and `ResearchReplanningLoop`
(Loop B) are separate classes, triggered by different diagnoses, and they read
**disjoint test categories**: conformance reads `contract`/`correctness`/
`edge_robustness`; goal attainment reads only `goal_acceptance`.

**Why.** The spec asks for the distinction to be visible in code, and the
category split is what makes it more than naming. It means a candidate that is
built exactly right but misses the goal is *not* routed to a repair loop — it
produces a research finding. Conversely a build defect does not get mistaken for
a refuted hypothesis. Conflating these is how a harness ends up "fixing" a
correct implementation of a bad idea.

---

## ADR-007: The executor cannot amend the test plan

**Decision.** `TestPlanRegistry` hashes each plan; `Executor.run` verifies the
hash before *and* after a run; `TestPlanRegistry.amend` raises for any author in
`EXECUTION_ROLES`; `Executor.amend_plan` exists only to refuse. Amendments
produce a new version with a recorded reason, never a mutation.

**Why.** "Do not weaken the tests to pass" is the kind of rule that holds while
anyone is watching. Structural separation holds when nobody is. Plans are
authored from the *contract* before execution, so acceptance criteria restate the
contract's own thresholds rather than the implementation's behaviour.

---

## ADR-008: Pairwise comparison measures its own bias

**Decision.** Each unordered pair is judged twice with positions swapped, over
length-normalised, id-redacted proposal cards. Verdicts name a *slot*; mapping
back to candidates happens in code. `insufficient_evidence` is a first-class
outcome. Inconsistent pairs contribute 0.5/0.5 — no preference signal.

**Why.** Position bias and verbosity bias in LLM judges are well documented, and
an unmeasured bias is indistinguishable from a real preference. Swapping makes
bias *observable*: a judge that picks slot A both times names two different
winners, which is detectable. The rate is reported and, when non-zero, the run
warns that the pairwise ordering is unreliable rather than quietly using it.

`Proposal.card()` redacts the candidate's own id and author from the rendered
text, so a designer that works its identifier into a title cannot defeat the
anonymisation — a hole found by `test_proposal_card_hides_identity_and_authorship`.

---

## ADR-009: Domain knowledge in a data registry

**Decision.** All domain specifics live in `agents/domains.py` as frozen
`DomainProfile` records: keywords, goal type, default metrics and constraints,
preferred adapters, strategy archetypes, validation routes. The orchestrator,
scoring engine and state object contain no domain branching.

**Why.** The spec's hardest requirement is that two unrelated domains run on
*identical* core machinery. A registry makes that checkable — and adding a domain
means appending a record, not editing the state machine.

**Limit, stated plainly.** Profile selection is keyword matching, which is weak.
It therefore only sets a weight *prior* and conservative placeholder metrics, all
flagged `assumed=True`, and `--domain` overrides it. Nothing load-bearing rests
on getting the guess right.

---

## ADR-010: No invented third-party APIs

**Decision.** `JevJudgeAdapter` raises on construction with instructions for
implementing it properly. `AnthropicWebSearchTool` and `AnthropicProvider` are
written against documented API surfaces and are excluded from the default test
run (`requires_network`).

**Why.** A judge wired to a guessed request shape fails at run time or, worse,
mis-parses scores into something plausible. For a component whose entire job is
to be trustworthy, a loud absence beats a silent fabrication. The docstring
records the steps — read the repo's current interface and licence, map only what
it genuinely returns, leave the rest `UNKNOWN` — so the stub is a specification
rather than a gap.

Related: a preference probability from such a model is that model's judgement
distribution over the options shown. It is **not** the real-world probability of
success, and the code never converts one into the other.

---

## ADR-011: argparse, sqlite3, pydantic — and nothing else

**Decision.** Runtime dependencies are `pydantic` and `pyyaml`. CLI is
`argparse`; persistence is stdlib `sqlite3`; orchestration is a hand-written
state machine.

**Why.** The spec warns against stacking frameworks. At this size LangGraph buys
a persistence model we already need to own (resume is a core feature, so we want
the state object to *be* the contract), Typer buys decorators over ~15
subcommands, and SQLAlchemy buys migrations over nine tables with a JSON payload
column. Each would add a dependency and a layer of indirection between a failure
and its cause.

**Revisit when.** Parallel agent fan-out with partial failure recovery, or a
schema that outgrows hand-written DDL.

---

## ADR-012: Attempted-and-blocked is not the same as never-run

**Decision.** `determine_maturity_level` checks `requires_human_input` *before*
the "not executed" branch.

**Why — found during implementation.** The manual adapter sets
`executed=False, requires_human_input=True`. With the obvious ordering, the
"never executed" branch caught it first and the product example reported
`R2_DESIGNED`: *"a complete design and test plan exist"*. Technically true,
quietly wrong — it lost the fact that validation was attempted, blocked on
real-world access, and had produced a concrete external plan. It also let the run
burn its full iteration budget re-deciding the same thing.

With the precedence fixed, the same run reports `INCONCLUSIVE`, attaches the
external validation plan, and stops immediately with
`request_external_validation`. Recorded here because the bug was invisible in
unit tests of either branch alone — only the end-to-end product example showed
it, which is an argument for keeping cross-domain examples in the suite rather
than only unit tests.

---

## ADR-013: One sandboxed command runner

**Decision.** All subprocess execution goes through `tools/execution.py`:
argument lists only (never a shell string), an executable denylist, credential-
path argument patterns refused, working directory confined to a declared root, an
environment allowlist that strips API keys, a hard timeout and output caps.

**Why.** Retrieved web content and model output are untrusted data. With no shell
there is no string to inject into; stripping credentials from the child
environment means experiment code cannot read the keys the harness itself uses.

**Not a security boundary, and the README says so.** It runs code the user
pointed it at with the harness's own privileges. For genuinely untrusted code,
run the harness in a container or VM. Claiming more than that would be the same
category of error as reporting simulated results as measured.

---

## ADR-014: Rank on the pessimistic bound, not the point estimate

**Decision.** `order_of` ranks by `score_lower` (every UNKNOWN dimension bounded
at its worst value), with `point_score` as a tiebreak. `point_score` remains the
headline number in reports but is no longer the comparison key.

**Why — found by the benchmark, not by review.** ADR-003 excludes UNKNOWN
dimensions from the point estimate and renormalises the remaining weights. That
is right for *describing* one candidate and wrong for *comparing* two: if the
missing dimension would have scored below a candidate's own average, dropping it
**raises** its point score. So a candidate citing no evidence at all outranked
one citing a cross-checked source — the harness rewarded having no evidence.

The `evidence_grounding` benchmark case caught this on its first run
(pairwise agreement 50%, rank correlation 0.0). It had been invisible to 140
passing tests because every test compared candidates with *equal* evidence
coverage, where the renormalisation cancels out.

`score_lower` is order-equivalent to `point_score` whenever two candidates have
the same dimensions scored, so the fix costs nothing in the common case. Where
coverage differs, a gap in the evidence can now only hurt. Under-evidenced
candidates rank low but are still retained, which is what the spec asks for:
keep them, and prioritise verifying them.

**Pinned by** `test_missing_evidence_can_never_improve_a_ranking`, which asserts
the perverse point-score relationship still exists *and* that it no longer
decides the ranking.

---

## ADR-015: Benchmark ground truth by construction, not by labelling

**Decision.** The judge-quality benchmark builds cases where the correct answer
follows from the definition: if A is at least as good as B on every rubric
dimension and strictly better on some, A dominates B, and preferring B is wrong
under any non-negative weighting.

**Why.** The spec rules out every convenient source of truth — the judge
labelling its own, multi-agent votes, model preference read as probability,
outcomes leaking into the prior baseline. Dominance sidesteps all of them: no
human labels, no outcome data, and the answer is not a matter of opinion.

**Stated limit, in the module docstring and in the report itself.** Dominance
cases are the *easy* ones. They detect a judge that is position-biased,
insensitive to the attributes it claims to score, non-invariant under
relabelling, or willing to promote a constraint violator. They say nothing about
whether a judge has good taste on genuinely contested trade-offs, which is where
expertise lives. A judge scoring 1.0 here has merely failed to be obviously
broken, and the report says so rather than letting the number imply more.

The benchmark exits non-zero on failure so it works as a CI gate, and it is
itself guarded: `test_dominance_case_is_actually_dominant_under_the_rubric`
checks the case's own premise, so the benchmark cannot silently stop testing
what it claims to test.

---

## ADR-016: Calibration reports discrimination, never a probability

**Decision.** `calibration.py` asks only whether higher-scored candidates have
in fact achieved their goals more often, and answers with an AUC
(Mann-Whitney). It excludes simulated runs entirely, excludes levels carrying no
outcome information, and reports **nothing** below `MIN_LABELLED_OUTCOMES` real
labels. Nothing feeds back into scoring automatically.

**Why.** Three failure modes, all cheap to fall into:

- Reading 73/100 as "73% likely to work" is the error the spec names explicitly.
  AUC answers the useful question — does the order mean anything? — without
  implying the score is a likelihood.
- Calibrating against mock experiment results would produce a confident number
  describing nothing. Simulated runs are filtered at the query.
- An AUC on three runs is noise with a decimal point, so below the minimum the
  report states the shortfall and what would fix it instead of printing a figure.

Auto-tuning weights on a handful of outcomes would overfit and quietly corrupt
the baseline, so the report is a diagnostic for a human and says so.

**One statistic removed during implementation.** An earlier version also printed
a Kendall tau between score order and outcome order. On a real 12-run sample
where every score was identical it reported **0.67** next to an AUC of **0.50** —
the tau had to break ties somehow, and breaking them by score manufactured
correlation out of nothing. Two statistics answering one question, with the
flattering one wrong, is worse than one correct statistic: the tau is gone.

---

## ADR-017: Half-up rounding on anchored scales

**Decision.** `_clamp` uses `floor(v + 0.5)` rather than `round()`.

**Why.** Python rounds halves to even, so `round(4.5)` is 4 while `round(3.5)`
is also 4. On a 0–5 anchored scale that makes the boundary between two anchors
depend on the parity of the lower one — indefensible when the number is a
judgement about quality against written anchor descriptions. Surfaced while
investigating ADR-014: excellent cross-checked evidence was scoring 4/5 because
its computed 4.5 rounded down.

---

## ADR-018: Follow-up retrieval is aimed by value of information

**Decision.** When the research loop decides to gather more evidence, it searches
for the single unresolved assumption that the value-of-information shortlist
ranks highest (predicted impact divided by cost), via
`Researcher.gather_targeted`. Both confirming and disconfirming query forms are
issued.

**Why.** The previous implementation re-ran the original broad query set with a
larger limit, which returns substantially the same documents and burns an
iteration for no new information — the loop looked like it was working while
making no progress. Targeting the specific open question is the only version of
"gather more evidence" that can change a decision. Issuing only confirming
queries would not be a test of the assumption, so the disconfirming forms are
not optional.

---

# v2: Adaptive routing

The v2 spec changed one thing structurally and several things in detail. These
records cover the restructure and the bugs it surfaced.

## ADR-019: The router replaces the pipeline, and is the only decision point

**Decision.** `core/orchestrator.py` no longer walks a phase sequence. Each
iteration it builds a `RouterContext` of explicit signals, asks
`AdaptiveRouter` for one action, executes it, records what it achieved, and
loops. `core/planner.py` (the phase table) is deleted.

**Why.** The v2 spec forbids treating `Researcher → Designer → Critic → Judge →
Executor` as the path every goal must take. A fixed pipeline spends the same
effort on a goal with one obvious answer as on a contested one, and it cannot
express "this question is settled, stop" or "this needs a person".

**The harder half of the decision: there is exactly one decider.** The previous
version had `DiagnosisAgent` choosing an `Action` *and* the orchestrator mapping
phases. Adding a router on top would have made three. So `DiagnosisAgent` was
narrowed to what it is actually good at — attributing a failure to a cause — and
the router reads that cause as a signal. Two components choosing the next step
is how a control flow stops being traceable.

**Rules decide; the model advises.** The spec rules out routing on an opaque
model opinion. Every decision comes from a named, ordered rule over explicit
signals, and a model suggestion is recorded alongside — including when it
disagreed and was overridden (`RoutingDecision.overrode_model`). Rule order
encodes precedence: safety and authorisation, then budget, then "are we done",
then blocked-on-reality, then stagnation, then cheap information, then strategy.

**Honesty about the heuristic.** The rule table is a baseline, not an optimum.
Each decision carries `predicted_cost_units` and gets `realised_new_evidence` /
`realised_maturity_gain` attached afterwards, so it can eventually be assessed
rather than trusted.

---

## ADR-020: Mode gating makes Goal-Only a structural baseline

**Decision.** `MODE_ALLOWED` restricts which actions each mode may take.
`goal_only` *cannot* reach `SPAWN_SUBAGENT` or `RUN_PROPOSAL_TOURNAMENT` at all;
a rule that selects one is downgraded to `CONTINUE_SINGLE_AGENT`, and the
downgrade is recorded (`decided_by="mode"`) rather than being silent.

**Why.** The spec's methodological demand is a baseline that richer modes must
justify themselves against. A baseline implemented as "the full system with a
flag" is not a baseline — any leak in the gate makes the comparison meaningless.
Structural restriction is checkable, and
`test_baseline_arm_generates_exactly_one_candidate` checks it.

**What the gate must not swallow.** A downgrade is a reduction in machinery, not
in safety: `REQUEST_HUMAN` and `STOP` are permitted in every mode, and
`test_mode_gating_never_blocks_safety_or_stopping` pins that.

---

## ADR-021: `Phase` shrank from fifteen pipeline stages to five lifecycle states

**Decision.** `INIT`, `RUNNING`, `AWAITING_APPROVAL`, `DONE`, `STOPPED`.

**Why.** Once the router decides each step, "being in the EVALUATION phase" is
not a fact about the run — scoring can happen at several points and may be
redone when new evidence arrives. Keeping the old enum would have left thirteen
states nothing could ever be in, which is the sort of dead vocabulary that
misleads the next reader. What remains is exactly the set a run can be *resumed*
from.

---

## ADR-022: Side effects are guarded by idempotency keys, not by position

**Decision.** `EventLog` records an `idempotency_key` for every side-effecting
action; `guard()` skips an action whose key has already completed, and logs the
skip. `EventLog.restore()` rebuilds the key set when a run is resumed.

**Why.** The spec requires that resuming must not repeat a completed action with
side effects. In a pipeline you could resume "at the next phase"; with a router
there is no position to resume at, so safety has to come from the action's own
identity. The key is derived from *what the action is about* (run, candidate,
plan hash, index) and never from a timestamp or counter — those differ on
replay, which is precisely when the check must match.

Without `restore()` rebuilding the key set, a resumed run would believe no side
effect had happened yet. `test_restore_rebuilds_the_key_set_so_resume_stays_safe`
pins that specific hole.

---

## ADR-023: Layered TOML, and secrets that cannot reach an artifact

**Decision.** `config/global_config/*.toml` deep-merged, then
`config/tasks/<task>.toml`, then explicit CLI flags. Arrays *replace* rather
than concatenate. Unknown sections and keys are errors. The single
`configs/*.yaml` file and `src/research_harness/config.py` are deleted — not
kept alongside, since two config systems is exactly the redundancy worth
avoiding.

**Why arrays replace.** Silent appending to an inherited list is behaviour
nobody predicts correctly; a task overlay that sets `require_approval = []`
should mean "nothing is gated here", not "keep whatever the global file said".

**Redaction by key shape, not by list.** `redact()` matches names containing
`key`/`token`/`secret`/`password`/`credential`/`auth`, so a *new* secret field is
redacted by default rather than needing to be remembered. The one carve-out is
names ending `_env`, which hold the *name* of an environment variable
(`api_key_env = "ANTHROPIC_API_KEY"`) and are safe — and useful — to show. Every
run writes `redacted_effective_config.json`, and
`test_effective_config_snapshot_carries_no_secret_values` asserts no `sk-ant`
string survives into it.

Weight priors deliberately stayed in code rather than moving into
`evaluation.toml`: the 0–5 anchors define what a score *means*, and letting a
config file change that would make scores incomparable between runs. Only
*emphasis* is configurable, per-run, via `weights --set`.

---

## ADR-024: Pareto is reported beside the ranking, never instead of it

**Decision.** `pareto_frontier()` reports the non-dominated set; the weighted
ranking remains the primary output. Dominance requires at least two *comparable*
dimensions, and a dimension scored `UNKNOWN` on either side cannot participate.

**Why.** A weighted sum always permits a trade-off. When dimensions genuinely
cannot trade, the honest answer is a set of incomparable alternatives — but
silently replacing the ranking with a set would break every caller and hide the
weights' role. Reporting both makes the weights' contribution visible: when the
frontier's first entry differs from the ranking's leader, the interpretation
says so in as many words.

**Why UNKNOWN cannot dominate.** "We did not measure it" is not a value.
Treating it as one would manufacture dominance out of missing evidence, which is
the same error ADR-003 and ADR-014 exist to prevent.

---

## ADR-025: The test plan is authored at selection, not at execution

**Decision.** `_ensure_test_plan()` runs as soon as a candidate is selected, not
when an experiment is about to start.

**Why — found by a failing test.** With the plan created inside the experiment
handler, an `R2_DESIGNED` run (design and plan, execute nothing) could never
satisfy its own exit condition: the depth check wanted a plan, and only the
experiment produced one, so the run ran the experiment it had been asked not to.
Authoring at selection fixes that *and* makes "test-first" structural rather
than conventional — a plan provably exists, hashed and registered, before any
experiment is routed.

---

## ADR-026: Four bugs the restructure surfaced

Recorded because each was invisible to the test suite that existed at the time,
and three were found by *running the demo* rather than by review.

**1. Evidence exhaustion conflated with lack of progress.**
`evidence_sources_exhausted` was derived from `consecutive_no_gain`. A repair
attempt that made progress reset that counter, which re-opened retrieval that
had already been shown to have nothing left — so the run re-gathered, found
nothing, and burned iterations. They are different facts and now have different
fields (`ResearchState.evidence_exhausted`).

**2. Simulated evidence counted as progress.** A fixture-only run gathers items
but `real_evidence_count` stays zero, so the `critical_evidence_missing` rule
fired forever. A gather that adds no *non-simulated* evidence now counts as no
gain and marks retrieval exhausted. Simulated items are not progress toward
grounding a claim.

**3. The general agent could spin forever.** Re-running it with nothing new to
contribute looked like work and produced none — exactly the pointless self-loop
§11.2 warns about. It now snapshots `(evidence, proposals, evaluations, unknowns,
selection)` and records a no-gain step when nothing moved, so the stagnation rule
can fire. All three modes now stop with `STAGNATED` instead of exhausting the
budget.

**4. A gated stage ended the run instead of pausing it.** Both the router
(`REQUEST_HUMAN`) and the executor (`ApprovalRequired`) guarded approval, and the
router got there first — reporting a merely-paused run as finished and losing
its resumability. `_act_request_human` now distinguishes a *pause* (operator can
approve and continue) from *needing real-world validation* (the run ends), and
`resume()` refuses to proceed past an unapproved gate, so resume is not a way
around it. The executor's guard is kept deliberately: for an irreversible-action
gate, a second check is worth its cost.

---

## ADR-027: The general agent reports signals; it does not decide

**Decision.** `GeneralResearchAgent.solve()` returns findings, hypotheses, a
problem tree, counterexamples, `critical_unknowns`, `complexity_score`, and an
explicitly *advisory* `suggested_next_action`. Nothing it returns tells the
orchestrator what to do.

**Why.** An agent that both assesses its own uncertainty and decides whether to
continue will tend to conclude it is nearly finished. Forcing the uncertainty
into structured fields that a rule table reads keeps the assessment and the
decision in different hands — and makes over-claiming confidence visibly
degrade the routing, instead of quietly ending the run early.

In the offline path its signals are derived by *counting* contract structure
(metrics without thresholds, unaddressed constraints, evidence provenance)
rather than by self-assessment, so the router's rules get comparable numbers.

---

## ADR-028: The ablation refuses to name a winner

**Decision.** `core/ablation.py` runs one idea under several modes with the same
seed, budget and tools, and reports what each cost and reached. Its `findings`
are factual observations; `AblationReport.caveat` states that one run per mode
cannot establish that a richer mode is better.

**Why.** The spec says not to claim multi-agent superiority from a single demo,
and the first real output said *"on this instance the extra machinery bought no
measurable gain"* — which is the kind of result a system under pressure to look
impressive would quietly round up. It is also the one most worth printing.

The report flags its own failure modes too: a `goal_only` arm that produced more
than one candidate is reported as a possible mode-gate leak rather than treated
as a result.

---

## ADR-029: An OpenAI-compatible provider, tiered because endpoints disagree

**Decision.** `providers/openai_compatible.py` speaks the `/v1/chat/completions`
shape over stdlib `urllib`, with structured output attempted in three tiers:
`json_schema` (strict) → `json_object` → schema-in-prompt. The working tier is
discovered once and remembered.

**Why tiers, and why this shape was not guessed.** Probing a live endpoint
first — rather than writing against what the docs imply — produced the decisive
fact: it **rejects `json_schema` unless every property appears in `required`,
and enforces that even when `strict: false`.** Pydantic emits only
genuinely-required fields, so every real agent schema 400s. Guessing would have
produced a provider that worked on a toy flat schema and failed on everything
the harness actually sends.

So `to_strict_schema()` rewrites the schema: every property required, every
originally-optional property made nullable, `additionalProperties: false`. A
returned `null` then means "nothing to say", and `strip_nulls()` removes those
before validation so the Pydantic default applies — a `list[str]` field would
reject `None` outright.

**Why stdlib `urllib`.** One JSON POST does not justify a dependency (ADR-011),
and these endpoints vary enough that a vendor SDK's assumptions get in the way.

**Cost is reported as 0.0, not estimated.** Per-token prices for an arbitrary
proxy are unknown; token counts are tracked (including `reasoning_tokens`), and
an invented dollar figure would be worse than an absent one.

---

## ADR-030: A real model exposed two bugs the simulated path could not

Both were invisible to 293 passing tests, because the simulated provider always
produced valid enum values.

**1. Free-text fields that had to be closed sets.** `GoalAnalysis.goal_type`,
`domain_profile`, and `origin` were plain strings. A real model answered
`origin="derived from user idea"`, `goal_type="Performance optimization under a
cost constraint"`, `domain_profile="Read-only API performance optimization…"`.

The `_origin()` fallback defaulted unrecognised values to `ASSUMED` — safe in
direction, but it meant **every** metric was marked as a harness guess, silently
destroying the `user_specified` vs `assumed` distinction that §3 exists to
preserve. `goal_type` and `domain_profile` fell through to defaults, so the
weight prior and domain profile were quietly wrong.

Fix: make the schema do the work. `GoalType` and a registry-generated
`DomainName` enum, plus `Origin` on the DTOs, so structured output *cannot*
return an invalid value. The fuzzy fallback remains only for the weaker tiers.
Re-run with the real model: `origin=user_specified` on the two things the
requester actually stated, `goal_type=TECHNICAL_PERFORMANCE`,
`domain_profile=SOFTWARE_SYSTEM`.

**2. Metric names are emergent, so experiment binding needs stating.** A real
goal analyst names metrics from the goal (`spmv_execution_time`), not from the
domain template (`speedup_vs_baseline`) — so `--set-metric speedup_vs_baseline=5`
failed with "unknown metric key", and a real test suite's reported key no longer
matches any contract metric, which would make every metric read as "never
measured".

Fix: `--map-metric CONTRACT_KEY=OBSERVED_KEY`, recorded on the experiment
result. Deliberately *not* fuzzy name matching: deciding that a measured
quantity stands for a goal metric is a research judgement, and a silent
heuristic guess there is how an unrelated number ends up certifying a goal.

**Why this belongs in an ADR.** The lesson generalises past these two bugs: a
deterministic stand-in agrees with its own schema by construction, so it cannot
test whether the schema actually constrains anything. Offline fixtures verify
the pipeline; only a real model tests the contract with the model.

---

## ADR-031: Four more bugs, all found by running a real model

Recorded separately from ADR-030 because these came from *running* the harness
on a real endpoint rather than from inspecting its first response. None was
reachable with the simulated provider.

**1. A stale `--map-metric` bound nothing, silently.** The alias was recorded
and matched no observation, so the run finished reporting "no contract metric
was measured" — true, but it reads as the experiment's fault rather than the
mapping's. Now an unknown contract key is a hard error that lists the
contract's actual metric keys.

**2. Metric names are non-deterministic across runs.** Four real runs on the
same idea produced four namings (`spmv_execution_time` / `spmv_throughput` /
`spmv_performance` / `sparse_to_dense_memory_ratio` / `memory_ratio_to_dense` /
`peak_memory_footprint` / `sparse_memory_footprint`). This makes real
measurement a dead end: you must read the contract to learn the keys, but
starting a fresh run regenerates them.

`resume <id> --depth` deepens an *existing* run on its already-fixed, versioned
contract, so keys read a moment ago stay valid. Lowering the requested maturity
is refused: asking for less verification after seeing results is how a goal gets
quietly weakened.

**3. A config default hijacked an action-triggering flag.** Adding `--depth` to
`resume` meant `target.maturity_level` in `config/global_config/loop.toml`
landed in it and **silently reopened finished runs**. Options that set a *value*
may come from config; options that trigger an *action* must be typed. The CLI
now records `_explicit_flags` and the deepen branch requires membership. Caught
by the existing approval-gate test, which is the argument for keeping
end-to-end tests that assert on exit codes.

**4. "Measured but undecidable" was reported as "never measured".** Real models
frequently declare a metric with no numeric target, and both cases collapsed
into one message — blaming the experiment for a gap in the *contract* and
sending the reader to fix the wrong thing. Now three states: `decided`
(observed + threshold), `undecidable` (observed, no threshold — "a gap in the
contract, not in the experiment: supply a target and re-judge"), and
`unobserved`.

---

## ADR-032: Provenance is re-recorded on every resume, and warnings are derived

**Decision.** `_refresh_provenance()` runs each time the router loop is
entered and notes any backend change. The "experiments ran on a SIMULATED
adapter" warning is no longer appended at `start()`; it is *derived at report
time* from the experiments that actually ran, and a run with both kinds gets an
explicit mixed-provenance warning.

**Why.** Provenance was captured once at `start()`. A real two-phase run —
research at R1 with the defaults, then `resume --depth R5` with a real model,
real judge and the pytest adapter — reported `Judge: mock`,
`Adapter: simulated`, and warned about a simulated adapter, while the pytest
adapter was in fact producing the measurement. Wrong in the conservative
direction that time, but it is the same bug in both directions and the inverse —
claiming a real measurement for a simulated one — is the single worst thing this
harness could do.

Deriving the warning also removes a subtler trap: a warning *persisted into run
state* becomes a claim that outlives the condition that justified it. State now
holds only facts about what happened; anything that is an interpretation of
those facts is computed when the report is written.

---

## ADR-033: Re-scoring must move the selection, or its result is thrown away

**Decision.** `_score()` re-selects the leading eligible candidate whenever the
currently selected one has not yet been experimented on. A candidate that *has*
been tested is kept; retiring it is the router's decision via
`tried_proposal_ids`.

**Why — observed in a real run.** The sequence was
`RUN_PROPOSAL_TOURNAMENT -> SPAWN_SUBAGENT -> RUN_EXPERIMENT`. The tournament
picked a leader; the skeptic review then re-scored, which reordered the ranking;
but `_score` only assigned the selection when it was `None`, so the stale pick
survived. The run experimented on the **rank-3** candidate while the report
displayed rank 1 as the leader, with `tried_proposal_ids` empty — nothing had
retired the leader, the selection had simply gone stale.

Re-scoring exists *because* the ranking can move. Keeping the previous pick
discards the entire result of the step that was just performed, and produces a
report whose "Selected" line contradicts its own ranking table.

The already-tested exception matters too: swapping away from a candidate after
spending a real experiment on it would discard real work on a ranking change
that may itself be noise. A selection change is written into the notes, because
silently experimenting on a different candidate than the report shows is the
failure this fixes.

---

## ADR-034: The position-bias detector earned its place on a real judge

Not a decision — a measurement, recorded because it is the evidence for ADR-008.

Two real runs with the same model as judge, on the same three candidates:

| Run | Order-consistency | Position bias | Behaviour |
|---|---|---|---|
| A | 100% | 0% | agreed with the `score_lower` ranking unprompted |
| B | 33% | 33% | picked slot A twice on one pair, naming different winners |

In run B the judge also returned *identical* rubric scores for two
substantively different candidates. The harness handled it as designed: the
biased pair contributed no preference signal, the run warned that "the pairwise
ordering is unreliable and is used only as a tie-break", and the deterministic
ranking carried the decision.

Run A is the more interesting result for ADR-014. The judge independently
preferred the candidate that `score_lower` ranked first — which had the *lower*
point score (60.98 vs 65.37) and won only because it was the single
fully-evidenced candidate. An independent real judge agreeing with the
pessimistic-bound ranking is evidence that the fix tracks something real, not
that it is a convenient tiebreak.

Run B is the argument for measuring bias at all. The same model, same prompts,
same candidates, one run later, was unreliable enough that using its ordering
would have been worse than ignoring it — and nothing in its *output* said so.
Only the A/B swap revealed it.
