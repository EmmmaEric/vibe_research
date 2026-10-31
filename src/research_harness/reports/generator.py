"""Report generation (read-only over the run state).

``final_report.md`` carries the ten sections the spec requires, in order. Two
rules govern the wording throughout:

* The conclusion level is stated plainly, with the reason it is not higher. A
  simulated run says so in its first lines.
* No claim of global optimality -- preferences are always scoped to the
  candidates considered, the evidence gathered, and the weights in force.
"""

from __future__ import annotations

from ..core.schemas import (
    Base,
    ConstraintStatus,
    MaturityLevel,
    Mode,
    Provenance,
    RunOutcome,
    Stance,
)
from ..core.state import ResearchState

_LEVEL_MEANING = {
    MaturityLevel.R1_RESEARCHED: "sourced research completed",
    MaturityLevel.R2_DESIGNED: "a complete design and test plan exist; nothing executed",
    MaturityLevel.R3_PROTOTYPED: "a proof of concept or minimal experiment ran",
    MaturityLevel.R4_VALIDATED: "independently and repeatably validated",
    MaturityLevel.R5_GOAL_ACHIEVED: "the original contract goal was met",
    MaturityLevel.INCONCLUSIVE: "evidence insufficient to assert success or failure",
    MaturityLevel.REJECTED: "unsuitable, or violates a hard constraint",
}


def _fmt(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


class RunSummary(Base):
    """Machine-readable summary of a run (spec section 18).

    Every field is either observed or explicitly null. There are no placeholder
    numbers: a run that measured nothing reports nulls and a reason, which is
    what makes the file safe to aggregate across runs.
    """

    run_id: str
    mode: Mode
    idea: str
    maturity_level: MaturityLevel
    outcome: RunOutcome | None = None
    stop_reason: str | None = None
    stop_detail: str | None = None
    goal_satisfied: bool | None = None
    implementation_conforms: bool | None = None
    selected_proposal: str | None = None
    proposal_count: int = 0
    excluded_for_constraints: list[str] = []
    real_evidence_count: int = 0
    simulated_evidence_count: int = 0
    critical_unknowns: list[str] = []
    iterations: int = 0
    routing_actions: list[str] = []
    experiments_run: int = 0
    used_simulated_provider: bool = True
    used_simulated_experiments: bool = False
    tokens_used: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    llm_calls: int = 0
    top1_stable: bool | None = None
    position_bias_rate: float | None = None
    evidence_gaps: list[str] = []
    external_actions_required: list[str] = []
    warnings: list[str] = []


class ReportGenerator:
    """Renders Markdown from a :class:`ResearchState`. Never mutates it."""

    # ==================================================================

    def final_report(self, state: ResearchState, *, store=None) -> str:
        c = state.contract
        L: list[str] = []
        level = state.final_maturity_level or MaturityLevel.INCONCLUSIVE

        L.append(f"# Research Report -- {state.run_id}")
        L.append("")
        L.append(f"**Conclusion level: `{level.value}`** -- {_LEVEL_MEANING[level]}.")
        L.append("")

        if state.provenance.is_simulated_provider or any(
            e.is_simulated for e in state.experiments
        ):
            L += [
                "> ### This run contains simulated content",
                ">",
                "> Items below marked `[SIMULATED]` were produced by deterministic "
                "local heuristics or a mock experiment adapter, **not** by reasoning "
                "about the problem or by measuring anything. They demonstrate that the "
                "pipeline runs end to end. They are not research evidence, and no "
                "conclusion about the real world follows from them.",
                "",
            ]

        # -- 1. Original idea and success criteria -----------------------
        L += ["## 1. Original idea and success criteria", ""]
        if c is None:
            L += ["No contract was established.", ""]
        else:
            L += [
                "### Original idea (verbatim, immutable)",
                "",
                "> " + c.original_idea.replace("\n", "\n> "),
                "",
                f"**Problem statement (system-derived):** {c.problem_statement}",
                "",
                f"- Goal type: `{c.goal_type}` | Domain profile: `{c.domain_hint}`",
                f"- Contract version: {c.version} "
                f"({len(c.revisions)} revision(s) recorded)",
                f"- Validation depth requested: `{c.desired_maturity_level.value}`",
                "",
                "### Success metrics",
                "",
                "| Metric | Target | Unit | Direction | Measurement | Stated by |",
                "|---|---|---|---|---|---|",
            ]
            for m in c.success_metrics:
                L.append(
                    f"| `{m.key}` | {_fmt(m.target_threshold)} | {m.unit} | "
                    f"{m.direction.value} | {m.measurement_method} | "
                    f"{'ASSUMED by harness' if m.assumed else 'requester'} |"
                )
            L.append("")
            L += ["### Hard constraints (non-compensatory)", ""]
            if c.hard_constraints:
                for hc in c.hard_constraints:
                    checkable = (
                        f" *(machine-checkable: {hc.limit_metric} "
                        f"{hc.limit_direction.value} {hc.limit_value})*"
                        if hc.limit_value is not None
                        else ""
                    )
                    L.append(
                        f"- **{hc.key}** ({hc.category}): {hc.description}"
                        f"{' [ASSUMED]' if hc.assumed else ''}{checkable}"
                    )
            else:
                L.append("- none declared")
            L.append("")
            if c.assumptions:
                L += ["### Assumptions made by the harness", ""]
                L += [f"- {a}" for a in c.assumptions] + [""]
            if c.unknowns:
                L += ["### Unknowns at contract time", ""]
                L += [f"- {u}" for u in c.unknowns] + [""]

        # -- 2. Evidence -------------------------------------------------
        L += ["## 2. Key evidence, sources and counter-evidence", ""]
        real = [e for e in state.evidence if e.is_real]
        sim = [e for e in state.evidence if not e.is_real]
        L.append(
            f"{len(state.evidence)} evidence item(s): **{len(real)} real**, "
            f"{len(sim)} simulated."
        )
        L.append("")
        if not real:
            L += [
                "> **No real evidence was gathered.** Every evidence-dependent score "
                "in this report is `UNKNOWN`, and no conclusion here is empirically "
                "grounded.",
                "",
            ]
        for label, stance in (
            ("Supporting", Stance.SUPPORTS),
            ("Refuting / counter-evidence", Stance.REFUTES),
            ("Uncertain", Stance.UNCERTAIN),
        ):
            group = [e for e in state.evidence if e.stance is stance]
            if not group:
                continue
            L += [f"### {label} ({len(group)})", ""]
            for e in group:
                tag = "" if e.is_real else " `[SIMULATED]`"
                uri = f" <{e.source_uri}>" if e.source_uri else ""
                L.append(
                    f"- **{e.evidence_id}**{tag} [{e.source_type}, "
                    f"{e.verification_status}, quality {e.quality}/5] {e.claim}{uri}"
                )
                if e.applicability_conditions:
                    L.append(
                        f"  - Applies when: {'; '.join(e.applicability_conditions)}"
                    )
                if e.limitations:
                    L.append(f"  - Limitations: {'; '.join(e.limitations)}")
            L.append("")
        if not [e for e in state.evidence if e.stance is Stance.REFUTES]:
            L += [
                "**No counter-evidence was found.** Disconfirming queries were issued, "
                "so this may indicate a search too narrow rather than an absence of "
                "objections.",
                "",
            ]

        # -- 3. Candidates ----------------------------------------------
        L += ["## 3. Candidates: strengths, limits and risks", ""]
        for p in state.proposals:
            ev = state.evaluation(p.proposal_id)
            crit = state.critique_for(p.proposal_id)
            L += [
                f"### {p.proposal_id} -- {p.title}",
                "",
                f"- Archetype: `{p.strategy_archetype}` | declared cost: "
                f"{p.declared_cost_units} unit(s) | author: `{p.generated_by}`",
                f"- **Mechanism:** {p.core_mechanism}",
                f"- **Maps to goal by:** {p.goal_mapping}",
            ]
            if p.differentiators:
                L.append(f"- **Distinctive:** {'; '.join(p.differentiators)}")
            if p.dependencies:
                L.append(f"- **Depends on:** {'; '.join(p.dependencies)}")
            if p.failure_modes:
                L.append(f"- **Failure modes:** {'; '.join(p.failure_modes)}")
            if p.critical_assumptions:
                L.append(
                    f"- **Must be true (unverified):** {'; '.join(p.critical_assumptions)}"
                )
            L.append(
                f"- **Minimal PoC:** {p.poc.description} "
                f"*Refuted if:* {p.poc.falsification_condition}"
            )
            if p.evidence_ids:
                L.append(f"- **Cites:** {', '.join(p.evidence_ids)}")
            else:
                L.append("- **Cites:** no evidence (premises are unsupported assertions)")
            if ev:
                L.append(
                    f"- **Constraint status:** `{ev.constraint_status.value}`"
                    + (
                        f" -- violates {', '.join(ev.violated_constraints)}"
                        if ev.violated_constraints
                        else ""
                    )
                    + (
                        f" -- unverified: {', '.join(ev.unverified_constraints)}"
                        if ev.unverified_constraints
                        else ""
                    )
                )
            if crit and (crit.unstated_dependencies or crit.hidden_costs or crit.counterexamples):
                L.append(f"- **Critic (severity {crit.severity}/5):**")
                for item in (
                    crit.counterexamples + crit.unstated_dependencies + crit.hidden_costs
                )[:6]:
                    L.append(f"  - {item}")
            L.append("")

        # -- 4. Scoring --------------------------------------------------
        L += ["## 4. Scoring dimensions, weights, ranking and uncertainty", ""]
        L.append(self._scoring_section(state))

        # -- 5. Selection ------------------------------------------------
        L += ["## 5. Selected candidate and why", ""]
        if state.selected_proposal_id and state.ranking:
            from ..evaluation.decision_engine import scope_qualified_claim

            pid = state.selected_proposal_id
            L += [scope_qualified_claim(state.ranking, pid), ""]
            ev = state.evaluation(pid)
            if ev:
                L += [
                    f"- Score {ev.point_score:.1f}/100 "
                    f"(range {ev.score_lower:.1f}-{ev.score_upper:.1f} once unknowns "
                    f"are bounded); evidence sufficiency "
                    f"{ev.evidence_sufficiency:.0%} of total weight",
                    f"- Constraint status: `{ev.constraint_status.value}`",
                ]
                if ev.unresolved_assumptions:
                    L.append("- Unresolved assumptions carried into testing:")
                    L += [f"  - {a}" for a in ev.unresolved_assumptions]
                L.append("")
        else:
            L += ["No candidate was selected.", ""]
        if state.ranking and state.ranking.excluded_for_constraints:
            L += [
                f"**Excluded by hard constraint (could not be offset by any score): "
                f"{', '.join(state.ranking.excluded_for_constraints)}.**",
                "",
            ]

        # -- 6. Tests and raw results -----------------------------------
        L += ["## 6. Test plan and raw experimental results", ""]
        L.append(self._tests_section(state))

        # -- 7. Conformance vs goal -------------------------------------
        L += ["## 7. Did the implementation conform, and was the goal met?", ""]
        v = state.latest_validation()
        if v is None:
            L += ["No validation was performed.", ""]
        else:
            L += [
                "These are two separate questions, answered by disjoint test "
                "categories:",
                "",
                f"- **Loop A -- implementation conforms to the candidate's spec:** "
                f"`{_fmt(v.implementation_conforms)}`",
                f"  - {v.implementation_notes}",
                f"- **Loop B -- original contract goal met:** `{_fmt(v.goal_satisfied)}`",
                f"  - {v.blocking_reason}",
                "",
                "| Contract metric | Target | Observed | Met | Provenance |",
                "|---|---|---|---|---|",
            ]
            for mv in v.metric_verdicts:
                L.append(
                    f"| `{mv.metric_key}` | {mv.direction.value} "
                    f"{_fmt(mv.target_threshold)} | {_fmt(mv.observed)} | "
                    f"{_fmt(mv.met)} | {mv.provenance.value}"
                    f"{' **(not real)**' if mv.provenance is Provenance.SIMULATED else ''} |"
                )
            L.append("")

        # -- 8. Gaps and next actions -----------------------------------
        L += ["## 8. Failed assumptions, missing evidence, risks and next actions", ""]
        if v and v.evidence_gaps:
            L += ["### Evidence gaps", ""] + [f"- {g}" for g in v.evidence_gaps] + [""]
        if v and v.external_validation_plan:
            L += [
                "### Requires real-world validation (not performed here)",
                "",
            ] + [f"- {s}" for s in v.external_validation_plan] + [""]
        warnings = self.all_warnings(state)
        if warnings:
            L += ["### Warnings", ""] + [f"- {w}" for w in warnings] + [""]
        L += ["### Decision history", ""]
        if state.decisions:
            for d in state.decisions:
                L += [
                    f"- **{d.decision_id}** (iteration {d.iteration}, loop `{d.loop}`) "
                    f"action `{d.action}`",
                    f"  - Hypothesis: {d.hypothesis}",
                    f"  - Expected: {d.expected_result}",
                    f"  - Actual: {d.actual_result}",
                ]
                if d.divergence_cause:
                    L.append(f"  - Divergence cause: {d.divergence_cause}")
        else:
            L.append("- none recorded")
        L.append("")
        L += ["### Recommended next actions", ""]
        L += [f"- {s}" for s in self._next_actions(state)] + [""]

        # -- 9. Reproduction --------------------------------------------
        L += ["## 9. Reproduction", ""]
        L.append(self._repro_section(state, store))

        # -- 10. Conclusion level ---------------------------------------
        L += ["## 10. Conclusion level", ""]
        L += [
            f"**`{level.value}`** -- {_LEVEL_MEANING[level]}.",
            "",
        ]
        if v and v.blocking_reason:
            L += [f"Why not higher: {v.blocking_reason}", ""]
        if state.stop_decision:
            L += [
                f"**Run stopped:** `{state.stop_decision.reason}` -- "
                f"{state.stop_decision.detail}",
                "",
            ]
        if c:
            L += [
                f"**Budget at stop:** {state.budget_state.remaining_text(c)}",
                "",
            ]
        L += [
            "### Scope of any 'best' claim",
            "",
            "Any preference expressed in this report is scoped to the candidate set "
            "considered, the evidence gathered, and the weights in force at the time. "
            "It is not a claim of global optimality, and a different candidate set or "
            "weighting could reorder it.",
            "",
        ]
        return "\n".join(L)

    # ==================================================================

    def _scoring_section(self, state: ResearchState) -> str:
        if state.rubric is None or not state.evaluations:
            return "No scoring was performed.\n"
        r = state.rubric
        weights = r.normalized_weights()
        L: list[str] = []

        L += [
            f"**Rubric `{r.version}`** (derived from contract v"
            f"{r.derived_from_contract_version}). {r.rationale}",
            "",
        ]
        if state.rubric_history:
            L += [
                "Prior rubric version(s) retained: "
                + ", ".join(f"`{h.version}`" for h in state.rubric_history),
                "",
            ]
        L += ["| Dimension | Weight | What it measures | Boundary |", "|---|---|---|---|"]
        for d in r.dimensions:
            L.append(
                f"| {d.name} (`{d.key}`) | {weights[d.key]:.3f} | {d.definition} | "
                f"{d.boundary_note} |"
            )
        L += [
            "",
            "Scores are anchored 0-5 per dimension and shown normalised to 0-100. "
            "A dimension with insufficient evidence is `UNKNOWN`: it is excluded from "
            "the point estimate (remaining weights renormalised) and widens the range "
            "instead. No number is invented to fill a gap, and no subjective "
            "'confidence multiplier' is applied.",
            "",
            "### Ranking",
            "",
            "Ordered by the LOWER bound, not by the point score: the point estimate "
            "renormalises over the dimensions actually scored, which makes it "
            "incomparable between candidates with different evidence coverage. A "
            "candidate with a higher point score can therefore rank lower, and that "
            "is the intended behaviour -- missing evidence must never improve a "
            "candidate's position.",
            "",
            "| Rank | Candidate | Lower (rank key) | Point | Range | Evidence suff. | "
            "Constraints | Pairwise pts | Eligible |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        if state.ranking:
            for e in sorted(state.ranking.entries, key=lambda x: x.rank):
                L.append(
                    f"| {e.rank} | {e.proposal_id} | **{e.score_lower:.1f}** | "
                    f"{e.point_score:.1f} | "
                    f"{e.score_lower:.1f}-{e.score_upper:.1f} | "
                    f"{e.evidence_sufficiency:.0%} | `{e.constraint_status.value}` | "
                    f"{e.pairwise_points:.1f} | {'yes' if e.eligible else 'NO'} |"
                )
        L.append("")

        L += ["### Per-dimension scores", ""]
        for ev in sorted(state.evaluations, key=lambda e: e.proposal_id):
            L += [f"**{ev.proposal_id}** (rubric `{ev.rubric_version}`)", ""]
            L += ["| Dimension | Score | Source | Justification | Evidence |",
                  "|---|---|---|---|---|"]
            for s in ev.dimension_scores:
                raw = "UNKNOWN" if s.status == "UNKNOWN" else f"{s.raw}/5"
                L.append(
                    f"| `{s.dimension_key}` | {raw} | {s.score_source} | "
                    f"{s.justification} | {', '.join(s.evidence_ids) or '--'} |"
                )
            L.append("")

        if state.ranking and state.ranking.pairwise:
            pw = state.ranking.pairwise
            L += [
                "### Pairwise competition (A/B positions swapped)",
                "",
                f"- {pw.comparisons} judge call(s) over {len(pw.outcomes)} pair(s), "
                "each judged twice with positions swapped",
                f"- Order-consistency: **{pw.consistency_rate:.0%}**",
                f"- Position bias (judge picked slot A regardless of content): "
                f"**{pw.position_bias_rate:.0%}**",
                f"- Returned `insufficient_evidence`: {pw.insufficient_evidence_rate:.0%}",
                "",
                pw.note,
                "",
                "| Pair | Forward | Reverse | Consistent | Winner |",
                "|---|---|---|---|---|",
            ]
            for o in pw.outcomes:
                L.append(
                    f"| {o.proposal_x} vs {o.proposal_y} | {o.forward.verdict} | "
                    f"{o.reverse.verdict} | {'yes' if o.consistent else 'NO'} | "
                    f"{o.agreed_winner or 'none'} |"
                )
            L.append("")

        if state.ranking and state.ranking.stability:
            st = state.ranking.stability
            L += [
                "### Weight sensitivity",
                "",
                f"- Top choice stable under +/-20% single-weight changes: "
                f"**{'yes' if st.top1_stable else 'NO'}**",
                f"- Minimum Kendall tau vs. base ranking: {st.min_kendall_tau}",
                f"- Dimensions whose perturbation flips the top choice: "
                f"{', '.join(st.unstable_dimensions) or 'none'}",
                "",
                st.interpretation,
                "",
            ]
        return "\n".join(L)

    # ==================================================================

    def _tests_section(self, state: ResearchState) -> str:
        L: list[str] = []
        if not state.test_plans:
            return "No test plan was produced.\n"

        for plan in state.test_plans:
            L += [
                f"### Test plan `{plan.plan_id}` (v{plan.version}, hash "
                f"`{plan.content_hash()}`)",
                "",
                f"- Authored by: `{plan.author_role}` from contract v"
                f"{plan.derived_from_contract_version}, **before** execution",
                f"- Independent baseline: {plan.independent_baseline or 'none declared'}",
                f"- Randomness controls: {plan.randomness_controls or 'none declared'}",
            ]
            if plan.amendments:
                L.append("- Amendments:")
                for a in plan.amendments:
                    L.append(f"  - v{a.version} by `{a.author}`: {a.reason}")
            else:
                L.append(
                    "- No amendments. The executor is structurally barred from "
                    "amending acceptance criteria."
                )
            L += ["", "| Test | Category | Acceptance criterion | Result |", "|---|---|---|---|"]
            outcomes = {
                o.test_id: o
                for exp in state.experiments
                if exp.plan_id == plan.plan_id
                for o in exp.outcomes
            }
            for case in plan.cases:
                o = outcomes.get(case.test_id)
                if o is None:
                    res = "not run"
                elif o.passed is True:
                    res = "PASS"
                elif o.passed is False:
                    res = "**FAIL**"
                else:
                    res = f"not run ({o.detail})"
                L.append(
                    f"| `{case.test_id}` | {case.category} | "
                    f"{case.acceptance_criterion} | {res} |"
                )
            L.append("")

        L += ["### Experiments", ""]
        for exp in state.experiments:
            kind = (
                "**SIMULATED -- not a real measurement**"
                if exp.is_simulated
                else ("awaiting human execution" if exp.requires_human_input else "real execution")
            )
            L += [
                f"#### `{exp.experiment_id}` ({kind})",
                "",
                f"- Adapter `{exp.adapter_id}` | executed: {_fmt(exp.executed)} | "
                f"duration {exp.duration_seconds}s | seed {_fmt(exp.seed)}",
                f"- Against plan `{exp.plan_id}` hash `{exp.plan_hash}`",
            ]
            if exp.environment:
                L.append(
                    "- Environment: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(exp.environment.items()))
                )
            if exp.error:
                L.append(f"- **Reported problem:** {exp.error}")
            if exp.observations:
                L += [
                    "",
                    "| Metric | Value | Unit | Provenance | Repeats | Std dev |",
                    "|---|---|---|---|---|---|",
                ]
                for o in exp.observations:
                    L.append(
                        f"| `{o.metric_key}` | {o.value} | {o.unit} | "
                        f"{o.provenance.value}"
                        f"{' **(not real)**' if o.provenance is Provenance.SIMULATED else ''}"
                        f" | {o.n_repeats} | {_fmt(o.stddev)} |"
                    )
            else:
                L.append("- No metric observation was recorded.")
            if exp.artifacts:
                L.append("- Artefacts:")
                for k, v in sorted(exp.artifacts.items()):
                    L.append(f"  - `{k}`: `{v}`")
            L.append("")
        return "\n".join(L)

    # ==================================================================

    def _next_actions(self, state: ResearchState) -> list[str]:
        out: list[str] = []
        v = state.latest_validation()
        c = state.contract

        if c:
            no_target = [m.key for m in c.success_metrics if m.target_threshold is None]
            if no_target:
                out.append(
                    f"Set a numeric target for {', '.join(no_target)}: without one, "
                    "goal attainment is undecidable regardless of what is measured."
                )
            assumed = [m.key for m in c.success_metrics if m.assumed]
            if assumed:
                out.append(
                    f"Confirm or replace the harness-assumed metric(s) "
                    f"{', '.join(assumed)} -- these were not stated by the requester."
                )
        if not state.has_real_evidence():
            out.append(
                "Run with a real search backend (`--search web`, needs credentials) or "
                "supply sources via `--human-evidence`: this run gathered no real "
                "evidence."
            )
        if any(e.is_simulated for e in state.experiments):
            out.append(
                "Re-run with `--adapter pytest` against a real implementation to "
                "obtain measurements that can exceed R3."
            )
        if v and v.external_validation_plan:
            out.append(
                "Carry out the external validation steps in section 8, then re-run "
                "with the results supplied as human-provenance evidence."
            )
        if state.ranking and state.ranking.stability and not state.ranking.stability.top1_stable:
            out.append(
                "Gather evidence on "
                f"{', '.join(state.ranking.stability.unstable_dimensions)} before "
                "committing: the top choice flips under small weight changes."
            )
        if state.ranking and state.ranking.pairwise and (
            state.ranking.pairwise.position_bias_rate > 0
        ):
            out.append(
                "Judge position bias was detected; re-judge with a different model or "
                "prompt version before relying on the pairwise ordering."
            )
        remaining = state.remaining_candidates()
        if remaining:
            out.append(
                f"Untested eligible candidate(s) remain: {', '.join(remaining)}."
            )
        if not out:
            out.append("No further action is indicated within the current contract.")
        return out

    def _repro_section(self, state: ResearchState, store) -> str:
        p = state.provenance
        L = [
            "| Field | Value |",
            "|---|---|",
            f"| Run id | `{state.run_id}` |",
            f"| Harness version | {p.harness_version} |",
            f"| Provider | `{p.provider_id}` (simulated: {_fmt(p.is_simulated_provider)}) |",
            f"| Judge | `{p.judge_id}` |",
            f"| Experiment adapter | `{p.experiment_adapter_id}` |",
            f"| Search tools | {', '.join(f'`{t}`' for t in p.search_tool_ids) or 'none'} |",
            f"| Seed | {p.seed} |",
            f"| Prompt version | {p.prompt_version} |",
            f"| Python | {p.python_version} |",
            f"| Platform | {p.platform} |",
            f"| Started | {p.started_at} |",
            f"| LLM calls | {state.budget_state.llm_calls} |",
            f"| Tokens | {state.budget_state.tokens_used} |",
            f"| Cost | ${state.budget_state.cost_usd:.4f} |",
            f"| Elapsed | {state.budget_state.elapsed_seconds:.2f}s |",
            "",
        ]
        if store is not None:
            L += [
                "### Artefacts",
                "",
                f"- Contract: `{store.contract}`",
                f"- Evidence: `{store.evidence}`",
                f"- Proposals: `{store.root / 'proposals'}`",
                f"- Evaluation (rubric, scores, pairwise, ranking): "
                f"`{store.root / 'evaluation'}`",
                f"- Experiments and raw output: `{store.root / 'experiments'}`",
                f"- Decisions: `{store.decisions}`",
                f"- Resumable state: `{store.state}`",
                "",
                "### Re-run / inspect",
                "",
                "```bash",
                f"research-harness status {state.run_id}",
                f"research-harness proposals {state.run_id}",
                f"research-harness evaluate {state.run_id}",
                f"research-harness resume {state.run_id}",
                f"research-harness report {state.run_id}",
                "```",
                "",
                "The run is deterministic under the simulated provider: the same idea "
                "and seed reproduce the same contract, candidates, scores and ranking.",
                "",
            ]
        return "\n".join(L)

    # ==================================================================

    @staticmethod
    def derived_warnings(state: ResearchState) -> list[str]:
        """Warnings that depend on what actually ran, not on what was configured.

        Kept out of the run state so a resumed run cannot carry a stale claim
        about an adapter it no longer uses.
        """
        out: list[str] = []
        simulated = [e for e in state.experiments if e.is_simulated]
        real = [e for e in state.experiments if not e.is_simulated and e.executed]
        if simulated and real:
            out.append(
                f"MIXED experiment provenance: {len(real)} real and "
                f"{len(simulated)} SIMULATED experiment(s) ran in this run. Only "
                "the real ones can support a claim about the goal; check each "
                "observation's provenance before relying on it."
            )
        elif simulated:
            out.append(
                "Experiments ran on the SIMULATED adapter: no real computation was "
                "performed. Maturity is capped at R3_PROTOTYPED by design."
            )
        return out

    def all_warnings(self, state: ResearchState) -> list[str]:
        return list(dict.fromkeys([*state.warnings, *self.derived_warnings(state)]))

    def run_summary(self, state: ResearchState) -> RunSummary:
        """Structured run summary for `run_summary.json` and ablation."""
        v = state.latest_validation()
        contract = state.contract
        ranking = state.ranking
        return RunSummary(
            run_id=state.run_id,
            mode=state.mode,
            idea=contract.original_idea if contract else "",
            maturity_level=state.final_maturity_level or MaturityLevel.INCONCLUSIVE,
            outcome=state.outcome,
            stop_reason=state.stop_decision.reason if state.stop_decision else None,
            stop_detail=state.stop_decision.detail if state.stop_decision else None,
            goal_satisfied=v.goal_satisfied if v else None,
            implementation_conforms=v.implementation_conforms if v else None,
            selected_proposal=state.selected_proposal_id,
            proposal_count=len(state.proposals),
            excluded_for_constraints=(
                list(ranking.excluded_for_constraints) if ranking else []
            ),
            real_evidence_count=sum(1 for e in state.evidence if e.is_real),
            simulated_evidence_count=sum(1 for e in state.evidence if not e.is_real),
            critical_unknowns=list(state.critical_unknowns),
            iterations=state.iteration,
            routing_actions=state.route_actions(),
            experiments_run=len(state.experiments),
            used_simulated_provider=state.provenance.is_simulated_provider,
            used_simulated_experiments=any(e.is_simulated for e in state.experiments),
            tokens_used=state.budget_state.tokens_used,
            cost_usd=state.budget_state.cost_usd,
            elapsed_seconds=state.budget_state.elapsed_seconds,
            llm_calls=state.budget_state.llm_calls,
            top1_stable=(
                ranking.stability.top1_stable
                if ranking and ranking.stability
                else None
            ),
            position_bias_rate=(
                ranking.pairwise.position_bias_rate
                if ranking and ranking.pairwise
                else None
            ),
            evidence_gaps=list(v.evidence_gaps) if v else [],
            external_actions_required=list(v.external_validation_plan) if v else [],
            warnings=self.all_warnings(state),
        )

    def research_report(self, state: ResearchState) -> str:
        """Shorter narrative of what was researched and found."""
        c = state.contract
        L = [f"# Research summary -- {state.run_id}", ""]
        if c:
            L += [
                f"**Idea:** {c.original_idea}",
                "",
                f"**Problem:** {c.problem_statement}",
                "",
                f"**Goal type:** `{c.goal_type}` | **domain:** `{c.domain_hint}`",
                "",
            ]
        real = [e for e in state.evidence if e.is_real]
        L += [
            "## Evidence",
            "",
            f"- {len(state.evidence)} item(s); {len(real)} real, "
            f"{len(state.evidence) - len(real)} simulated",
            f"- Supporting: {sum(1 for e in state.evidence if e.stance is Stance.SUPPORTS)}, "
            f"refuting: {sum(1 for e in state.evidence if e.stance is Stance.REFUTES)}, "
            f"uncertain: {sum(1 for e in state.evidence if e.stance is Stance.UNCERTAIN)}",
            "",
            "## Candidates",
            "",
        ]
        for p in state.proposals:
            ev = state.evaluation(p.proposal_id)
            score = f"{ev.point_score:.1f}" if ev else "n/a"
            status = ev.constraint_status.value if ev else "n/a"
            L.append(
                f"- **{p.proposal_id}** ({p.strategy_archetype}): {p.title} -- "
                f"score {score}, constraints `{status}`"
            )
        L += ["", "## Process notes", ""]
        L += [f"- {n}" for n in state.notes] or ["- none"]
        warnings = self.all_warnings(state)
        if warnings:
            L += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
        L.append("")
        return "\n".join(L)

    def ranking_report(self, state: ResearchState) -> str:
        L = [f"# Ranking report -- {state.run_id}", ""]
        if state.ranking:
            L += [state.ranking.scope_caveat, ""]
        L.append(self._scoring_section(state))
        if state.ranking and state.ranking.excluded_for_constraints:
            L += [
                "## Excluded for hard-constraint violation",
                "",
                "These candidates are ineligible regardless of their other scores; "
                "hard constraints are non-compensatory.",
                "",
            ]
            for pid in state.ranking.excluded_for_constraints:
                ev = state.evaluation(pid)
                L.append(
                    f"- **{pid}**: violates "
                    f"{', '.join(ev.violated_constraints) if ev else 'unknown'}"
                    + (
                        f" (would otherwise score {ev.point_score:.1f}/100)"
                        if ev
                        else ""
                    )
                )
            L.append("")
        return "\n".join(L)

    def status_report(self, state: ResearchState) -> str:
        c = state.contract
        level = state.final_maturity_level
        L = [
            f"Run:        {state.run_id}",
            f"Mode:       {state.mode.value}",
            f"Phase:      {state.phase.value}",
            f"Iteration:  {state.iteration}",
            f"Level:      {level.value if level else '(not yet determined)'}",
            f"Selected:   {state.selected_proposal_id or '(none)'}",
            f"Provider:   {state.provenance.provider_id}"
            f"{'  [SIMULATED]' if state.provenance.is_simulated_provider else ''}",
            f"Judge:      {state.provenance.judge_id}",
            f"Adapter:    {state.provenance.experiment_adapter_id}",
        ]
        if c:
            L.append(f"Budget:     {state.budget_state.remaining_text(c)}")
        if state.outcome:
            L.append(f"Outcome:    {state.outcome.value}")
        if state.stop_decision:
            L.append(
                f"Stopped:    {state.stop_decision.reason} -- {state.stop_decision.detail}"
            )
        if state.routing_decisions:
            L.append(f"Route:      {' -> '.join(state.route_actions())}")
        if state.awaiting_approval_stage:
            L.append(f"AWAITING APPROVAL for stage: {state.awaiting_approval_stage}")
        v = state.latest_validation()
        if v:
            L += [
                f"Conforms:   {_fmt(v.implementation_conforms)}",
                f"Goal met:   {_fmt(v.goal_satisfied)}",
                f"Reason:     {v.blocking_reason}",
            ]
        if state.ranking:
            L.append("")
            L.append("Ranking (ordered by LOWER bound, so missing evidence cannot help):")
            for e in sorted(state.ranking.entries, key=lambda x: x.rank):
                flag = "" if e.eligible else "  <-- INELIGIBLE (hard constraint)"
                L.append(
                    f"  {e.rank}. {e.proposal_id}  lower {e.score_lower:6.2f}  "
                    f"(point {e.point_score:6.2f}, range "
                    f"{e.score_lower:.1f}-{e.score_upper:.1f}, evidence "
                    f"{e.evidence_sufficiency:.0%})  "
                    f"{e.constraint_status.value}{flag}"
                )
            leader = next(
                (
                    e.proposal_id
                    for e in sorted(state.ranking.entries, key=lambda x: x.rank)
                    if e.eligible
                ),
                None,
            )
            selected = state.selected_proposal_id
            if selected and leader and selected != leader:
                tested = any(e.proposal_id == selected for e in state.experiments)
                if tested:
                    L.append(
                        f"  note: {selected} is selected rather than {leader} because "
                        f"it was already experimented on; re-ranking after a real "
                        "experiment does not discard that work. Retiring it is the "
                        "router's call."
                    )
                else:
                    L.append(
                        f"  note: {selected} is selected but {leader} leads the "
                        "ranking and nothing has been tested -- this disagreement is "
                        "a bug, please report it."
                    )
            if any(e.evidence_sufficiency < 1.0 for e in state.ranking.entries):
                L.append(
                    "  note: a higher point score can rank lower -- the point "
                    "estimate renormalises over scored dimensions only, so it is "
                    "not comparable across candidates with different evidence "
                    "coverage. See ADR-014."
                )
        warnings = self.all_warnings(state)
        if warnings:
            L += ["", "Warnings:"] + [f"  - {w}" for w in warnings]
        return "\n".join(L)
