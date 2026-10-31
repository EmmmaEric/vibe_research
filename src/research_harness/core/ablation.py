"""Ablation across capability modes (spec section 16).

The spec's central methodological demand: a Goal-Only baseline that every
richer mode must justify itself against. This runs the *same* idea under
several modes with the same model, tools, seed and budget, and reports what
each cost and reached.

What it deliberately does **not** do is declare a winner. One run per mode on
one idea cannot establish that adaptive beats goal-only — the spec says so
directly ("不要仅凭一次 demo 宣称多 Agent 优于单 Agent"). The report states cost
and outcome per arm, flags where they differ, and names what a real comparison
would need.
"""

from __future__ import annotations

from pathlib import Path

from ..core.schemas import (
    AblationReport,
    Budget,
    MaturityLevel,
    Mode,
    ModeResult,
)
from ..reports.generator import ReportGenerator

DEFAULT_ARMS: tuple[Mode, ...] = (Mode.GOAL_ONLY, Mode.ADAPTIVE, Mode.TOURNAMENT)

#: Maturity ordering for "did this arm get further?" comparisons. Non-numeric
#: outcomes share rank 0: INCONCLUSIVE is not "less far" than REJECTED, it is a
#: different kind of answer.
_RANK = {
    MaturityLevel.INCONCLUSIVE: 0,
    MaturityLevel.REJECTED: 0,
    MaturityLevel.R1_RESEARCHED: 1,
    MaturityLevel.R2_DESIGNED: 2,
    MaturityLevel.R3_PROTOTYPED: 3,
    MaturityLevel.R4_VALIDATED: 4,
    MaturityLevel.R5_GOAL_ACHIEVED: 5,
}


def _to_result(state, mode: Mode) -> ModeResult:
    summary = ReportGenerator().run_summary(state)
    return ModeResult(
        mode=mode,
        run_id=state.run_id,
        maturity_level=summary.maturity_level,
        outcome=summary.outcome,
        goal_satisfied=summary.goal_satisfied,
        selected_proposal=summary.selected_proposal,
        proposal_count=summary.proposal_count,
        real_evidence_count=summary.real_evidence_count,
        hard_constraint_violations=len(summary.excluded_for_constraints),
        iterations=summary.iterations,
        llm_calls=summary.llm_calls,
        tokens_used=summary.tokens_used,
        cost_usd=summary.cost_usd,
        elapsed_seconds=summary.elapsed_seconds,
        experiments_run=summary.experiments_run,
        routing_actions=summary.routing_actions,
        top1_stable=summary.top1_stable,
        position_bias_rate=summary.position_bias_rate,
    )


def run_ablation(
    *,
    idea: str,
    build_orchestrator,
    workspace: Path,
    arms: tuple[Mode, ...] = DEFAULT_ARMS,
    budget: Budget | None = None,
    seed: int = 0,
    run_prefix: str = "abl",
    **start_kwargs,
) -> AblationReport:
    """Run ``idea`` once per mode. ``build_orchestrator(run_dir, mode)`` -> Orchestrator.

    Every arm gets the same idea, seed and budget; only the mode differs. That
    is the whole point — a comparison where two things changed measures nothing.
    """
    results: list[ModeResult] = []
    for mode in arms:
        run_id = f"{run_prefix}_{mode.value}"
        orch = build_orchestrator(workspace / f"research_run_{run_id}", mode)
        state = orch.start(
            idea=idea,
            run_id=run_id,
            mode=mode,
            budget=budget or Budget(),
            **start_kwargs,
        )
        results.append(_to_result(state, mode))

    return AblationReport(
        idea=idea, seed=seed, results=results, findings=_findings(results)
    )


def _findings(results: list[ModeResult]) -> list[str]:
    """Factual observations only. No winner is declared."""
    out: list[str] = []
    if not results:
        return ["No arms were run."]

    baseline = next((r for r in results if r.mode is Mode.GOAL_ONLY), None)

    best = max(_RANK.get(r.maturity_level, 0) for r in results)
    reached = [r.mode.value for r in results if _RANK.get(r.maturity_level, 0) == best]
    out.append(
        f"Highest maturity reached was "
        f"{max(results, key=lambda r: _RANK.get(r.maturity_level, 0)).maturity_level.value}"
        f", by: {', '.join(reached)}."
    )

    if baseline is not None:
        base_rank = _RANK.get(baseline.maturity_level, 0)
        for r in results:
            if r.mode is Mode.GOAL_ONLY:
                continue
            rank = _RANK.get(r.maturity_level, 0)
            extra_calls = r.llm_calls - baseline.llm_calls
            extra_tokens = r.tokens_used - baseline.tokens_used
            cost_note = (
                f"{extra_calls:+d} model call(s), {extra_tokens:+d} token(s)"
                if (extra_calls or extra_tokens)
                else "no extra model cost (offline run)"
            )
            if rank > base_rank:
                out.append(
                    f"'{r.mode.value}' reached {r.maturity_level.value} vs "
                    f"'{baseline.maturity_level.value}' for the baseline, for "
                    f"{cost_note}. One instance; not evidence of a general gain."
                )
            elif rank == base_rank:
                out.append(
                    f"'{r.mode.value}' reached the same maturity as the baseline "
                    f"({r.maturity_level.value}) while spending {cost_note} and "
                    f"{r.iterations - baseline.iterations:+d} iteration(s). On this "
                    "instance the extra machinery bought no measurable gain."
                )
            else:
                out.append(
                    f"'{r.mode.value}' reached {r.maturity_level.value}, LOWER than "
                    f"the baseline's {baseline.maturity_level.value}, for {cost_note}."
                )

    counts = {r.mode.value: r.proposal_count for r in results}
    out.append(f"Candidates generated per arm: {counts}.")
    if baseline is not None and baseline.proposal_count > 1:
        out.append(
            "NOTE: the goal_only arm produced more than one candidate, which the "
            "baseline is not supposed to do — the mode gate may be leaking."
        )

    violations = {
        r.mode.value: r.hard_constraint_violations
        for r in results
        if r.hard_constraint_violations
    }
    if violations:
        out.append(
            f"Hard-constraint violations detected and excluded per arm: {violations}."
        )

    selections = {r.selected_proposal for r in results if r.selected_proposal}
    if len(selections) > 1:
        out.append(
            f"Arms disagreed on the selected candidate ({sorted(selections)}); the "
            "choice is mode-sensitive, so it should not be reported as settled."
        )

    unstable = [r.mode.value for r in results if r.top1_stable is False]
    if unstable:
        out.append(
            f"Top choice was unstable under weight perturbation in: "
            f"{', '.join(unstable)}."
        )
    biased = [
        r.mode.value
        for r in results
        if r.position_bias_rate is not None and r.position_bias_rate > 0
    ]
    if biased:
        out.append(f"Judge position bias detected in: {', '.join(biased)}.")

    out.append(
        "To draw a general conclusion, repeat across several goals with a fixed "
        "model and budget and report the distribution; a single instance cannot "
        "separate a real gain from run-to-run variation."
    )
    return out


def format_report(report: AblationReport) -> str:
    lines = [
        f"Ablation: {report.idea[:90]}",
        f"Seed: {report.seed}",
        "",
        f"{'mode':<12}{'maturity':<18}{'outcome':<24}{'cand':>5}{'iter':>6}"
        f"{'exp':>5}{'calls':>7}{'tokens':>9}{'cost$':>9}",
    ]
    for r in report.results:
        lines.append(
            f"{r.mode.value:<12}{r.maturity_level.value:<18}"
            f"{(r.outcome.value if r.outcome else '-'):<24}"
            f"{r.proposal_count:>5}{r.iterations:>6}{r.experiments_run:>5}"
            f"{r.llm_calls:>7}{r.tokens_used:>9}{r.cost_usd:>9.4f}"
        )
    lines += ["", "Route taken per arm:"]
    for r in report.results:
        lines.append(f"  {r.mode.value:<12}{' -> '.join(r.routing_actions) or '(none)'}")
    lines += ["", "Observations:"]
    lines += [f"  - {f}" for f in report.findings]
    lines += ["", "Caveat:", f"  {report.caveat}", ""]
    return "\n".join(lines)
