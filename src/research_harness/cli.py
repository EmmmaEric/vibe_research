"""Command-line interface (spec section 18).

    research-harness init
    research-harness run --idea "..." --mode goal_only --offline
    research-harness run --idea-file examples/algorithm_idea.md --mode adaptive
    research-harness status    <run_id>
    research-harness resume    <run_id>
    research-harness proposals <run_id>
    research-harness compare   --idea "..."        # Goal-Only vs Adaptive vs Tournament
    research-harness evaluate  <run_id>
    research-harness weights   <run_id> --set dim=w
    research-harness approve   <run_id> --stage experiment
    research-harness report    <run_id> --format md
    research-harness benchmark
    research-harness calibration
    research-harness chain <run_id> | runs | search "query"

``argparse`` from the standard library: one fewer dependency for a CLI of this
shape. Option precedence is ``--config``/``--task`` < explicit flag, and every
run writes a redacted ``redacted_effective_config.json`` recording what it
actually used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import domains
from .core.ablation import DEFAULT_ARMS, format_report as format_ablation, run_ablation
from .core.budget import ArtifactStore
from .core.orchestrator import Orchestrator, new_run_id
from .core.router import AdaptiveRouter
from .core.schemas import ApprovalPolicy, Budget, MaturityLevel, Mode, Phase
from .core.state import ResearchState
from .evaluation.benchmark import format_report as format_benchmark, run_benchmark
from .evaluation.calibration import calibrate, format_report as format_calibration
from .evaluation.judge_adapter import JevUnavailable, LLMJudgeAdapter, MockJudgeAdapter
from .evaluation.ranking import pareto_frontier
from .experiments.adapters import (
    ManualEvidenceAdapter,
    PytestExperimentAdapter,
    SimulatedExperimentAdapter,
)
from .infrastructure.config import (
    ConfigError,
    apply_defaults,
    effective_config,
    load_layered,
    to_cli_defaults,
)
from .memory.repository import Repository
from .memory.retrieval import MemoryRetrieval
from .providers.base import ProviderError
from .providers.simulated import SimulatedProvider
from .reports.generator import ReportGenerator
from .tools.search import build_search_tools

DEFAULT_WORKSPACE = Path("runs")
DB_NAME = "research_memory.sqlite3"
BUNDLED_FIXTURES = Path(__file__).parent / "fixtures"
DEFAULT_GLOBAL_CONFIG = Path("config/global_config")
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"

DEPTH_CHOICES = [
    MaturityLevel.R1_RESEARCHED.value,
    MaturityLevel.R2_DESIGNED.value,
    MaturityLevel.R3_PROTOTYPED.value,
    MaturityLevel.R4_VALIDATED.value,
    MaturityLevel.R5_GOAL_ACHIEVED.value,
]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _workspace(args) -> Path:
    return Path(getattr(args, "workspace", None) or DEFAULT_WORKSPACE)


def _run_dir(workspace: Path, run_id: str) -> Path:
    return workspace / f"research_run_{run_id}"


def _find_run(workspace: Path, run_id: str) -> Path:
    direct = _run_dir(workspace, run_id) / "state.json"
    if direct.is_file():
        return direct
    for candidate in sorted(workspace.glob("research_run_*/state.json")):
        if run_id in candidate.parent.name:
            return candidate
    raise SystemExit(
        f"no run found for '{run_id}' under {workspace}. List runs with: "
        "research-harness runs"
    )


def _read_idea(args) -> str:
    if getattr(args, "idea_file", None):
        path = Path(args.idea_file)
        if not path.is_file():
            raise SystemExit(f"idea file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"idea file is empty: {path}")
        return text
    if getattr(args, "idea", None):
        return args.idea
    raise SystemExit("provide --idea or --idea-file")


def _build_provider(args):
    if args.provider == "simulated":
        return SimulatedProvider(seed=args.seed)
    if args.provider == "openai":
        if getattr(args, "offline", False):
            raise SystemExit(
                "--offline forbids a network provider. Drop --offline, or use "
                "--provider simulated."
            )
        from .providers.openai_compatible import from_env, from_opencode

        kwargs = {"max_tokens": args.max_output_tokens, "timeout": args.provider_timeout}
        if args.provider_config or args.provider_key:
            return from_opencode(
                config_path=args.provider_config,
                provider_key=args.provider_key or "local",
                model=args.model if args.model != DEFAULT_ANTHROPIC_MODEL else None,
                **kwargs,
            )
        return from_env(**kwargs)
    if args.provider == "anthropic":
        if getattr(args, "offline", False):
            raise SystemExit(
                "--offline forbids a network provider. Drop --offline, or use "
                "--provider simulated."
            )
        from .providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=args.model, effort=args.effort)
    raise SystemExit(f"unknown provider '{args.provider}'")


def _build_judge(args, provider):
    if args.judge == "mock":
        return MockJudgeAdapter()
    if args.judge == "llm":
        if provider.is_simulated:
            raise SystemExit(
                "--judge llm needs a real provider; add --provider openai or "
                "--provider anthropic (or use --judge mock for the offline demo)."
            )
        return LLMJudgeAdapter(provider)
    if args.judge == "jev":
        from .evaluation.judge_adapter import JevJudgeAdapter

        try:
            return JevJudgeAdapter()
        except JevUnavailable as exc:
            raise SystemExit(str(exc)) from exc
    raise SystemExit(f"unknown judge '{args.judge}'")


def _build_adapter(args, provider):
    if args.adapter == "simulated":
        return SimulatedExperimentAdapter(provider, seed=args.seed)
    if args.adapter == "manual":
        return ManualEvidenceAdapter(
            reason="Adapter 'manual' selected: validation requires real-world input."
        )
    if args.adapter == "pytest":
        if not args.project_dir:
            raise SystemExit(
                "--adapter pytest needs --project-dir pointing at a project whose "
                "test suite writes results.json (see examples/ for the contract)."
            )
        adapter = PytestExperimentAdapter(
            project_dir=args.project_dir,
            test_target=args.test_target,
            timeout_seconds=args.experiment_timeout,
            metric_aliases=_metric_aliases(args),
        )
        ok, why = adapter.available()
        if not ok:
            raise SystemExit(f"pytest adapter cannot run: {why}")
        return adapter
    raise SystemExit(f"unknown adapter '{args.adapter}'")


def _build_orchestrator(args, store: ArtifactStore, repo: Repository | None):
    provider = _build_provider(args)
    judge = _build_judge(args, provider)
    adapter = _build_adapter(args, provider)
    mode = args.search
    if getattr(args, "offline", False) and mode in ("web", "all"):
        mode = "fixture"
    tools, notes = build_search_tools(
        mode=mode,
        fixture_dir=args.fixture_dir or BUNDLED_FIXTURES,
        local_root=args.local_root,
        human_evidence=args.human_evidence,
    )
    if getattr(args, "offline", False):
        notes.append(
            "Run was --offline: no network retrieval was attempted. Evidence comes "
            "from bundled fixtures and/or local files only."
        )
    return Orchestrator(
        provider=provider,
        judge=judge,
        search_tools=tools,
        experiment_adapter=adapter,
        store=store,
        repository=repo,
        seed=args.seed,
        search_notes=notes,
        router=AdaptiveRouter(tradeoff_threshold=args.tradeoff_threshold),
    )


def _budget(args) -> Budget:
    return Budget(
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens,
        max_wall_clock_seconds=args.max_seconds,
        max_repair_attempts=args.max_repairs,
        max_consecutive_no_gain=args.max_no_gain,
        max_cost_usd=args.max_cost_usd,
    )


def _metric_aliases(args) -> dict[str, str]:
    """Parse ``--map-metric CONTRACT_KEY=OBSERVED_KEY`` pairs."""
    out: dict[str, str] = {}
    for item in getattr(args, "map_metric", None) or []:
        if "=" not in item:
            raise SystemExit(
                f"--map-metric expects CONTRACT_KEY=OBSERVED_KEY, got '{item}'"
            )
        contract_key, observed = item.split("=", 1)
        if not contract_key.strip() or not observed.strip():
            raise SystemExit(f"--map-metric has an empty side: '{item}'")
        out[contract_key.strip()] = observed.strip()
    return out


def _metric_overrides(args) -> tuple[dict[str, float], list[str]]:
    out: dict[str, float] = {}
    for item in getattr(args, "set_metric", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set-metric expects KEY=THRESHOLD, got '{item}'")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = float(raw)
        except ValueError as exc:
            raise SystemExit(f"threshold for '{key}' is not a number: {raw}") from exc
    return out, list(getattr(args, "remove_metric", None) or [])


def _write_effective_config(store: ArtifactStore, args) -> None:
    cfg = getattr(args, "_merged_config", {}) or {}
    overrides = {
        k: v
        for k, v in vars(args).items()
        if not k.startswith("_") and k not in ("func",) and v is not None
    }
    (store.root / "redacted_effective_config.json").write_text(
        json.dumps(effective_config(cfg, cli_overrides=overrides), indent=2,
                   ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_init(args) -> int:
    ws = _workspace(args)
    ws.mkdir(parents=True, exist_ok=True)
    repo = Repository(ws / DB_NAME)
    fts = "available" if repo.fts_available else "NOT available (search falls back to LIKE)"
    repo.close()
    print(f"Workspace:       {ws.resolve()}")
    print(f"Memory DB:       {(ws / DB_NAME).resolve()}")
    print(f"SQLite FTS5:     {fts}")
    print(f"Bundled fixtures:{BUNDLED_FIXTURES.resolve()}  [SIMULATED demo evidence]")
    print(f"Global config:   {DEFAULT_GLOBAL_CONFIG} "
          f"({'present' if DEFAULT_GLOBAL_CONFIG.is_dir() else 'not found'})")
    print(f"Domain profiles: {', '.join(p.name for p in domains.REGISTRY)}")
    print(f"Modes:           {', '.join(m.value for m in Mode)}")
    print()
    print('Next: research-harness run --idea "<your idea>" --offline')
    return 0


def cmd_run(args) -> int:
    ws = _workspace(args)
    ws.mkdir(parents=True, exist_ok=True)
    idea = _read_idea(args)
    run_id = args.run_id or new_run_id()
    store = ArtifactStore(_run_dir(ws, run_id))
    repo = None if args.no_db else Repository(ws / DB_NAME)
    set_metrics, remove_metrics = _metric_overrides(args)

    try:
        orch = _build_orchestrator(args, store, repo)
        _write_effective_config(orch.store, args)
        state = orch.start(
            idea=idea,
            run_id=run_id,
            mode=Mode(args.mode),
            budget=_budget(args),
            approval_policy=ApprovalPolicy(
                require_approval_for=list(args.require_approval or [])
            ),
            desired_maturity_level=MaturityLevel(args.depth),
            domain_hint=args.domain,
            set_metrics=set_metrics,
            remove_metrics=remove_metrics,
        )
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2
    finally:
        if repo:
            repo.close()

    print(ReportGenerator().status_report(state))
    print()
    print(f"Artifacts:    {store.root.resolve()}")
    print(f"Final report: {store.final_report.resolve()}")
    if state.phase is Phase.AWAITING_APPROVAL:
        print()
        print(
            f"Paused awaiting approval for stage '{state.awaiting_approval_stage}':\n"
            f"  research-harness approve {state.run_id} "
            f"--stage {state.awaiting_approval_stage}"
        )
    return 0


def cmd_compare(args) -> int:
    """Goal-Only vs Adaptive vs Tournament on one identical idea (spec 16)."""
    ws = _workspace(args)
    ws.mkdir(parents=True, exist_ok=True)
    idea = _read_idea(args)
    arms = (
        tuple(Mode(m) for m in args.modes) if args.modes else DEFAULT_ARMS
    )
    repo = None if args.no_db else Repository(ws / DB_NAME)
    prefix = args.run_prefix or f"abl{new_run_id()[-6:]}"
    set_metrics, remove_metrics = _metric_overrides(args)

    def build(run_dir: Path, mode: Mode) -> Orchestrator:
        return _build_orchestrator(args, ArtifactStore(run_dir), repo)

    try:
        report = run_ablation(
            idea=idea,
            build_orchestrator=build,
            workspace=ws,
            arms=arms,
            budget=_budget(args),
            seed=args.seed,
            run_prefix=prefix,
            desired_maturity_level=MaturityLevel(args.depth),
            domain_hint=args.domain,
            set_metrics=set_metrics,
            remove_metrics=remove_metrics,
        )
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 2
    finally:
        if repo:
            repo.close()

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(format_ablation(report))
    out = ws / f"ablation_{prefix}.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if not args.json:
        print(f"written to {out}")
    return 0


def cmd_status(args) -> int:
    state = ResearchState.load(_find_run(_workspace(args), args.run_id))
    print(ReportGenerator().status_report(state))
    return 0


def cmd_proposals(args) -> int:
    state = ResearchState.load(_find_run(_workspace(args), args.run_id))
    if args.json:
        print(
            json.dumps(
                [p.model_dump(mode="json") for p in state.proposals],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if not state.proposals:
        print("No candidates were generated.")
        return 0
    for p in state.proposals:
        ev = state.evaluation(p.proposal_id)
        print(f"=== {p.proposal_id}  [{p.strategy_archetype}]  {p.title}")
        print(f"  mechanism: {p.core_mechanism}")
        print(f"  cost:      {p.declared_cost_units} unit(s)")
        if ev:
            print(
                f"  score:     {ev.point_score:.2f}/100 "
                f"(range {ev.score_lower:.1f}-{ev.score_upper:.1f}, "
                f"evidence sufficiency {ev.evidence_sufficiency:.0%})"
            )
            print(f"  constraints: {ev.constraint_status.value}", end="")
            if ev.violated_constraints:
                print(f"  VIOLATES: {', '.join(ev.violated_constraints)}", end="")
            print(f"  eligible: {'yes' if ev.eligible else 'NO'}")
        if p.critical_assumptions:
            print(f"  must be true: {'; '.join(p.critical_assumptions)}")
        print(f"  cites: {', '.join(p.evidence_ids) or '(no evidence)'}")
        print()
    return 0


def cmd_evaluate(args) -> int:
    state = ResearchState.load(_find_run(_workspace(args), args.run_id))
    print(ReportGenerator().ranking_report(state))
    if state.evaluations:
        pareto = pareto_frontier(state.evaluations)
        print("## Pareto analysis (no-trade-off view)")
        print()
        print(f"Non-dominated: {', '.join(pareto.frontier) or '(none)'}")
        if pareto.dominated:
            for pid, dominators in sorted(pareto.dominated.items()):
                print(f"  {pid} is dominated by {', '.join(dominators)}")
        print()
        print(pareto.interpretation)
        print()
    return 0


def cmd_weights(args) -> int:
    ws = _workspace(args)
    path = _find_run(ws, args.run_id)
    state = ResearchState.load(path)
    rubric = state.rubric
    if rubric is None:
        print("This run has no rubric yet.")
        return 1

    if not args.set:
        norm = rubric.normalized_weights()
        print(f"Rubric version: {rubric.version}")
        print(f"{'dimension':<22}{'raw':>8}{'normalised':>13}")
        for d in rubric.dimensions:
            print(f"{d.key:<22}{d.weight:>8.3f}{norm[d.key]:>13.3f}")
        if state.rubric_history:
            print()
            print(
                "Previous versions retained: "
                + ", ".join(h.version for h in state.rubric_history)
            )
        return 0

    weights: dict[str, float] = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects dimension=weight, got '{item}'")
        key, raw = item.split("=", 1)
        try:
            weights[key.strip()] = float(raw)
        except ValueError as exc:
            raise SystemExit(f"weight for '{key}' is not a number: {raw}") from exc

    repo = None if args.no_db else Repository(ws / DB_NAME)
    try:
        orch = _build_orchestrator(args, ArtifactStore(path.parent), repo)
        before = (
            [e.proposal_id for e in sorted(state.ranking.entries, key=lambda x: x.rank)]
            if state.ranking
            else []
        )
        state = orch.reweight(state, weights, reason=args.reason)
        after = (
            [e.proposal_id for e in sorted(state.ranking.entries, key=lambda x: x.rank)]
            if state.ranking
            else []
        )
    except ValueError as exc:
        raise SystemExit(f"cannot reweight: {exc}") from exc
    finally:
        if repo:
            repo.close()

    print(f"Rubric is now {state.rubric.version if state.rubric else '?'}")
    print(f"Ranking before: {' > '.join(before) or '(none)'}")
    print(f"Ranking after:  {' > '.join(after) or '(none)'}")
    print("CHANGED" if before != after else "unchanged")
    print(f"Selected: {state.selected_proposal_id}")
    return 0


def cmd_approve(args) -> int:
    ws = _workspace(args)
    path = _find_run(ws, args.run_id)
    state = ResearchState.load(path)
    repo = None if args.no_db else Repository(ws / DB_NAME)
    try:
        orch = _build_orchestrator(args, ArtifactStore(path.parent), repo)
        state = orch.approve(state, args.stage)
    finally:
        if repo:
            repo.close()
    print(f"Approved stage '{args.stage}'.")
    print()
    print(ReportGenerator().status_report(state))
    return 0


def cmd_resume(args) -> int:
    ws = _workspace(args)
    path = _find_run(ws, args.run_id)
    state = ResearchState.load(path)

    if args.depth and "depth" in getattr(args, "_explicit_flags", set()):
        # Deepening reopens a finished run on its existing contract, which is
        # the only way to bind an experiment to emergent metric names. Requires
        # an explicitly typed --depth: a config default must not reopen a run.
        repo = None if args.no_db else Repository(ws / DB_NAME)
        try:
            orch = _build_orchestrator(args, ArtifactStore(path.parent), repo)
            state = orch.deepen(state, MaturityLevel(args.depth))
        except ValueError as exc:
            print(f"cannot deepen: {exc}", file=sys.stderr)
            return 2
        finally:
            if repo:
                repo.close()
        print(ReportGenerator().status_report(state))
        return 0

    if state.phase in (Phase.DONE, Phase.STOPPED):
        print(
            f"Run {state.run_id} is already {state.phase.value}"
            + (
                f" ({state.stop_decision.reason}: {state.stop_decision.detail})"
                if state.stop_decision
                else ""
            )
        )
        return 0
    if state.phase is Phase.AWAITING_APPROVAL:
        print(
            f"Run is awaiting approval for '{state.awaiting_approval_stage}'. Use: "
            f"research-harness approve {state.run_id} "
            f"--stage {state.awaiting_approval_stage}"
        )
        return 1

    repo = None if args.no_db else Repository(ws / DB_NAME)
    try:
        orch = _build_orchestrator(args, ArtifactStore(path.parent), repo)
        state = orch.resume(state)
    finally:
        if repo:
            repo.close()
    print(ReportGenerator().status_report(state))
    return 0


def cmd_report(args) -> int:
    path = _find_run(_workspace(args), args.run_id)
    state = ResearchState.load(path)
    if args.format == "json":
        print(state.to_json())
        return 0
    if args.format == "summary":
        print(ReportGenerator().run_summary(state).model_dump_json(indent=2))
        return 0
    report = path.parent / "final_report.md"
    if report.is_file() and not args.regenerate:
        print(report.read_text(encoding="utf-8"))
    else:
        print(ReportGenerator().final_report(state, store=ArtifactStore(path.parent)))
    return 0


def cmd_chain(args) -> int:
    repo = Repository(_workspace(args) / DB_NAME)
    try:
        chain = repo.causal_chain(args.run_id)
    finally:
        repo.close()
    if not any(chain.values()):
        print(f"No stored chain for run '{args.run_id}'.")
        return 1
    if args.json:
        print(json.dumps(chain, indent=2, ensure_ascii=False))
        return 0
    for stage, rows in chain.items():
        print(f"--- {stage} ({len(rows)}) ---")
        for r in rows:
            print(
                "  "
                + "  ".join(
                    f"{k}={v}" for k, v in r.items() if k != "payload" and v is not None
                )
            )
        print()
    return 0


def cmd_events(args) -> int:
    """Print the append-only audit log for a run."""
    state = ResearchState.load(_find_run(_workspace(args), args.run_id))
    if not state.events:
        print("No events recorded.")
        return 0
    if args.json:
        print(
            json.dumps(
                [e.model_dump(mode="json") for e in state.events],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    for e in state.events:
        key = f"  [key={e.idempotency_key}]" if e.idempotency_key else ""
        print(f"{e.seq:>3}. {e.event.value:<22}{e.summary}{key}")
    return 0


def cmd_runs(args) -> int:
    db = _workspace(args) / DB_NAME
    if not db.is_file():
        print(f"No memory database at {db}. Run `research-harness init` first.")
        return 1
    repo = Repository(db)
    try:
        rows = repo.list_runs(limit=args.limit)
    finally:
        repo.close()
    if not rows:
        print("No runs recorded.")
        return 0
    print(f"{'run_id':<22}{'phase':<12}{'maturity':<18}{'sim':<5}idea")
    for r in rows:
        idea = (r["original_idea"] or "")[:46].replace("\n", " ")
        print(
            f"{r['run_id']:<22}{r['phase']:<12}"
            f"{(r['maturity_level'] or '-'):<18}"
            f"{('yes' if r['is_simulated'] else 'no'):<5}{idea}"
        )
    return 0


def cmd_search(args) -> int:
    db = _workspace(args) / DB_NAME
    if not db.is_file():
        print(f"No memory database at {db}.")
        return 1
    repo = Repository(db)
    try:
        hits = MemoryRetrieval(repo).search(args.query, limit=args.limit)
    finally:
        repo.close()
    if not hits:
        print("No matches in research memory.")
        return 0
    for h in hits:
        print(f"[{h.kind:<9}] {h.run_id} {h.ref_id}: {h.text[:160]}")
    return 0


def cmd_benchmark(args) -> int:
    provider = _build_provider(args)
    judge = _build_judge(args, provider)
    report = run_benchmark(judge=judge, top_k=args.top_k)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(format_benchmark(report))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"written to {path}")
    return 0 if report.passed else 1


def cmd_calibration(args) -> int:
    db = _workspace(args) / DB_NAME
    if not db.is_file():
        print(f"No memory database at {db}. Run `research-harness init` first.")
        return 1
    repo = Repository(db)
    try:
        report = calibrate(repo)
    finally:
        repo.close()
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(format_calibration(report))
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.add_argument(
        "--provider",
        default="simulated",
        choices=("simulated", "anthropic", "openai"),
        help="simulated = offline deterministic; openai = any OpenAI-compatible "
        "endpoint (incl. proxies); anthropic = Claude via the official SDK",
    )
    p.add_argument("--model", default=DEFAULT_ANTHROPIC_MODEL)
    p.add_argument(
        "--provider-config",
        default=None,
        metavar="FILE",
        help="opencode.json to read baseURL/apiKey from (--provider openai)",
    )
    p.add_argument(
        "--provider-key",
        default=None,
        metavar="NAME",
        help="provider entry in the opencode config (default 'local')",
    )
    p.add_argument("--max-output-tokens", type=int, default=16000)
    p.add_argument("--provider-timeout", type=float, default=300.0)
    p.add_argument(
        "--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max")
    )
    p.add_argument("--judge", default="mock", choices=("mock", "llm", "jev"))
    p.add_argument(
        "--adapter",
        default="simulated",
        choices=("simulated", "pytest", "manual"),
        help="experiment adapter; only 'pytest' can produce real measurements",
    )
    p.add_argument("--project-dir", default=None, help="for --adapter pytest")
    p.add_argument("--test-target", default="tests", help="for --adapter pytest")
    p.add_argument(
        "--map-metric",
        nargs="*",
        default=None,
        metavar="CONTRACT_KEY=OBSERVED_KEY",
        help="bind a contract metric to the key the test suite reports; needed "
        "when a real model names metrics from the goal rather than a template",
    )
    p.add_argument("--experiment-timeout", type=float, default=300.0)
    p.add_argument(
        "--search",
        default="fixture",
        choices=("fixture", "local", "web", "all", "none"),
        help="evidence retrieval; 'web' needs credentials",
    )
    p.add_argument("--fixture-dir", default=None)
    p.add_argument("--local-root", default=None)
    p.add_argument("--human-evidence", default=None, help="JSONL of human findings")
    p.add_argument(
        "--offline",
        action="store_true",
        help="refuse any network tool; guarantees a self-contained run",
    )
    p.add_argument("--tradeoff-threshold", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-db", action="store_true", help="skip SQLite persistence")
    p.add_argument(
        "--config",
        default=None,
        metavar="DIR",
        help=f"global config directory (default {DEFAULT_GLOBAL_CONFIG} if present)",
    )
    p.add_argument(
        "--task", default=None, metavar="FILE", help="task config overlay (TOML)"
    )


def _add_goal_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--idea", default=None)
    p.add_argument("--idea-file", default=None, help="read the idea from a file")
    p.add_argument(
        "--mode",
        default=Mode.ADAPTIVE.value,
        choices=[m.value for m in Mode],
        help="goal_only = single-agent baseline; adaptive = routed; tournament = always compete",
    )
    p.add_argument(
        "--domain", default=None, choices=[d.name for d in domains.REGISTRY]
    )
    p.add_argument(
        "--depth",
        default=MaturityLevel.R3_PROTOTYPED.value,
        choices=DEPTH_CHOICES,
        help="requested maturity; R1 stops after research, R2 after planning",
    )
    p.add_argument("--max-iterations", type=int, default=8)
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument("--max-no-gain", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--max-cost-usd", type=float, default=None)
    p.add_argument(
        "--set-metric", nargs="*", default=None, metavar="KEY=THRESHOLD"
    )
    p.add_argument("--remove-metric", nargs="*", default=None, metavar="KEY")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-harness",
        description=(
            "Adaptive research harness: a goal becomes evidence, competing "
            "candidates, deterministic scoring, a verification plan, and an "
            "honest maturity level."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a workspace")
    _add_engine_args(p)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="run a research cycle on an idea")
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--require-approval", nargs="*", default=None, metavar="STAGE"
    )
    _add_goal_args(p)
    _add_engine_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "compare", help="run one idea under several modes and compare (spec 16)"
    )
    p.add_argument(
        "--modes", nargs="*", default=None, choices=[m.value for m in Mode]
    )
    p.add_argument("--run-prefix", default=None)
    p.add_argument("--json", action="store_true")
    _add_goal_args(p)
    _add_engine_args(p)
    p.set_defaults(func=cmd_compare)

    for name, fn, helptext in (
        ("status", cmd_status, "show a run's current state"),
        ("evaluate", cmd_evaluate, "print the ranking and Pareto report"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("run_id")
        p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
        p.set_defaults(func=fn)

    p = sub.add_parser("proposals", help="list candidates and their scores")
    p.add_argument("run_id")
    p.add_argument("--json", action="store_true")
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.set_defaults(func=cmd_proposals)

    p = sub.add_parser("weights", help="view or change scoring weights and re-rank")
    p.add_argument("run_id")
    p.add_argument("--set", nargs="*", metavar="DIM=WEIGHT")
    p.add_argument("--reason", default="user adjustment")
    _add_engine_args(p)
    p.set_defaults(func=cmd_weights)

    p = sub.add_parser("approve", help="approve a gated stage and continue")
    p.add_argument("run_id")
    p.add_argument("--stage", required=True)
    _add_engine_args(p)
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("resume", help="resume an interrupted run")
    p.add_argument("run_id")
    p.add_argument(
        "--depth",
        default=None,
        choices=DEPTH_CHOICES,
        help="raise the requested maturity and continue on the SAME contract; "
        "needed to bind an experiment to a real model's emergent metric names",
    )
    _add_engine_args(p)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("report", help="print the final report")
    p.add_argument("run_id")
    p.add_argument(
        "--format", default="markdown", choices=("markdown", "md", "json", "summary")
    )
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("chain", help="show the stored research causal chain")
    p.add_argument("run_id")
    p.add_argument("--json", action="store_true")
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.set_defaults(func=cmd_chain)

    p = sub.add_parser("events", help="show the append-only audit log")
    p.add_argument("run_id")
    p.add_argument("--json", action="store_true")
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.set_defaults(func=cmd_events)

    p = sub.add_parser("runs", help="list recorded runs")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("search", help="search research memory across runs")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.set_defaults(func=cmd_search)

    p = sub.add_parser(
        "benchmark", help="grade the scoring system against constructed ground truth"
    )
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None)
    _add_engine_args(p)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser(
        "calibration", help="check whether scores predicted real outcomes"
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.set_defaults(func=cmd_calibration)

    return parser


def _force_utf8_stdout() -> None:
    """Print non-ASCII reliably on consoles that default to a legacy codepage.

    Windows terminals often default to cp936/cp1252, which turns a Chinese idea
    string into mojibake in the report. Reconfiguring is harmless where stdout
    is already UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass  # redirected or already wrapped; nothing to do


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)

    # Which options the user actually typed. A flag that *triggers an action*
    # (rather than setting a value) must never be supplied by a config default:
    # `resume --depth` reopens a finished run, and letting
    # `target.maturity_level` in a config file do that silently was a real bug.
    explicit = {
        tok.lstrip("-").split("=", 1)[0].replace("-", "_")
        for tok in raw
        if tok.startswith("--")
    }
    args._explicit_flags = explicit

    global_dir = getattr(args, "config", None)
    task_file = getattr(args, "task", None)
    if global_dir is None and hasattr(args, "config") and DEFAULT_GLOBAL_CONFIG.is_dir():
        global_dir = DEFAULT_GLOBAL_CONFIG
    if global_dir or task_file:
        try:
            merged = load_layered(global_dir=global_dir, task_file=task_file)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        args._merged_config = merged
        apply_defaults(args, to_cli_defaults(merged), explicit)
    else:
        args._merged_config = {}

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted; run state is preserved on disk", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
