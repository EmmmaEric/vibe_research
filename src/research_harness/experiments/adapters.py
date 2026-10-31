"""Experiment adapters.

Three adapters, differing in what their results are actually worth:

* :class:`PytestExperimentAdapter` -- runs real code in a subprocess and reports
  what it actually measured. ``is_simulated=False``, observations carry
  ``Provenance.MEASURED``. This is the only adapter whose results can support L4
  or L5.
* :class:`SimulatedExperimentAdapter` -- deterministic synthetic numbers for
  demonstrating the pipeline offline. ``is_simulated=True``; the decision engine
  refuses to let its output exceed R3, no matter how good the numbers look.
* :class:`ManualEvidenceAdapter` -- for validation that requires real users,
  devices or physical measurement. It does not pretend to run; it emits
  ``requires_human_input=True`` and a concrete external validation plan, which
  the decision engine turns into ``INCONCLUSIVE``.

Safety posture for the executing adapter: fixed timeout, output caps, a minimal
environment, no shell, and execution confined to a declared working directory.
It runs code the user pointed it at, and nothing it fetched from the internet.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..core.schemas import (
    ExperimentResult,
    MetricObservation,
    Provenance,
    Proposal,
    ResearchContract,
    TestOutcome,
    TestPlan,
)
from ..tools.execution import ExecutionRefused, run_command

MAX_OUTPUT_BYTES = 200_000


@runtime_checkable
class ExperimentAdapter(Protocol):
    adapter_id: str
    is_simulated: bool

    def execute(
        self,
        *,
        contract: ResearchContract,
        proposal: Proposal,
        plan: TestPlan,
        workdir: Path,
        experiment_id: str,
    ) -> ExperimentResult: ...


# --------------------------------------------------------------------------


class PytestExperimentAdapter:
    """Executes a real pytest suite and collects real measurements.

    Contract with the experiment under test: the suite must write a
    ``results.json`` into its working directory of the form::

        {
          "tests": {"T001": true, "T002": false},
          "observations": {
            "metric_key": {"value": 1.23, "unit": "ms", "n_repeats": 5,
                           "stddev": 0.04, "seed": 7}
          }
        }

    Test ids key into the registered test plan, so results are attributable to
    plan cases rather than to whatever the suite chose to call itself. Metrics
    reported here are recorded as ``MEASURED``; if ``results.json`` is missing or
    malformed, that is reported as an error -- never silently substituted.
    """

    adapter_id = "pytest"
    is_simulated = False

    def __init__(
        self,
        *,
        project_dir: str | Path,
        test_target: str = "tests",
        timeout_seconds: float = 300.0,
        results_filename: str = "results.json",
        metric_aliases: dict[str, str] | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.test_target = test_target
        self.timeout = timeout_seconds
        self.results_filename = results_filename
        #: ``contract_metric_key -> key the suite reports``.
        #:
        #: A real goal analyst names metrics from the goal ("spmv_execution_time")
        #: while a suite reports what it measured ("speedup_vs_baseline"); with
        #: template metric names this coincided, with a real model it does not.
        #: Binding a measured quantity to a goal metric is a research judgement,
        #: so it must be stated explicitly here and is recorded on the result --
        #: never guessed by fuzzy name matching.
        self.metric_aliases = dict(metric_aliases or {})

    def available(self) -> tuple[bool, str]:
        if not self.project_dir.is_dir():
            return False, f"project directory {self.project_dir} does not exist"
        target = self.project_dir / self.test_target
        if not target.exists():
            return False, f"test target {target} does not exist"
        return True, ""

    def execute(
        self,
        *,
        contract: ResearchContract,
        proposal: Proposal,
        plan: TestPlan,
        workdir: Path,
        experiment_id: str,
    ) -> ExperimentResult:
        workdir.mkdir(parents=True, exist_ok=True)
        base = ExperimentResult(
            experiment_id=experiment_id,
            run_id=contract.run_id,
            proposal_id=proposal.proposal_id,
            plan_id=plan.plan_id,
            plan_hash=plan.content_hash(),
            adapter_id=self.adapter_id,
            is_simulated=False,
            environment={
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "project_dir": str(self.project_dir),
                "test_target": self.test_target,
            },
        )

        ok, why = self.available()
        if not ok:
            base.error = (
                f"pytest adapter could not run: {why}. No result is reported; this is "
                "not a failure of the candidate."
            )
            return base

        results_path = self.project_dir / self.results_filename
        if results_path.exists():
            results_path.unlink()

        cmd = [sys.executable, "-m", "pytest", self.test_target, "-q", "--tb=short"]
        try:
            run = run_command(
                cmd,
                cwd=self.project_dir,
                root=self.project_dir,
                timeout_seconds=self.timeout,
                extra_env={"RESULTS_PATH": str(results_path)},
            )
        except ExecutionRefused as exc:
            base.error = f"execution refused by the safety check: {exc}"
            return base

        base.duration_seconds = run.duration_seconds
        base.executed = True

        if run.timed_out or run.error:
            base.error = f"the experiment did not complete: {run.error}"
            return base
        code, stdout, stderr = run.exit_code, run.stdout, run.stderr

        log = workdir / "pytest_output.txt"
        log.write_text(
            (f"$ {' '.join(cmd)}\n[exit {code}]\n\n--- stdout ---\n{stdout}"
             f"\n--- stderr ---\n{stderr}")[:MAX_OUTPUT_BYTES],
            encoding="utf-8",
        )
        base.stdout_path = str(log)
        base.artifacts["pytest_output"] = str(log)

        if not results_path.exists():
            base.error = (
                f"the suite ran (exit {code}) but produced no {self.results_filename}, "
                "so no metric can be attributed to a plan case. Reporting no "
                "observations rather than inferring any."
            )
            return base

        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            base.error = f"{self.results_filename} is unreadable: {exc}"
            return base

        base.artifacts["results_json"] = str(results_path)
        base.seed = payload.get("seed")

        plan_ids = {c.test_id for c in plan.cases}
        reported = payload.get("tests") or {}
        for test_id, passed in reported.items():
            if test_id not in plan_ids:
                continue  # results for cases not in the registered plan are ignored
            base.outcomes.append(
                TestOutcome(
                    test_id=test_id,
                    passed=bool(passed),
                    detail="real execution",
                    raw_output_path=str(log),
                )
            )
        for case in plan.cases:
            if case.test_id not in reported:
                base.outcomes.append(
                    TestOutcome(
                        test_id=case.test_id,
                        passed=None,
                        detail=(
                            "not executed"
                            + (
                                f" (requires {case.requires_external_resource})"
                                if case.requires_external_resource
                                else ""
                            )
                        ),
                    )
                )

        declared = {m.key: m for m in contract.success_metrics}
        observed_to_contract = {v: k for k, v in self.metric_aliases.items()}
        if self.metric_aliases:
            base.environment["metric_aliases"] = ", ".join(
                f"{c}<-{o}" for c, o in sorted(self.metric_aliases.items())
            )
        for observed_key, spec in (payload.get("observations") or {}).items():
            key = observed_to_contract.get(observed_key, observed_key)
            if key not in declared:
                continue  # only contract metrics count toward goal attainment
            try:
                value = float(spec["value"])
            except (KeyError, TypeError, ValueError):
                continue
            base.observations.append(
                MetricObservation(
                    metric_key=key,
                    value=value,
                    unit=str(spec.get("unit") or declared[key].unit),
                    provenance=Provenance.MEASURED,
                    n_repeats=int(spec.get("n_repeats", 1)),
                    stddev=(
                        float(spec["stddev"]) if spec.get("stddev") is not None else None
                    ),
                    seed=spec.get("seed"),
                )
            )

        if code != 0 and not any(o.passed is False for o in base.outcomes):
            base.error = (
                f"pytest exited {code} but no plan case was reported as failing; the "
                "suite may have errored during collection. See the captured output."
            )
        return base


# --------------------------------------------------------------------------


class SimulatedExperimentAdapter:
    """Deterministic synthetic results for offline demonstration.

    Every observation it emits carries ``Provenance.SIMULATED`` and
    ``is_simulated=True``, and
    :func:`~research_harness.evaluation.decision_engine.determine_maturity_level`
    caps any run touching it at L3. The numbers are a function of declared
    proposal attributes and a seed -- reproducible, and meaningless as evidence
    about the real world.
    """

    adapter_id = "simulated"
    is_simulated = True

    def __init__(self, provider, *, seed: int = 0) -> None:
        self.provider = provider
        self.seed = seed

    def execute(
        self,
        *,
        contract: ResearchContract,
        proposal: Proposal,
        plan: TestPlan,
        workdir: Path,
        experiment_id: str,
    ) -> ExperimentResult:
        workdir.mkdir(parents=True, exist_ok=True)
        result = ExperimentResult(
            experiment_id=experiment_id,
            run_id=contract.run_id,
            proposal_id=proposal.proposal_id,
            plan_id=plan.plan_id,
            plan_hash=plan.content_hash(),
            adapter_id=self.adapter_id,
            is_simulated=True,
            executed=True,
            seed=self.seed,
            environment={
                "adapter": "SIMULATED -- no real computation was performed",
                "python": sys.version.split()[0],
            },
            duration_seconds=0.0,
        )

        digest = getattr(self.provider, "digest", None)

        def frac(*key: str) -> float:
            if digest is not None:
                return (digest(*key) % 10_000) / 10_000.0
            import hashlib

            blob = "|".join([str(self.seed), *key]).encode()
            return (int.from_bytes(hashlib.sha256(blob).digest()[:4], "big") % 10_000) / 10_000.0

        # Executable cases get a synthetic verdict; non-executable ones stay None.
        for case in plan.cases:
            if not case.executable:
                result.outcomes.append(
                    TestOutcome(
                        test_id=case.test_id,
                        passed=None,
                        detail=(
                            "[SIMULATED] not executable in this adapter"
                            + (
                                f"; requires {case.requires_external_resource}"
                                if case.requires_external_resource
                                else ""
                            )
                        ),
                    )
                )
                continue
            # Candidates admitting more failure modes get more synthetic failures,
            # so the implementation-repair loop has something to react to.
            fail_pressure = 0.12 + 0.05 * len(proposal.failure_modes)
            passed = frac(proposal.proposal_id, case.test_id, "pass") > fail_pressure
            result.outcomes.append(
                TestOutcome(
                    test_id=case.test_id,
                    passed=passed,
                    detail="[SIMULATED] synthetic verdict; not a real test execution",
                )
            )

        for m in contract.success_metrics:
            if m.target_threshold is None:
                continue  # nothing to simulate against
            f = frac(proposal.proposal_id, m.key, "value")
            # Cheap candidates land near the threshold, expensive ones overshoot.
            reach = 0.75 + 0.1 * proposal.declared_cost_units + 0.3 * f
            if m.direction.value == "maximize":
                value = m.target_threshold * reach
            else:
                value = m.target_threshold / max(reach, 0.2)
            result.observations.append(
                MetricObservation(
                    metric_key=m.key,
                    value=round(value, 6),
                    unit=m.unit,
                    provenance=Provenance.SIMULATED,
                    n_repeats=3,
                    stddev=round(abs(value) * 0.03, 6),
                    seed=self.seed,
                )
            )

        note = workdir / "SIMULATED_RESULTS.md"
        note.write_text(
            "# SIMULATED EXPERIMENT -- NOT REAL EVIDENCE\n\n"
            f"Run: {contract.run_id}  Candidate: {proposal.proposal_id}  "
            f"Seed: {self.seed}\n\n"
            "Every number in this directory was generated deterministically from the "
            "candidate's declared attributes and the seed above. No computation "
            "relating to the research goal was performed, nothing was measured, and "
            "these values say nothing about the real world.\n\n"
            "They exist to exercise the harness end to end without external "
            "dependencies. The decision engine caps any run using this adapter at "
            "validation level R3_PROTOTYPED and will never report the goal as achieved on "
            "this basis.\n",
            encoding="utf-8",
        )
        result.artifacts["simulation_notice"] = str(note)
        return result


# --------------------------------------------------------------------------


class ManualEvidenceAdapter:
    """For validation requiring real users, devices or physical measurement.

    Does not simulate and does not execute. Emits the external validation plan
    and ``requires_human_input=True``, which the decision engine renders as
    ``INCONCLUSIVE`` -- the honest outcome when the question cannot be settled
    from here.
    """

    adapter_id = "manual"
    is_simulated = False

    def __init__(self, *, reason: str = "") -> None:
        self.reason = reason

    def execute(
        self,
        *,
        contract: ResearchContract,
        proposal: Proposal,
        plan: TestPlan,
        workdir: Path,
        experiment_id: str,
    ) -> ExperimentResult:
        workdir.mkdir(parents=True, exist_ok=True)
        blocked = [c for c in plan.cases if c.requires_external_resource]

        result = ExperimentResult(
            experiment_id=experiment_id,
            run_id=contract.run_id,
            proposal_id=proposal.proposal_id,
            plan_id=plan.plan_id,
            plan_hash=plan.content_hash(),
            adapter_id=self.adapter_id,
            is_simulated=False,
            requires_human_input=True,
            executed=False,
            environment={"adapter": "manual -- awaiting real-world input"},
            error=(
                self.reason
                or "Validation requires real-world input this harness cannot obtain "
                "autonomously."
            ),
        )
        for case in plan.cases:
            result.outcomes.append(
                TestOutcome(
                    test_id=case.test_id,
                    passed=None,
                    detail=(
                        f"awaiting human execution"
                        + (
                            f" (needs {case.requires_external_resource})"
                            if case.requires_external_resource
                            else ""
                        )
                    ),
                )
            )

        lines = [
            "# External validation plan (human execution required)",
            "",
            f"Run: {contract.run_id}  Candidate: {proposal.proposal_id}",
            "",
            "This harness did not and cannot execute the steps below. Until a person "
            "performs them and records the results, the research goal is neither "
            "confirmed nor refuted.",
            "",
        ]
        for c in blocked or plan.cases:
            lines += [
                f"## {c.test_id} -- {c.description}",
                f"- Needs: {c.requires_external_resource or 'real-world execution'}",
                f"- Procedure: {c.procedure}",
                f"- Accept if: {c.acceptance_criterion}",
                "",
            ]
        lines += [
            "## Recording results",
            "",
            "Append each finding to a JSONL file and re-run with "
            "`--human-evidence <file>`; entries enter the evidence pool with "
            "provenance `human`:",
            "",
            '    {"claim": "...", "source_type": "interview", "stance": "supports",',
            '     "quality": 3, "related_metric": "<contract metric key>"}',
            "",
        ]
        path = workdir / "EXTERNAL_VALIDATION_PLAN.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        result.artifacts["external_validation_plan"] = str(path)
        return result


def external_plan_steps(plan: TestPlan) -> list[str]:
    """Human-readable list of the steps this harness cannot perform."""
    return [
        f"{c.test_id}: {c.description} -- requires {c.requires_external_resource}"
        for c in plan.cases
        if c.requires_external_resource
    ]
