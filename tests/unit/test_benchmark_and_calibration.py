"""Benchmark, calibration, and value-of-information-driven retrieval.

These cover the Phase 3 additions. The sharpest test here is
``test_benchmark_catches_a_judge_that_rewards_missing_evidence``, which pins the
regression the benchmark originally found: under the old point-score ranking, a
candidate citing nothing outranked one citing a cross-checked source.
"""

from __future__ import annotations

import pytest

from research_harness.core.schemas import (
    ConstraintStatus,
    Provenance,
    MaturityLevel,
)
from research_harness.evaluation import benchmark as bench
from research_harness.evaluation.calibration import (
    MIN_LABELLED_OUTCOMES,
    calibrate,
    collect_records,
    discrimination_auc,
    format_report as format_calibration,
)
from research_harness.evaluation.judge_adapter import MockJudgeAdapter, _clamp
from research_harness.evaluation.mcda import evaluate_proposal
from research_harness.evaluation.rubric import build_rubric
from research_harness.evaluation.stability import order_of
from research_harness.memory.repository import Repository

from ..conftest import make_contract, passing, scores_for


# --------------------------------------------------------------------------
# The two bugs the benchmark found
# --------------------------------------------------------------------------


def test_missing_evidence_can_never_improve_a_ranking():
    """Regression: an UNKNOWN dimension must not raise a candidate's rank.

    Dropping an UNKNOWN dimension and renormalising weights used to *raise* the
    point score whenever the missing dimension would have scored below the
    candidate's own average -- so citing nothing beat citing a mediocre source.
    Ranking on ``score_lower`` removes the perverse incentive.
    """
    r = build_rubric(make_contract(), version="v1")

    # Identical candidates except one has evidence_strength scored, one UNKNOWN.
    cited_scores = scores_for(r, 5)
    for s in cited_scores:
        if s.dimension_key == "evidence_strength":
            s.raw = 3  # real but unspectacular evidence, below its own average
    uncited_scores = scores_for(r, 5, unknown=("evidence_strength",))

    cited, uncited = (
        evaluate_proposal(
            run_id="t", proposal_id=pid, rubric=r, dimension_scores=sc,
            constraint_status=ConstraintStatus.PASS, constraint_assessments=passing([]),
        )
        for pid, sc in (("CITED", cited_scores), ("UNCITED", uncited_scores))
    )

    # The perverse effect is real at the point-estimate level...
    assert uncited.point_score > cited.point_score
    # ...and the ranking key is chosen so it does not decide the outcome.
    assert order_of([cited, uncited])[0] == "CITED"
    assert cited.score_lower > uncited.score_lower
    assert uncited.evidence_sufficiency < cited.evidence_sufficiency


def test_ranking_matches_point_score_when_coverage_is_equal():
    """Using score_lower must cost nothing in the ordinary case."""
    r = build_rubric(make_contract(), version="v1")
    evals = [
        evaluate_proposal(
            run_id="t", proposal_id=pid, rubric=r, dimension_scores=scores_for(r, raw),
            constraint_status=ConstraintStatus.PASS, constraint_assessments=passing([]),
        )
        for pid, raw in (("LOW", 2), ("MID", 3), ("HIGH", 5))
    ]
    assert order_of(evals) == ["HIGH", "MID", "LOW"]
    by_point = [
        e.proposal_id for e in sorted(evals, key=lambda e: -e.point_score)
    ]
    assert order_of(evals) == by_point


def test_anchor_rounding_is_half_up_not_bankers():
    """Regression: round() ties-to-even made anchor boundaries parity-dependent."""
    assert _clamp(4.5) == 5  # round() would give 4
    assert _clamp(3.5) == 4
    assert _clamp(2.5) == 3  # round() would give 2
    assert _clamp(0.5) == 1  # round() would give 0
    # Clamping still holds at the ends.
    assert _clamp(-3.0) == 0
    assert _clamp(99.0) == 5


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


def test_default_suite_passes_with_the_mock_judge():
    report = bench.run_benchmark(judge=MockJudgeAdapter())
    assert report.passed, report.failures
    assert report.pairwise_agreement == 1.0
    assert report.constraint_respect == 1.0
    assert report.goal_fidelity == 1.0
    assert report.relabel_invariant
    assert report.position_bias_rate == 0.0
    assert len(report.cases) == 3


def test_benchmark_detects_a_position_biased_judge():
    report = bench.run_benchmark(
        judge=MockJudgeAdapter(always_prefer_first_slot=True)
    )
    assert not report.passed
    assert report.position_bias_rate > 0
    assert any("position bias" in f for f in report.failures)


def test_benchmark_catches_a_judge_that_prefers_a_dominated_candidate():
    """A deliberately inverted judge must fail the dominance case."""

    class _Inverted(MockJudgeAdapter):
        judge_id = "inverted-test-judge"

        def compare_pair(self, **kw):
            verdict = super().compare_pair(**kw)
            flip = {"A": "B", "B": "A"}
            return verdict.model_copy(
                update={"verdict": flip.get(verdict.verdict, verdict.verdict)}
            )

    report = bench.run_benchmark(judge=_Inverted())
    assert not report.passed
    assert report.pairwise_agreement == 0.0
    assert any("dominated candidate" in f for f in report.failures)


def test_benchmark_catches_a_judge_that_waves_through_a_constraint_violation():
    """A judge that passes every constraint must fail the non-compensation case."""
    from research_harness.core.schemas import ConstraintAssessment

    class _Permissive(MockJudgeAdapter):
        judge_id = "permissive-test-judge"

        @staticmethod
        def _assess_constraint(c, proposal):
            return ConstraintAssessment(
                constraint_key=c.key,
                status=ConstraintStatus.PASS,
                rationale="waved through",
            )

    report = bench.run_benchmark(judge=_Permissive())
    assert not report.passed
    case = next(c for c in report.cases if c.name == "constraint_non_compensation")
    assert not case.constraint_respected
    assert case.selected == "OVERBUDGET"
    assert any("constraint" in f.lower() for f in report.failures)


def test_dominance_case_is_actually_dominant_under_the_rubric():
    """Guard the benchmark's own premise: STRONG must win on every dimension."""
    case = bench.dominance_case()
    rubric = build_rubric(case.contract, version="bench")
    judge = MockJudgeAdapter()

    scores = {
        p.proposal_id: {
            s.dimension_key: s
            for s in judge.score_rubric(
                goal=case.contract, rubric=rubric, proposal=p, evidence=case.evidence
            ).dimension_scores
        }
        for p in case.proposals
    }
    for key in (d.key for d in rubric.dimensions):
        strong, weak = scores["STRONG"][key], scores["WEAK"][key]
        if strong.status == "UNKNOWN" or weak.status == "UNKNOWN":
            continue
        assert strong.raw >= weak.raw, (
            f"benchmark premise broken: STRONG scores {strong.raw} vs WEAK "
            f"{weak.raw} on '{key}', so the case is not a dominance case"
        )


def test_report_states_its_own_scope_limits():
    report = bench.run_benchmark(judge=MockJudgeAdapter())
    text = bench.format_report(report)
    assert "dominance by construction, not expert labelling" in report.scope_note
    assert "Scope" in text
    assert "PASS" in text


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_auc_is_half_for_identical_scores_and_one_for_separation():
    assert discrimination_auc([5.0, 5.0], [5.0, 5.0]) == 0.5
    assert discrimination_auc([9.0, 8.0], [2.0, 1.0]) == 1.0
    assert discrimination_auc([1.0], [9.0]) == 0.0
    # A comparison needs both groups present.
    assert discrimination_auc([], [1.0]) is None
    assert discrimination_auc([1.0], []) is None


def _store_outcome(
    repo: Repository,
    *,
    run_id: str,
    score: float,
    level: MaturityLevel,
    simulated: bool,
    goal: bool | None,
) -> None:
    """Insert one (prediction, outcome) pair straight into the memory."""
    repo.conn.execute(
        "INSERT OR REPLACE INTO scores(run_id, proposal_id, rubric_version, "
        "constraint_status, point_score, score_lower, score_upper, evidence_suff, "
        "eligible, computed_at, payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, "PA", "v1", "PASS", score, score, score, 1.0, 1, "now", "{}"),
    )
    repo.conn.execute(
        "INSERT OR REPLACE INTO experiments(run_id, experiment_id, proposal_id, "
        "plan_id, plan_hash, adapter_id, is_simulated, executed, requires_human, "
        "seed, duration_s, error, started_at, payload) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, "e1", "PA", "p1", "h", "pytest", int(simulated), 1, 0,
         1, 0.1, None, "now", "{}"),
    )
    repo.conn.execute(
        "INSERT OR REPLACE INTO validations(run_id, experiment_id, proposal_id, "
        "conforms, goal_satisfied, maturity_level, blocking_reason, validated_at, "
        "payload) VALUES(?,?,?,?,?,?,?,?,?)",
        (run_id, "e1", "PA", 1, None if goal is None else int(goal),
         level.value, "", "now", "{}"),
    )
    repo.conn.commit()


def test_below_the_minimum_no_statistic_is_reported(tmp_path):
    repo = Repository(tmp_path / "m.sqlite3")
    try:
        for i in range(3):
            _store_outcome(
                repo, run_id=f"r{i}", score=80.0,
                level=MaturityLevel.R5_GOAL_ACHIEVED, simulated=False, goal=True,
            )
        report = calibrate(repo)
        assert report.n_labelled == 3
        assert not report.sufficient_data
        assert report.discrimination_auc is None
        assert "Insufficient data" in report.interpretation
        assert str(MIN_LABELLED_OUTCOMES) in report.interpretation
    finally:
        repo.close()


def test_simulated_runs_are_excluded_entirely(tmp_path):
    """Calibrating on synthetic outcomes would describe nothing."""
    repo = Repository(tmp_path / "m.sqlite3")
    try:
        for i in range(20):
            _store_outcome(
                repo, run_id=f"sim{i}", score=90.0,
                level=MaturityLevel.R5_GOAL_ACHIEVED, simulated=True, goal=True,
            )
        report = calibrate(repo)
        assert report.n_labelled == 0
        assert report.n_excluded_simulated == 20
        assert not report.sufficient_data
        assert report.discrimination_auc is None
    finally:
        repo.close()


def test_inconclusive_runs_are_excluded_not_counted_as_failures(tmp_path):
    repo = Repository(tmp_path / "m.sqlite3")
    try:
        for i in range(12):
            _store_outcome(
                repo, run_id=f"inc{i}", score=70.0,
                level=MaturityLevel.INCONCLUSIVE, simulated=False, goal=None,
            )
        report = calibrate(repo)
        assert report.n_labelled == 0
        assert report.n_excluded_uninformative == 12
    finally:
        repo.close()


def test_discriminating_scores_are_recognised(tmp_path):
    repo = Repository(tmp_path / "m.sqlite3")
    try:
        for i in range(6):  # high scores, goals achieved
            _store_outcome(
                repo, run_id=f"hi{i}", score=85.0 + i,
                level=MaturityLevel.R5_GOAL_ACHIEVED, simulated=False, goal=True,
            )
        for i in range(6):  # low scores, goals missed
            _store_outcome(
                repo, run_id=f"lo{i}", score=30.0 + i,
                level=MaturityLevel.R4_VALIDATED, simulated=False, goal=False,
            )
        report = calibrate(repo)
        assert report.sufficient_data
        assert report.n_labelled == 12
        assert report.discrimination_auc == 1.0
        assert report.mean_score_achieved > report.mean_score_not_achieved
        assert "discriminate" in report.interpretation
        assert sum(b.n for b in report.buckets) == 12
    finally:
        repo.close()


def test_non_predictive_scores_are_reported_as_such(tmp_path):
    """Identical scores with mixed outcomes must yield AUC 0.5 and say so."""
    repo = Repository(tmp_path / "m.sqlite3")
    try:
        for i in range(6):
            _store_outcome(
                repo, run_id=f"a{i}", score=75.0,
                level=MaturityLevel.R5_GOAL_ACHIEVED, simulated=False, goal=True,
            )
        for i in range(6):
            _store_outcome(
                repo, run_id=f"b{i}", score=75.0,
                level=MaturityLevel.R4_VALIDATED, simulated=False, goal=False,
            )
        report = calibrate(repo)
        assert report.discrimination_auc == 0.5
        assert "No useful discrimination" in report.interpretation
        assert "Do not auto-tune weights" in report.interpretation
    finally:
        repo.close()


def test_report_never_presents_a_score_as_a_probability(tmp_path):
    repo = Repository(tmp_path / "m.sqlite3")
    try:
        for i in range(12):
            _store_outcome(
                repo, run_id=f"r{i}", score=60.0 + i,
                level=MaturityLevel.R5_GOAL_ACHIEVED if i % 2 else
                MaturityLevel.R4_VALIDATED,
                simulated=False, goal=bool(i % 2),
            )
        text = format_calibration(calibrate(repo))
        assert "not a probability" in text
        assert "0.5 = no signal" in text
    finally:
        repo.close()


def test_one_outcome_per_candidate_even_with_repeated_rows(tmp_path):
    repo = Repository(tmp_path / "m.sqlite3")
    try:
        _store_outcome(
            repo, run_id="r1", score=80.0,
            level=MaturityLevel.R5_GOAL_ACHIEVED, simulated=False, goal=True,
        )
        _store_outcome(  # same run+proposal written again
            repo, run_id="r1", score=80.0,
            level=MaturityLevel.R5_GOAL_ACHIEVED, simulated=False, goal=True,
        )
        records, _, _ = collect_records(repo)
        assert len(records) == 1
    finally:
        repo.close()


# --------------------------------------------------------------------------
# Value-of-information-driven retrieval
# --------------------------------------------------------------------------


def test_targeted_queries_name_the_assumption_and_seek_disconfirmation(provider):
    from research_harness.agents.researcher import Researcher

    contract = make_contract(idea="speed up a batch pipeline")
    researcher = Researcher(provider, [])
    assumption = "that the input distribution stays sparse in production"

    queries = researcher.plan_targeted_queries(contract, assumption)

    assert queries, "a targeted search must produce queries"
    joined = " ".join(queries).lower()
    assert "sparse in production" in joined
    # Confirming and disconfirming forms are both issued.
    assert any("against" in q for q in queries)
    assert any("did not hold" in q for q in queries)
    # They differ from the broad query set, which is the point.
    assert set(queries) != set(researcher.plan_queries(contract))


def test_targeted_queries_are_bounded_and_deduplicated(provider):
    from research_harness.agents.researcher import Researcher

    researcher = Researcher(provider, [])
    queries = researcher.plan_targeted_queries(
        make_contract(), "x" * 400, limit=4
    )
    assert len(queries) <= 4
    assert len(set(queries)) == len(queries)


def test_critical_unknowns_outrank_the_voi_shortlist(tmp_path, bundled_fixture_dir):
    """Precedence: a blocking unknown beats a candidate-separating assumption.

    An unknown that makes goal attainment undecidable (a required metric with no
    threshold) has to be resolved whichever candidate wins, so it is targeted
    before an assumption that merely distinguishes two candidates.
    """
    from ..integration.test_pipeline import ALGORITHM_IDEA, FULL, build

    orch = build(tmp_path, fixtures=bundled_fixture_dir)
    state = orch.start(idea=ALGORITHM_IDEA, budget=FULL)

    assert state.critical_unknowns, "this goal should leave a blocking unknown"
    assert orch._information_target(state) == state.critical_unknowns[0]


def test_voi_shortlist_is_used_once_no_critical_unknown_remains(
    tmp_path, bundled_fixture_dir
):
    from research_harness.evaluation.decision_engine import rank_experiments

    from ..integration.test_pipeline import ALGORITHM_IDEA, FULL, build

    orch = build(tmp_path, fixtures=bundled_fixture_dir)
    state = orch.start(idea=ALGORITHM_IDEA, budget=FULL)

    # Clear the blocking unknowns; the fallback must be the VoI shortlist.
    state.critical_unknowns = []
    target = orch._information_target(state)

    shortlist = rank_experiments(
        proposals=state.proposals,
        evaluations=state.evaluations,
        ranking=state.ranking,
    )
    assert shortlist, "a scored run should expose candidate experiments"
    assert target == shortlist[0].targets_assumption
    assert shortlist[0].impact_is_predicted is True


def test_follow_up_retrieval_is_recorded_as_targeted(tmp_path, bundled_fixture_dir):
    from ..integration.test_pipeline import ALGORITHM_IDEA, FULL, build

    orch = build(tmp_path, fixtures=bundled_fixture_dir)
    state = orch.start(idea=ALGORITHM_IDEA, budget=FULL)
    before = len(state.evidence)

    orch._act_gather(state)

    assert any("targeted the highest-value open question" in n for n in state.notes)
    assert len(state.evidence) >= before
    # Ids stay unique and sequential after the merge.
    ids = [e.evidence_id for e in state.evidence]
    assert len(ids) == len(set(ids))
