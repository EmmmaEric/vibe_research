"""Calibration of scores against real outcomes (Phase 3).

The question: **do higher-scored candidates actually achieve their goals more
often?** Until that is measured, a score is an internally-consistent ordering
and nothing more.

Three rules this module follows, all of them things the spec warns about:

1. **Only real outcomes count.** Runs whose experiments were simulated are
   excluded entirely. Calibrating against synthetic results would produce a
   confident number describing nothing.
2. **A score is not a probability.** We report *discrimination* -- whether
   achieved goals scored higher than unachieved ones (a Mann-Whitney / AUC
   statistic) -- not a claimed success probability. Converting a 73/100 into
   "73% likely to work" is exactly the error the spec forbids.
3. **Below a minimum sample, report nothing.** ``MIN_LABELLED_OUTCOMES``
   outcomes are required before any statistic is emitted. An AUC computed on
   three runs is noise with a decimal point.

Nothing here feeds back into scoring automatically. It is a diagnostic for a
human deciding whether to trust the rubric, because auto-tuning weights on a
handful of outcomes would overfit and quietly corrupt the baseline.
"""

from __future__ import annotations

from ..core.schemas import Base, MaturityLevel

#: Minimum real (non-simulated) outcome labels before any statistic is reported.
MIN_LABELLED_OUTCOMES = 10

#: Validation levels that count as the goal being genuinely reached.
_ACHIEVED = {MaturityLevel.R5_GOAL_ACHIEVED.value}
#: Levels that count as a real, measured negative result.
_NOT_ACHIEVED = {
    MaturityLevel.R4_VALIDATED.value,
    MaturityLevel.REJECTED.value,
}
#: Levels carrying no outcome information -- excluded, not counted as failure.
_UNINFORMATIVE = {
    MaturityLevel.R1_RESEARCHED.value,
    MaturityLevel.R2_DESIGNED.value,
    MaturityLevel.R3_PROTOTYPED.value,
    MaturityLevel.INCONCLUSIVE.value,
}


class CalibrationRecord(Base):
    """One historical (prediction, outcome) pair from a real run."""

    run_id: str
    proposal_id: str
    rubric_version: str
    predicted_point_score: float
    predicted_lower: float
    evidence_sufficiency: float
    maturity_level: str
    goal_achieved: bool
    was_simulated: bool = False


class ScoreBucket(Base):
    label: str
    lower: float
    upper: float
    n: int
    achieved: int

    @property
    def achieved_rate(self) -> float | None:
        return (self.achieved / self.n) if self.n else None


class CalibrationReport(Base):
    n_candidate_runs: int = 0
    n_excluded_simulated: int = 0
    n_excluded_uninformative: int = 0
    n_labelled: int = 0
    min_required: int = MIN_LABELLED_OUTCOMES
    sufficient_data: bool = False

    #: P(score of an achieved run > score of an unachieved run). 0.5 = no signal.
    discrimination_auc: float | None = None
    mean_score_achieved: float | None = None
    mean_score_not_achieved: float | None = None
    buckets: list[ScoreBucket] = []
    records: list[CalibrationRecord] = []
    interpretation: str = ""
    caveat: str = (
        "Scores are an ordering, not a probability. This report says whether "
        "higher scores have so far coincided with achieved goals; it does not "
        "license reading a score as a success likelihood. Only runs with real "
        "(non-simulated) experiments are included."
    )


def _is_achieved(level: str, goal_satisfied: int | None) -> bool:
    return level in _ACHIEVED or goal_satisfied == 1


def collect_records(repo) -> tuple[list[CalibrationRecord], int, int]:
    """Gather (prediction, outcome) pairs from the research memory.

    Returns ``(records, n_excluded_simulated, n_excluded_uninformative)``.

    A record is kept only when the run has a validation whose experiment was
    *not* simulated, and whose level carries outcome information.
    """
    rows = repo.conn.execute(
        """
        SELECT v.run_id        AS run_id,
               v.proposal_id   AS proposal_id,
               v.maturity_level AS maturity_level,
               v.goal_satisfied   AS goal_satisfied,
               e.is_simulated     AS is_simulated,
               s.point_score      AS point_score,
               s.score_lower      AS score_lower,
               s.evidence_suff    AS evidence_suff,
               s.rubric_version   AS rubric_version
        FROM validations v
        JOIN experiments e
          ON e.run_id = v.run_id AND e.experiment_id = v.experiment_id
        LEFT JOIN scores s
          ON s.run_id = v.run_id AND s.proposal_id = v.proposal_id
        ORDER BY v.run_id, v.proposal_id
        """
    ).fetchall()

    records: list[CalibrationRecord] = []
    excluded_simulated = 0
    excluded_uninformative = 0
    seen: set[tuple[str, str]] = set()

    for row in rows:
        if row["is_simulated"]:
            excluded_simulated += 1
            continue
        level = row["maturity_level"]
        achieved = _is_achieved(level, row["goal_satisfied"])
        if not achieved and level in _UNINFORMATIVE:
            excluded_uninformative += 1
            continue
        if row["point_score"] is None:
            excluded_uninformative += 1
            continue
        key = (row["run_id"], row["proposal_id"])
        if key in seen:
            continue  # one outcome per candidate per run
        seen.add(key)
        records.append(
            CalibrationRecord(
                run_id=row["run_id"],
                proposal_id=row["proposal_id"],
                rubric_version=row["rubric_version"] or "unknown",
                predicted_point_score=float(row["point_score"]),
                predicted_lower=float(row["score_lower"] or 0.0),
                evidence_sufficiency=float(row["evidence_suff"] or 0.0),
                maturity_level=level,
                goal_achieved=achieved,
                was_simulated=False,
            )
        )
    return records, excluded_simulated, excluded_uninformative


def discrimination_auc(
    achieved: list[float], not_achieved: list[float]
) -> float | None:
    """P(random achieved scores above random unachieved), ties counted as half.

    This is the Mann-Whitney U statistic normalised -- equivalent to ROC AUC.
    0.5 means the score carries no signal about outcome; 1.0 means it separates
    them perfectly. Returns None when either group is empty, because a
    comparison needs both.
    """
    if not achieved or not_achieved == []:
        return None
    wins = 0.0
    for a in achieved:
        for b in not_achieved:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return round(wins / (len(achieved) * len(not_achieved)), 4)


def _buckets(records: list[CalibrationRecord], width: float = 20.0) -> list[ScoreBucket]:
    out: list[ScoreBucket] = []
    edge = 0.0
    while edge < 100.0:
        upper = edge + width
        inside = [
            r
            for r in records
            if edge <= r.predicted_point_score < upper
            or (upper >= 100.0 and r.predicted_point_score == 100.0)
        ]
        out.append(
            ScoreBucket(
                label=f"{edge:.0f}-{upper:.0f}",
                lower=edge,
                upper=upper,
                n=len(inside),
                achieved=sum(1 for r in inside if r.goal_achieved),
            )
        )
        edge = upper
    return out


def calibrate(repo) -> CalibrationReport:
    """Build the calibration report from the research memory."""
    records, n_sim, n_uninf = collect_records(repo)
    total_runs = len(
        repo.conn.execute("SELECT DISTINCT run_id FROM validations").fetchall()
    )

    report = CalibrationReport(
        n_candidate_runs=total_runs,
        n_excluded_simulated=n_sim,
        n_excluded_uninformative=n_uninf,
        n_labelled=len(records),
        records=records,
        buckets=_buckets(records),
    )

    if len(records) < MIN_LABELLED_OUTCOMES:
        report.sufficient_data = False
        report.interpretation = (
            f"Insufficient data: {len(records)} real outcome label(s), "
            f"{MIN_LABELLED_OUTCOMES} required. No calibration statistic is "
            "reported, because one computed on this few runs would be noise. "
            f"({n_sim} validation(s) excluded for running on simulated "
            f"experiments; {n_uninf} excluded for carrying no outcome "
            "information, e.g. INCONCLUSIVE or design-only runs.) "
            "Accumulate real experiment runs -- `--adapter pytest` against a real "
            "project -- and re-run this report."
        )
        return report

    achieved = [r.predicted_point_score for r in records if r.goal_achieved]
    missed = [r.predicted_point_score for r in records if not r.goal_achieved]

    report.sufficient_data = True
    report.discrimination_auc = discrimination_auc(achieved, missed)
    report.mean_score_achieved = (
        round(sum(achieved) / len(achieved), 2) if achieved else None
    )
    report.mean_score_not_achieved = (
        round(sum(missed) / len(missed), 2) if missed else None
    )

    # Deliberately no rank-correlation statistic alongside the AUC. A Kendall
    # tau between "ranked by score" and "ranked by outcome" has to break ties
    # somehow, and breaking them by score manufactures correlation out of tied
    # scores -- it reported 0.67 on a sample where every score was identical and
    # the true signal was zero. The AUC already answers this question and
    # credits ties at 0.5, which is the correct treatment.
    auc = report.discrimination_auc
    if auc is None:
        report.interpretation = (
            f"{len(records)} labelled outcome(s), but all fell on one side "
            "(all achieved, or none did). Discrimination needs both outcomes "
            "present; gather runs with mixed results."
        )
    elif auc >= 0.75:
        report.interpretation = (
            f"Scores discriminate outcomes (AUC {auc}): higher-scored candidates "
            f"achieved their goals more often across {len(records)} real run(s). "
            "This supports using the ranking to choose what to try first. It "
            "still does not make a score a probability."
        )
    elif auc >= 0.55:
        report.interpretation = (
            f"Weak discrimination (AUC {auc}) over {len(records)} real run(s). "
            "The ranking carries some signal but not enough to lean on; prefer "
            "cheap verification over trusting the order."
        )
    else:
        report.interpretation = (
            f"No useful discrimination (AUC {auc}) over {len(records)} real "
            "run(s): score has not predicted goal achievement. Treat the current "
            "ranking as unvalidated, and investigate whether the rubric's "
            "dimensions or weights reflect what actually matters for these goals. "
            "Do not auto-tune weights on this sample alone."
        )
    return report


def format_report(report: CalibrationReport) -> str:
    lines = [
        "Score calibration against real outcomes",
        "",
        f"  runs with a validation      : {report.n_candidate_runs}",
        f"  usable outcome labels       : {report.n_labelled} "
        f"(minimum {report.min_required})",
        f"  excluded, simulated run     : {report.n_excluded_simulated}",
        f"  excluded, no outcome info   : {report.n_excluded_uninformative}",
        f"  sufficient data             : {report.sufficient_data}",
    ]
    if report.sufficient_data:
        lines += [
            f"  discrimination (AUC)        : {report.discrimination_auc}"
            "   (0.5 = no signal, 1.0 = perfect separation)",
            f"  mean score, goal achieved   : {report.mean_score_achieved}",
            f"  mean score, not achieved    : {report.mean_score_not_achieved}",
            "",
            "  score bucket    n   achieved   rate",
        ]
        for b in report.buckets:
            rate = b.achieved_rate
            lines.append(
                f"  {b.label:<14}{b.n:<4}{b.achieved:<10}"
                f"{'n/a' if rate is None else f'{rate:.0%}'}"
            )
    lines += ["", f"  {report.interpretation}", "", f"  {report.caveat}", ""]
    return "\n".join(lines)
