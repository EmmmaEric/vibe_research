"""Dynamic rubric construction.

The *catalogue* of dimensions and their semantic anchors is deterministic code,
not model output, so scores mean the same thing across runs and domains. What
the Goal Analyst may influence is limited and validated: which optional
dimensions are included, and the weight prior.

Cost is expressed as *cost favourability* (5 = very cheap) so that every
dimension is "higher is better" by default and no inversion bug can creep into
the aggregation. ``higher_is_better=False`` is still supported for custom
dimensions.
"""

from __future__ import annotations

from ..core.schemas import Dimension, ResearchContract, Rubric

# --------------------------------------------------------------------------
# Dimension catalogue
# --------------------------------------------------------------------------

_CATALOGUE: dict[str, dict] = {
    "goal_alignment": {
        "name": "Goal Alignment",
        "definition": (
            "Does the proposal address the ORIGINAL goal as written in the research "
            "contract, rather than substituting an easier or adjacent goal?"
        ),
        "boundary_note": (
            "Concerns only *what* is targeted, not whether it can be built "
            "(Feasibility) or whether it would pay off (Expected Benefit)."
        ),
        "anchors": {
            0: "Addresses a different problem, or silently redefines the goal.",
            1: "Touches the goal only incidentally; most success metrics unaddressed.",
            2: "Addresses part of the goal; at least one key metric left unaddressed.",
            3: "Addresses the stated goal with minor drift in scope or emphasis.",
            4: "Addresses the stated goal directly; all success metrics mapped.",
            5: "Addresses the goal directly and makes the mapping to every success "
            "metric explicit and measurable.",
        },
        "evidence": ["contract", "proposal_text"],
    },
    "feasibility": {
        "name": "Feasibility",
        "definition": (
            "Can this actually be carried out with the available theory, technique, "
            "skills and resources named in the contract?"
        ),
        "boundary_note": (
            "Concerns whether it *can be done at all* with what we have. The "
            "probability that it fails once attempted belongs to Risk & Robustness; "
            "how much it costs belongs to Cost & Time."
        ),
        "anchors": {
            0: "Requires something physically, legally or mathematically impossible.",
            1: "Depends on capabilities or resources that are absent and unobtainable "
            "within scope.",
            2: "Depends on unavailable resources, but substitutes plausibly exist.",
            3: "Achievable, but needs non-trivial new capability or a hard dependency.",
            4: "Achievable with known technique and declared resources.",
            5: "Clearly achievable; every step maps to a demonstrated capability.",
        },
        "evidence": ["paper", "repo", "spec", "measured"],
    },
    "evidence_strength": {
        "name": "Evidence Strength",
        "definition": (
            "Quality, relevance and reproducibility of the cited evidence supporting "
            "the proposal's core claims."
        ),
        "boundary_note": (
            "Judges the *support* for the claims, not the claims' attractiveness. "
            "A well-evidenced mediocre plan scores high here and low on Expected Benefit."
        ),
        "anchors": {
            0: "No evidence, or cited sources are fabricated/unavailable.",
            1: "Only unsourced model assertion.",
            2: "Weak or tangential sources; applicability to this context unshown.",
            3: "Relevant sources, but single-source or not independently reproduced.",
            4: "Multiple relevant, cross-checked sources with stated applicability.",
            5: "Multiple independent sources plus a reproduction or direct measurement "
            "in a comparable setting.",
        },
        "evidence": ["paper", "repo", "spec", "measured", "cross_checked"],
    },
    "expected_benefit": {
        "name": "Expected Benefit / Utility",
        "definition": (
            "If it works, how much does it move the contract's success metrics or "
            "user utility?"
        ),
        "boundary_note": (
            "Upside magnitude *conditional on success*. The chance of not working is "
            "Risk & Robustness; the price of trying is Cost & Time."
        ),
        "anchors": {
            0: "No improvement, or makes the target metrics worse.",
            1: "Marginal improvement, likely below measurement noise.",
            2: "Small but measurable improvement on a secondary metric.",
            3: "Meaningful improvement on a primary metric, short of the target.",
            4: "Reaches the contract target on the primary metric.",
            5: "Clearly exceeds the contract target, or unlocks a capability the "
            "target implies.",
        },
        "evidence": ["paper", "measured", "benchmark"],
    },
    "cost_time": {
        "name": "Cost & Time Favourability",
        "definition": (
            "Total time, compute, money, labour and maintenance burden. Scored as "
            "FAVOURABILITY: 5 means very cheap and fast, 0 means prohibitive."
        ),
        "boundary_note": (
            "Expenditure only. Whether the spend is affordable *at all* under a "
            "budget ceiling is a hard constraint, not a score."
        ),
        "anchors": {
            0: "Cost or duration exceeds the declared budget outright.",
            1: "Consumes essentially the whole budget.",
            2: "Large share of budget; little left for iteration.",
            3: "Moderate cost; one or two iterations remain affordable.",
            4: "Low cost relative to budget.",
            5: "Negligible cost; many iterations affordable.",
        },
        "evidence": ["estimate", "measured", "quote"],
    },
    "risk_robustness": {
        "name": "Risk & Robustness",
        "definition": (
            "Likelihood and severity of failure, sensitivity to conditions, and "
            "behaviour at boundaries. Scored as ROBUSTNESS: 5 = robust, 0 = fragile."
        ),
        "boundary_note": (
            "Probability and blast radius of failure *given it is attempted*. "
            "Whether it is attemptable at all is Feasibility."
        ),
        "anchors": {
            0: "Known to fail under ordinary conditions, or unsafe.",
            1: "Fails outside a narrow happy path; failure is costly.",
            2: "Several identified failure modes without mitigations.",
            3: "Known failure modes with partial mitigations.",
            4: "Few failure modes, each mitigated and detectable.",
            5: "Degrades gracefully; failure modes bounded, detected and recoverable.",
        },
        "evidence": ["paper", "measured", "incident_report", "analysis"],
    },
    "testability": {
        "name": "Testability",
        "definition": (
            "Is there a clear, cheap, falsifiable way to find out whether this works, "
            "with an independent baseline?"
        ),
        "boundary_note": (
            "Quality of the *verification path*, independent of whether the proposal "
            "is good. An elegant but unfalsifiable idea scores 0 here."
        ),
        "anchors": {
            0: "No way to tell success from failure; unfalsifiable as stated.",
            1: "Only subjective or self-reported assessment possible.",
            2: "Testable in principle, but no baseline or acceptance criterion given.",
            3: "Concrete test exists; baseline weak or acceptance criterion vague.",
            4: "Concrete falsifiable test with an independent baseline.",
            5: "Cheap, fast, falsifiable test with independent baseline and a stated "
            "condition that would refute the proposal.",
        },
        "evidence": ["test_plan", "benchmark", "dataset"],
    },
    "novelty": {
        "name": "Novelty / Differentiation",
        "definition": (
            "Degree to which the approach differs from the known prior art or "
            "competing offerings in a way that matters to the goal."
        ),
        "boundary_note": (
            "Included ONLY when the contract's goal actually rewards novelty or "
            "differentiation. Anchors differ for research vs. commercial goals."
        ),
        "anchors": {
            0: "Identical to an existing approach, with no advantage.",
            1: "Cosmetic variation on prior art.",
            2: "Incremental variation with unclear advantage.",
            3: "Recognisably distinct approach with a plausible advantage.",
            4: "Substantially distinct with a clear, goal-relevant advantage.",
            5: "Materially new mechanism or position with a defensible advantage.",
        },
        "evidence": ["paper", "repo", "competitor_analysis", "patent"],
    },
}

#: Dimensions every rubric must carry regardless of domain.
CORE_DIMENSIONS = (
    "goal_alignment",
    "feasibility",
    "evidence_strength",
    "expected_benefit",
    "cost_time",
    "risk_robustness",
    "testability",
)

OPTIONAL_DIMENSIONS = ("novelty",)

#: Weight priors per goal_type. Keys are dimension keys; unlisted -> 1.0.
#: These are *priors over emphasis*, not truth claims; users may override.
_WEIGHT_PRIORS: dict[str, dict[str, float]] = {
    "technical_performance": {
        "goal_alignment": 1.6,
        "expected_benefit": 1.5,
        "evidence_strength": 1.2,
        "testability": 1.2,
        "feasibility": 1.0,
        "cost_time": 0.8,
        "risk_robustness": 0.9,
    },
    "scientific_validation": {
        "goal_alignment": 1.5,
        "evidence_strength": 1.8,
        "testability": 1.6,
        "feasibility": 1.0,
        "expected_benefit": 1.0,
        "risk_robustness": 0.8,
        "novelty": 1.2,
    },
    "product_utility": {
        "goal_alignment": 1.6,
        "expected_benefit": 1.4,
        "feasibility": 1.2,
        "cost_time": 1.2,
        "risk_robustness": 1.0,
        "evidence_strength": 1.0,
        "testability": 0.9,
        "novelty": 1.1,
    },
    "market_demand": {
        "goal_alignment": 1.4,
        "evidence_strength": 1.6,
        "expected_benefit": 1.3,
        "cost_time": 1.1,
        "testability": 1.1,
        "feasibility": 0.9,
        "risk_robustness": 0.9,
        "novelty": 1.2,
    },
    "cost_reduction": {
        "cost_time": 1.8,
        "goal_alignment": 1.4,
        "expected_benefit": 1.2,
        "feasibility": 1.1,
        "risk_robustness": 1.0,
        "evidence_strength": 1.0,
        "testability": 0.9,
    },
}

DEFAULT_PRIOR = "technical_performance"

#: goal_types for which novelty is goal-relevant by default.
_NOVELTY_RELEVANT = {"scientific_validation", "market_demand", "product_utility"}


def build_dimension(key: str, weight: float) -> Dimension:
    """Instantiate a catalogue dimension at a given weight."""
    if key not in _CATALOGUE:
        raise KeyError(f"unknown dimension {key!r}; known: {sorted(_CATALOGUE)}")
    spec = _CATALOGUE[key]
    return Dimension(
        key=key,
        name=spec["name"],
        definition=spec["definition"],
        anchors=dict(spec["anchors"]),
        weight=weight,
        acceptable_evidence_types=list(spec["evidence"]),
        higher_is_better=True,
        boundary_note=spec["boundary_note"],
    )


def build_rubric(
    contract: ResearchContract,
    *,
    version: str = "v1",
    include_optional: list[str] | None = None,
    weight_overrides: dict[str, float] | None = None,
) -> Rubric:
    """Derive a rubric from the contract deterministically.

    ``include_optional`` defaults to including ``novelty`` only when the goal type
    makes novelty goal-relevant, per the spec's requirement that novelty enter
    only when the goal rewards it.
    """
    prior_key = contract.goal_type if contract.goal_type in _WEIGHT_PRIORS else DEFAULT_PRIOR
    prior = _WEIGHT_PRIORS[prior_key]

    keys = list(CORE_DIMENSIONS)
    if include_optional is None:
        optional = (
            ["novelty"] if contract.goal_type in _NOVELTY_RELEVANT else []
        )
    else:
        optional = [k for k in include_optional if k in OPTIONAL_DIMENSIONS]
    keys.extend(k for k in optional if k not in keys)

    # Soft preferences nudge emphasis: a preference naming a dimension key raises it.
    nudges: dict[str, float] = {}
    for pref in contract.soft_preferences:
        for k in keys:
            if k in pref.key.lower() or k.replace("_", " ") in pref.description.lower():
                nudges[k] = nudges.get(k, 0.0) + (pref.importance - 3) * 0.1

    dims: list[Dimension] = []
    for k in keys:
        w = prior.get(k, 1.0) + nudges.get(k, 0.0)
        if weight_overrides and k in weight_overrides:
            w = weight_overrides[k]
        dims.append(build_dimension(k, max(w, 0.0)))

    rationale = (
        f"Weight prior '{prior_key}' selected from contract.goal_type="
        f"'{contract.goal_type}'. Optional dimensions included: {optional or 'none'}. "
        f"Soft-preference nudges: {nudges or 'none'}. "
        "Weights are an emphasis prior, not a measured quantity; override with "
        "`research-harness weights`."
    )
    return Rubric(
        version=version,
        run_id=contract.run_id,
        dimensions=dims,
        rationale=rationale,
        derived_from_contract_version=contract.version,
    )

