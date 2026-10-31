"""Researcher: query generation, retrieval, de-duplication and cross-checking.

Produces :class:`Evidence` records that preserve where each claim came from.
Deliberate properties:

* Queries are generated from the contract (metrics, constraints, unknowns) and
  the domain profile, including explicit *disconfirming* queries -- looking only
  for support is how a harness talks itself into a bad plan.
* De-duplication is by normalised claim text, so the same finding arriving from
  two tools does not double-count as corroboration.
* Cross-checking is structural: a claim asserted by two or more *independent*
  sources is promoted to ``cross_checked``. Two simulated fixtures never
  corroborate each other.
"""

from __future__ import annotations

import re

from ..core.schemas import Evidence, Provenance, ResearchContract, Stance
from ..tools.search import ResearchTool, SearchHit
from . import domains


def _normalise(claim: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", claim.lower()).strip()[:160]


class Researcher:
    def __init__(self, provider, tools: list[ResearchTool]) -> None:
        self.provider = provider
        self.tools = tools

    # ------------------------------------------------------------------

    def plan_queries(self, contract: ResearchContract, *, limit: int = 10) -> list[str]:
        """Derive search queries from the contract. Deterministic."""
        profile = domains.select_profile(
            contract.original_idea, hint=contract.domain_hint
        )
        topic = contract.original_idea.strip()
        short = topic if len(topic) <= 90 else topic[:90].rsplit(" ", 1)[0]

        queries: list[str] = [t.format(topic=short) for t in profile.research_queries]

        for m in contract.success_metrics:
            queries.append(f"measured {m.description} for {short}")
        for c in contract.hard_constraints:
            if not c.assumed:
                queries.append(f"{c.description} requirements for {short}")
        for u in contract.unknowns[:2]:
            queries.append(f"{u} regarding {short}")

        # Disconfirming queries, always included.
        queries += [
            f"why {short} fails or does not work",
            f"criticism, negative results or retractions about {short}",
            f"hidden costs and operational burden of {short}",
        ]

        seen: set[str] = set()
        out: list[str] = []
        for q in queries:
            k = _normalise(q)
            if k and k not in seen:
                seen.add(k)
                out.append(q)
        return out[:limit]

    # ------------------------------------------------------------------

    def plan_targeted_queries(
        self, contract: ResearchContract, assumption: str, *, limit: int = 6
    ) -> list[str]:
        """Queries aimed at ONE unresolved assumption.

        Used by the research loop when the value-of-information shortlist names
        the assumption whose resolution would most change the decision. Searching
        the original broad queries again returns the same documents; asking about
        the specific open question is what can actually produce new evidence.

        Both confirming and disconfirming forms are issued, because finding only
        support for an assumption you are trying to test is not a test.
        """
        topic = contract.original_idea.strip()
        short = topic if len(topic) <= 70 else topic[:70].rsplit(" ", 1)[0]
        claim = assumption.strip().rstrip(".")
        if len(claim) > 120:
            claim = claim[:120].rsplit(" ", 1)[0]

        queries = [
            claim,
            f"evidence that {claim}",
            f"evidence against {claim}",
            f"{claim} in the context of {short}",
            f"cases where {claim} did not hold",
            f"how to verify {claim}",
        ]
        seen: set[str] = set()
        out: list[str] = []
        for q in queries:
            k = _normalise(q)
            if k and k not in seen:
                seen.add(k)
                out.append(q)
        return out[:limit]

    def gather_targeted(
        self, contract: ResearchContract, assumption: str, *, limit: int = 6
    ) -> tuple[list[Evidence], list[str]]:
        """Retrieve evidence for one assumption. Ids start at ``E001``.

        The caller is responsible for renumbering and de-duplicating against the
        existing pool (see ``Orchestrator._gather_more_evidence``).
        """
        queries = self.plan_targeted_queries(contract, assumption, limit=limit)
        return self._run_queries(contract, queries)

    # ------------------------------------------------------------------

    def gather(
        self, contract: ResearchContract, *, limit: int = 10
    ) -> tuple[list[Evidence], list[str]]:
        """Run every tool over every query. Returns ``(evidence, notes)``."""
        return self._run_queries(contract, self.plan_queries(contract, limit=limit))

    def _run_queries(
        self, contract: ResearchContract, queries: list[str]
    ) -> tuple[list[Evidence], list[str]]:
        context = {
            "run_id": contract.run_id,
            "goal_type": contract.goal_type,
            "domain": contract.domain_hint,
        }

        collected: dict[str, tuple[SearchHit, list[str]]] = {}
        notes: list[str] = []
        order: list[str] = []

        for query in queries:
            for tool in self.tools:
                try:
                    hits = tool.search(query, context)
                except Exception as exc:  # a broken tool must not abort research
                    notes.append(f"Tool {tool.tool_id} failed on {query!r}: {exc}")
                    continue
                for hit in hits:
                    key = _normalise(hit.claim)
                    if key in collected:
                        # Same claim from another tool: record the corroborator.
                        existing, sources = collected[key]
                        tag = f"{tool.tool_id}:{hit.source_uri or 'n/a'}"
                        if tag not in sources:
                            sources.append(tag)
                        if hit.quality > existing.quality:
                            collected[key] = (hit, sources)
                    else:
                        collected[key] = (
                            hit,
                            [f"{tool.tool_id}:{hit.source_uri or 'n/a'}"],
                        )
                        order.append(key)

        evidence: list[Evidence] = []
        for i, key in enumerate(order, start=1):
            hit, sources = collected[key]
            independent_real = [
                s for s in sources if not s.startswith("fixture-search")
            ]
            status = hit.verification_status
            limitations = list(hit.limitations)
            if len(independent_real) >= 2 and hit.provenance is not Provenance.SIMULATED:
                status = "cross_checked"
            elif hit.provenance is Provenance.SIMULATED:
                status = "unverified"
                if len(sources) > 1:
                    limitations.append(
                        "Appeared in multiple fixtures; simulated sources do not "
                        "corroborate one another."
                    )

            evidence.append(
                Evidence(
                    evidence_id=f"E{i:03d}",
                    claim=hit.claim,
                    source_type=hit.source_type,
                    source_uri=hit.source_uri,
                    applicability_conditions=hit.applicability_conditions,
                    stance=hit.stance,
                    provenance=hit.provenance,
                    verification_status=status,  # type: ignore[arg-type]
                    limitations=limitations,
                    quality=hit.quality,
                    related_metric=hit.related_metric,
                )
            )

        real = [e for e in evidence if e.is_real]
        if not real:
            notes.append(
                f"No non-simulated evidence was gathered across {len(queries)} queries. "
                "Evidence Strength will be scored UNKNOWN for every candidate, and no "
                "conclusion in this run is empirically grounded."
            )
        refuting = [e for e in evidence if e.stance is Stance.REFUTES]
        if real and not refuting:
            notes.append(
                "No refuting evidence was found. This is itself a warning sign: "
                "disconfirming queries were issued but returned nothing, which may "
                "mean the search is too narrow rather than that no objections exist."
            )
        return evidence, notes
