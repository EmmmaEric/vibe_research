"""Research memory retrieval.

Text search over prior runs' evidence, proposals and contracts, so past work
(including failures) can be found and reused. FTS5 when the SQLite build has it,
LIKE otherwise -- the degraded path is slower and cruder, but present.

Deliberately no vector store: at this scale it would add a dependency and a
model call per query for a retrieval problem that keyword search handles. It is
a clean extension point, not a prerequisite.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .repository import Repository


@dataclass
class Hit:
    run_id: str
    kind: str
    ref_id: str
    text: str
    score: float = 0.0


def _sanitize(query: str) -> str:
    """Make a user string safe as an FTS5 MATCH expression.

    FTS5 treats characters like ``"``, ``*``, ``:``, ``-`` and parentheses as
    operators, so an unescaped query raises ``OperationalError``. We reduce the
    query to bare terms and quote each one.
    """
    terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
    return " OR ".join(f'"{t}"' for t in terms)


class MemoryRetrieval:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def search(
        self, query: str, *, kinds: tuple[str, ...] | None = None, limit: int = 20
    ) -> list[Hit]:
        if not query.strip():
            return []
        if self.repo.fts_available:
            hits = self._fts(query, limit)
            if hits or _sanitize(query):
                return self._filter(hits, kinds)
        return self._filter(self._like(query, limit), kinds)

    @staticmethod
    def _filter(hits: list[Hit], kinds: tuple[str, ...] | None) -> list[Hit]:
        return [h for h in hits if kinds is None or h.kind in kinds]

    def _fts(self, query: str, limit: int) -> list[Hit]:
        expr = _sanitize(query)
        if not expr:
            return []
        try:
            rows = self.repo.conn.execute(
                "SELECT run_id, kind, ref_id, text, bm25(search_index) AS rank "
                "FROM search_index WHERE search_index MATCH ? "
                "ORDER BY rank LIMIT ?",
                (expr, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25 returns lower-is-better; negate so higher is more relevant.
        return [
            Hit(
                run_id=r["run_id"],
                kind=r["kind"],
                ref_id=r["ref_id"],
                text=r["text"],
                score=-float(r["rank"] or 0.0),
            )
            for r in rows
        ]

    def _like(self, query: str, limit: int) -> list[Hit]:
        """Fallback search for SQLite builds without FTS5."""
        terms = [t for t in query.lower().split() if len(t) > 2][:6]
        if not terms:
            return []
        patterns = [f"%{t}%" for t in terms]

        # (kind, select clause, text column) -- the text column is interpolated
        # into the WHERE clause, so it must come from this literal table only,
        # never from user input.
        sources = (
            ("evidence", "SELECT run_id, evidence_id, claim FROM evidence", "claim"),
            ("proposal", "SELECT run_id, proposal_id, mechanism FROM proposals", "mechanism"),
            ("contract", "SELECT run_id, 'contract', original_idea FROM runs", "original_idea"),
        )

        hits: list[Hit] = []
        for kind, select, text_col in sources:
            where = " OR ".join([f"LOWER({text_col}) LIKE ?"] * len(terms))
            try:
                rows = self.repo.conn.execute(
                    f"{select} WHERE {where} LIMIT ?", [*patterns, limit]
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            for row in rows:
                run_id, ref_id, text = tuple(row)
                hits.append(
                    Hit(
                        run_id=run_id,
                        kind=kind,
                        ref_id=str(ref_id),
                        text=str(text or ""),
                        score=1.0,
                    )
                )
        return hits[:limit]

    def similar_prior_runs(self, idea: str, *, limit: int = 5) -> list[Hit]:
        """Prior runs on a related idea -- including the ones that failed.

        A past run that concluded INCONCLUSIVE or REJECTED is often the most
        valuable thing in the memory, so nothing is filtered by outcome.
        """
        return [h for h in self.search(idea, kinds=("contract",), limit=limit)]
