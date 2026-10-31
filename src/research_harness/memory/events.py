"""Append-only event log with replay protection (spec section 12).

Two jobs:

1. **Audit.** An immutable sequence of what happened, in the spec's event
   vocabulary, written to ``events.jsonl`` as it goes. Appended to, never
   rewritten — if a correction is needed it becomes a new event.

2. **Idempotent resume.** The spec requires that resuming a run must not repeat
   a completed action that had side effects. Actions register an
   ``idempotency_key``; :meth:`EventLog.already_done` answers whether that exact
   action has already completed, so a resumed run skips it instead of, say,
   running a paid experiment twice.

The key is the mechanism that makes resume safe, so it must be derived from
*what the action does* — not from a counter or a timestamp, which would differ
on replay and defeat the check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.schemas import EventType, ResearchEvent, utcnow


def idempotency_key(*parts: object) -> str:
    """Stable key for a side-effecting action.

    Built from the action's identity (what it is about), never from wall-clock
    time or a sequence number: those change between the original run and the
    resume, which is exactly when the check needs to match.
    """
    return "|".join(str(p) for p in parts if p is not None and str(p) != "")


class EventLog:
    """In-memory event sequence with an optional JSONL mirror."""

    def __init__(self, run_id: str, path: str | Path | None = None) -> None:
        self.run_id = run_id
        self.path = Path(path) if path else None
        self.events: list[ResearchEvent] = []
        self._keys: set[str] = set()

    # ------------------------------------------------------------------

    def append(
        self,
        event: EventType,
        summary: str,
        *,
        payload: dict[str, Any] | None = None,
        key: str | None = None,
    ) -> ResearchEvent:
        record = ResearchEvent(
            seq=len(self.events) + 1,
            run_id=self.run_id,
            event=event,
            summary=summary,
            payload=payload or {},
            idempotency_key=key,
            at=utcnow(),
        )
        self.events.append(record)
        if key:
            self._keys.add(key)
        self._flush(record)
        return record

    def already_done(self, key: str) -> bool:
        """True when an action with this key has already completed."""
        return key in self._keys

    def guard(self, key: str, *, action: str) -> bool:
        """Check-and-report helper for side-effecting actions.

        Returns True when the action should be skipped, logging the skip so the
        report can show that a resumed run deliberately did not repeat it.
        """
        if not self.already_done(key):
            return False
        self.append(
            EventType.ROUTE_DECIDED,
            f"Skipped '{action}': already completed in an earlier attempt of this "
            "run. Replaying a side-effecting action on resume is not safe, so it "
            "is skipped rather than repeated.",
            payload={"skipped_action": action, "key": key},
        )
        return True

    # ------------------------------------------------------------------

    def of_type(self, event: EventType) -> list[ResearchEvent]:
        return [e for e in self.events if e.event is event]

    def last(self, event: EventType) -> ResearchEvent | None:
        matches = self.of_type(event)
        return matches[-1] if matches else None

    def summary_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            out[e.event.value] = out.get(e.event.value, 0) + 1
        return out

    # ------------------------------------------------------------------

    def _flush(self, record: ResearchEvent) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def rewrite(self) -> None:
        """Write the whole log out fresh. Used once on resume, not per event."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(e.model_dump_json() + "\n")

    @classmethod
    def restore(
        cls, run_id: str, events: list[ResearchEvent], path: str | Path | None = None
    ) -> "EventLog":
        """Rebuild a log from persisted events, preserving idempotency keys.

        Called on resume. Without restoring the key set, a resumed run would
        believe no side effect had happened yet.
        """
        log = cls(run_id, path)
        log.events = list(events)
        log._keys = {e.idempotency_key for e in events if e.idempotency_key}
        return log

    @classmethod
    def load(cls, run_id: str, path: str | Path) -> "EventLog":
        p = Path(path)
        events: list[ResearchEvent] = []
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    events.append(ResearchEvent.model_validate(json.loads(line)))
        return cls.restore(run_id, events, p)
