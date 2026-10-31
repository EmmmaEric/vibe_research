"""Offline deterministic provider.

Marks itself ``is_simulated = True``. Agents detect this and take their local,
typed simulation path; this class supplies only shared deterministic utilities
(seeded choice, stable hashing, the SIMULATED label).

Calling :meth:`complete_structured` is a programming error -- it means an agent
tried to reach a model while configured for offline operation -- so it raises
rather than inventing a response.
"""

from __future__ import annotations

import hashlib
from typing import TypeVar

from pydantic import BaseModel

from .base import ProviderError, Usage

T = TypeVar("T", bound=BaseModel)

#: Prefix stamped onto every human-readable string this provider influences.
SIMULATED_LABEL = "[SIMULATED]"


class SimulatedProvider:
    """Deterministic, offline, reproducible stand-in for a model backend."""

    provider_id = "simulated-v1"
    is_simulated = True

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.usage = Usage()

    # -- deterministic helpers -------------------------------------------

    def digest(self, *parts: str) -> int:
        """Stable integer hash of the inputs, salted by the run seed.

        Uses SHA-256 rather than ``hash()`` so results are identical across
        processes (Python salts ``str.__hash__`` per interpreter run, which would
        make "deterministic" output irreproducible between invocations).
        """
        blob = "|".join([str(self.seed), *parts]).encode()
        return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")

    def pick(self, options: list[str], *key: str) -> str:
        if not options:
            raise ValueError("pick() needs at least one option")
        return options[self.digest(*key) % len(options)]

    def jitter(self, *key: str, lo: float = 0.0, hi: float = 1.0) -> float:
        """Reproducible pseudo-random float in ``[lo, hi)``."""
        frac = (self.digest(*key) % 10_000) / 10_000.0
        return lo + frac * (hi - lo)

    # -- provider interface ----------------------------------------------

    def complete_structured(
        self, *, system: str, prompt: str, schema: type[T], purpose: str
    ) -> T:
        raise ProviderError(
            f"SimulatedProvider cannot serve a model call (purpose={purpose!r}). "
            "This means an agent took its LLM path while configured for offline "
            "operation. Either give the agent a simulated branch, or run with "
            "--provider anthropic."
        )
