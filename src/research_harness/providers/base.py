"""LLM provider abstraction.

Two concrete providers ship with the harness:

* :class:`~research_harness.providers.simulated.SimulatedProvider` -- offline and
  deterministic. ``is_simulated`` is ``True``, which every agent checks to take a
  clearly-separated, typed, local code path. Nothing it produces is research
  evidence, and it is labelled as such at the data level (``Provenance.SIMULATED``).
* :class:`~research_harness.providers.anthropic_provider.AnthropicProvider` --
  a real Claude model via the official SDK, producing schema-validated output.

Agents branch on ``provider.is_simulated`` rather than having the mock
reverse-engineer rendered prompts. The simulated path therefore operates on the
same typed inputs the real path does, and the two cannot silently diverge in what
they are given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """Raised for provider misconfiguration or an unusable model response."""


@dataclass
class Usage:
    """Accumulated spend for budget enforcement."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    per_purpose: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(
        self, *, purpose: str, input_tokens: int, output_tokens: int, cost_usd: float
    ) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost_usd
        self.calls += 1
        self.per_purpose[purpose] = self.per_purpose.get(purpose, 0) + 1


@runtime_checkable
class AgentProvider(Protocol):
    """Minimal contract an agent needs from a model backend."""

    provider_id: str
    is_simulated: bool
    usage: Usage

    def complete_structured(
        self, *, system: str, prompt: str, schema: type[T], purpose: str
    ) -> T:
        """Return an instance of ``schema``, validated.

        Implementations must raise :class:`ProviderError` rather than returning a
        partially-filled object, so a malformed model response cannot be mistaken
        for a real judgement.
        """
        ...
