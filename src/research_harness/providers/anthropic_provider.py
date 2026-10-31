"""Real Claude provider via the official Anthropic SDK.

**Verification status.** The request shaping, refusal handling, truncation
handling, schema validation, usage accounting and the legacy fallback path are
all covered by unit tests that inject a stub client (``tests/unit/
test_anthropic_provider.py``) -- so this module's *logic* is exercised even
without credentials. What remains untested here is the actual network round
trip and the real SDK's exact object shapes; see README "Known limitations".

Enable with::

    pip install anthropic
    export ANTHROPIC_API_KEY=...        # or: ant auth login
    research-harness run --provider anthropic --idea "..."
"""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel

from .base import ProviderError, Usage

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"

#: USD per million tokens, for budget accounting only. Update from the published
#: pricing page; a stale number here affects cost *reporting*, not correctness.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class _NeverMatches(Exception):
    """Placeholder so ``except`` clauses stay valid when the SDK is absent."""


def _sdk_errors(module: Any) -> tuple[type[BaseException], type[BaseException]]:
    """Resolve (APIStatusError, APIConnectionError) defensively.

    A stub client in tests has no such attributes, and an older SDK may rename
    them; falling back to a never-matching class keeps the ``except`` clauses
    valid instead of raising ``AttributeError`` at handler-setup time.
    """
    status = getattr(module, "APIStatusError", _NeverMatches)
    conn = getattr(module, "APIConnectionError", _NeverMatches)
    return status, conn


class AnthropicProvider:
    """Schema-validated structured completion backed by a real Claude model."""

    is_simulated = False

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 16_000,
        effort: str = "medium",
        api_key: str | None = None,
        timeout: float = 300.0,
        client: Any = None,
        sdk_module: Any = None,
    ) -> None:
        """``client`` / ``sdk_module`` exist for testing.

        When a client is injected the SDK is never imported, so the provider's
        logic can be tested in an environment that has neither the package nor
        credentials.
        """
        if client is not None:
            self.client = client
            self._anthropic = sdk_module
        else:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ProviderError(
                    "The 'anthropic' package is not installed. Run "
                    "`pip install anthropic`, or use --provider simulated."
                ) from exc

            self._anthropic = anthropic
            # A bare constructor also resolves an `ant auth login` profile, so an
            # unset ANTHROPIC_API_KEY does not necessarily mean "no credentials".
            kwargs: dict[str, object] = {"timeout": timeout}
            if api_key or os.environ.get("ANTHROPIC_API_KEY"):
                kwargs["api_key"] = api_key or os.environ["ANTHROPIC_API_KEY"]
            try:
                self.client = anthropic.Anthropic(**kwargs)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                raise ProviderError(
                    f"could not construct Anthropic client: {exc}"
                ) from exc

        self._status_error, self._conn_error = _sdk_errors(self._anthropic)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.provider_id = f"anthropic:{model}:effort={effort}"
        self.usage = Usage()

    # ------------------------------------------------------------------

    def build_request(self, *, system: str, prompt: str) -> dict[str, Any]:
        """The request body. Separated out so tests can assert its shape.

        No sampling parameters are sent: ``temperature`` / ``top_p`` / ``top_k``
        are rejected on current models. ``effort`` is the cost/quality dial.
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.effort},
        }

    def complete_structured(
        self, *, system: str, prompt: str, schema: type[T], purpose: str
    ) -> T:
        request = self.build_request(system=system, prompt=prompt)

        try:
            parsed, response = self._call(request, schema)
        except ProviderError:
            raise
        except self._status_error as exc:
            status = getattr(exc, "status_code", "unknown")
            raise ProviderError(
                f"Anthropic API error during {purpose}: {status} {exc}"
            ) from exc
        except self._conn_error as exc:
            raise ProviderError(
                f"could not reach the Anthropic API during {purpose}: {exc}"
            ) from exc

        # Check stop_reason BEFORE trusting content: a refusal or a truncated
        # response is reported, never quietly used as a judgement.
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            raise ProviderError(
                f"the model declined this request (purpose={purpose}, "
                f"category={getattr(details, 'category', None)}). Reported as a "
                "refusal rather than retried with a weakened prompt."
            )
        if stop == "max_tokens":
            raise ProviderError(
                f"response for {purpose} hit max_tokens ({self.max_tokens}) and is "
                "truncated; refusing to use a partial judgement."
            )

        self._record(response, purpose)
        return parsed

    # ------------------------------------------------------------------

    def _call(self, request: dict, schema: type[T]) -> tuple[T, Any]:
        """Prefer ``messages.parse``; fall back to raw JSON-schema mode."""
        parse = getattr(self.client.messages, "parse", None)
        if parse is not None:
            response = parse(output_format=schema, **request)
            parsed = getattr(response, "parsed_output", None)
            if parsed is None:
                raise ProviderError(
                    "messages.parse returned no parsed_output; the response did not "
                    "conform to the requested schema."
                )
            return parsed, response

        # Older SDK: constrain the format by hand and validate ourselves.
        req = dict(request)
        oc = dict(req.get("output_config") or {})
        oc["format"] = {"type": "json_schema", "schema": schema.model_json_schema()}
        req["output_config"] = oc
        response = self.client.messages.create(**req)

        text = next(
            (
                b.text
                for b in getattr(response, "content", [])
                if getattr(b, "type", None) == "text"
            ),
            None,
        )
        if text is None:
            raise ProviderError("no text block in the model response")
        try:
            return schema.model_validate(json.loads(text)), response
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(
                f"model response failed schema validation: {exc}"
            ) from exc

    def _record(self, response: object, purpose: str) -> None:
        u = getattr(response, "usage", None)
        inp = int(getattr(u, "input_tokens", 0) or 0)
        out = int(getattr(u, "output_tokens", 0) or 0)
        in_rate, out_rate = PRICING.get(self.model, (0.0, 0.0))
        cost = (inp / 1_000_000) * in_rate + (out / 1_000_000) * out_rate
        self.usage.add(
            purpose=purpose, input_tokens=inp, output_tokens=out, cost_usd=cost
        )
