"""OpenAI-compatible chat-completions provider.

For any endpoint speaking the OpenAI `/v1/chat/completions` shape — including
proxies and self-hosted gateways. Verified against a live
`@ai-sdk/openai-compatible` endpoint serving a reasoning model.

**Why stdlib ``urllib`` and not a client library.** One POST with a JSON body
does not justify a dependency (ADR-011), and these endpoints vary enough that a
vendor SDK's assumptions get in the way more than they help.

**Structured output is tiered, because endpoints disagree.** Probing a real
proxy showed that `response_format: json_schema` is rejected unless *every*
property appears in `required` — enforced even with `strict: false`, which is
not what Pydantic emits. So:

1. ``json_schema`` with a transformed schema (all properties required, the
   originally-optional ones made nullable). Strongest guarantee.
2. ``json_object`` with the schema in the system prompt.
3. Plain completion with the schema in the prompt.

The working tier is remembered per instance, so the cost of discovery is paid
once. Every tier validates with Pydantic on our side and raises
:class:`ProviderError` rather than returning a half-filled object — a malformed
response must never be mistaken for a judgement.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .base import ProviderError, Usage

T = TypeVar("T", bound=BaseModel)

#: Tier names, strongest first.
TIER_JSON_SCHEMA = "json_schema"
TIER_JSON_OBJECT = "json_object"
TIER_PROMPT_ONLY = "prompt_only"
TIERS = (TIER_JSON_SCHEMA, TIER_JSON_OBJECT, TIER_PROMPT_ONLY)

RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


# --------------------------------------------------------------------------
# Schema transformation
# --------------------------------------------------------------------------


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a Pydantic JSON schema for strict structured output.

    Two changes, applied to every object in the tree including ``$defs``:

    * ``required`` lists every property. Endpoints enforcing OpenAI strict
      semantics reject anything less.
    * a property that was *not* originally required becomes nullable, so the
      model has a legal way to say "not applicable" instead of inventing a
      value. :func:`strip_nulls` removes those nulls before validation, letting
      the Pydantic default apply.

    ``additionalProperties: false`` is set so an endpoint cannot pad the object
    with fields the schema never declared.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node

        out = {k: walk(v) for k, v in node.items()}

        if out.get("type") == "object" and isinstance(out.get("properties"), dict):
            props = out["properties"]
            originally_required = set(out.get("required") or [])
            for name, prop in props.items():
                if name not in originally_required:
                    props[name] = _make_nullable(prop)
            out["required"] = list(props)
            out["additionalProperties"] = False
        return out

    return walk(schema)


def _make_nullable(prop: dict[str, Any]) -> dict[str, Any]:
    """Allow null for a property, without discarding its original type."""
    if not isinstance(prop, dict):
        return prop
    if "anyOf" in prop:
        branches = prop["anyOf"]
        if not any(b.get("type") == "null" for b in branches if isinstance(b, dict)):
            prop = {**prop, "anyOf": [*branches, {"type": "null"}]}
        return prop
    if "$ref" in prop:
        # A $ref cannot carry a sibling type, so wrap it.
        ref = {k: v for k, v in prop.items() if k != "default"}
        return {"anyOf": [ref, {"type": "null"}]}
    if "type" in prop:
        t = prop["type"]
        types = t if isinstance(t, list) else [t]
        if "null" not in types:
            return {**prop, "type": [*types, "null"]}
        return prop
    return prop


def strip_nulls(value: Any) -> Any:
    """Drop null-valued keys so Pydantic defaults apply.

    Needed because :func:`to_strict_schema` makes optional fields nullable: a
    returned ``null`` means "nothing to say", not ``None`` — and a field typed
    ``list[str]`` would reject ``None`` outright.
    """
    if isinstance(value, dict):
        return {k: strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [strip_nulls(v) for v in value]
    return value


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------


def load_opencode_provider(
    *,
    config_path: str | Path | None = None,
    provider_key: str = "local",
    model: str | None = None,
) -> tuple[str, str, str]:
    """Read ``base_url, api_key, model`` from an opencode config file.

    Returns the key for immediate use; it is never logged, stored in run state,
    or written to an artifact. ``redacted_effective_config.json`` records only
    that a credential was present.
    """
    path = Path(config_path or (Path.home() / ".config" / "opencode" / "opencode.json"))
    if not path.is_file():
        raise ProviderError(f"opencode config not found: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"could not parse {path}: {exc}") from exc

    providers = cfg.get("provider") or {}
    if provider_key not in providers:
        raise ProviderError(
            f"provider '{provider_key}' not in {path}; available: "
            f"{sorted(providers)}"
        )
    entry = providers[provider_key]
    opts = entry.get("options") or {}
    base_url = (opts.get("baseURL") or "").rstrip("/")
    api_key = opts.get("apiKey") or ""
    if not base_url or not api_key:
        raise ProviderError(
            f"provider '{provider_key}' in {path} is missing baseURL or apiKey"
        )

    models = entry.get("models") or {}
    chosen = model or (next(iter(models)) if models else None)
    if chosen is None:
        raise ProviderError(f"provider '{provider_key}' declares no models")
    if models and chosen not in models:
        raise ProviderError(
            f"model '{chosen}' not declared for provider '{provider_key}'; "
            f"available: {sorted(models)}"
        )
    return base_url, api_key, chosen


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------


class OpenAICompatibleProvider:
    """Structured completion against an OpenAI-compatible endpoint."""

    is_simulated = False

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = 16_000,
        timeout: float = 300.0,
        max_retries: int = 2,
        label: str | None = None,
        transport=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.usage = Usage()
        #: Cost reporting needs per-model prices we do not have for a proxy, so
        #: spend is tracked in tokens and cost stays 0.0 rather than invented.
        self.provider_id = label or f"openai-compatible:{model}"
        self._transport = transport or self._http_post
        self._tier: str | None = None
        self.reasoning_tokens = 0

    # ------------------------------------------------------------------

    @property
    def tier(self) -> str | None:
        """Which structured-output tier is in use, once discovered."""
        return self._tier

    def complete_structured(
        self, *, system: str, prompt: str, schema: type[T], purpose: str
    ) -> T:
        tiers = TIERS if self._tier is None else (self._tier,)
        last_error: str = ""

        for tier in tiers:
            body = self._build_body(system=system, prompt=prompt, schema=schema, tier=tier)
            status, payload = self._request(body, purpose=purpose)

            if status == 400 and tier != TIER_PROMPT_ONLY:
                # The endpoint rejected this response_format. Try a weaker tier
                # rather than failing: proxies differ in what they accept.
                last_error = self._error_text(payload)
                continue
            if status != 200:
                raise ProviderError(
                    f"{self.provider_id} returned HTTP {status} during {purpose}: "
                    f"{self._error_text(payload)}"
                )

            parsed = self._parse(payload, schema=schema, purpose=purpose)
            if self._tier != tier:
                self._tier = tier
            self._record(payload, purpose)
            return parsed

        raise ProviderError(
            f"no structured-output mode was accepted by {self.provider_id} for "
            f"{purpose}. Last error: {last_error}"
        )

    # ------------------------------------------------------------------

    def _build_body(
        self, *, system: str, prompt: str, schema: type[T], tier: str
    ) -> dict[str, Any]:
        raw = schema.model_json_schema()
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }

        if tier == TIER_JSON_SCHEMA:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": to_strict_schema(raw),
                },
            }
            return body

        # Weaker tiers: the schema has to travel in the prompt.
        instruction = (
            "\n\nReturn ONLY a JSON object conforming to this JSON Schema. No "
            "prose, no markdown fence.\n" + json.dumps(raw)
        )
        body["messages"][0]["content"] = system + instruction
        if tier == TIER_JSON_OBJECT:
            body["response_format"] = {"type": "json_object"}
        return body

    def _request(self, body: dict, *, purpose: str) -> tuple[int | None, Any]:
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            status, payload = self._transport(body)
            if status in RETRY_STATUS and attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            return status, payload
        return status, payload  # pragma: no cover - loop always returns

    def _http_post(self, body: dict) -> tuple[int | None, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            try:
                return exc.code, json.loads(text)
            except json.JSONDecodeError:
                return exc.code, text
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"could not reach {self.base_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                f"{self.base_url} did not respond within {self.timeout}s"
            ) from exc

    # ------------------------------------------------------------------

    def _parse(self, payload: Any, *, schema: type[T], purpose: str) -> T:
        if not isinstance(payload, dict):
            raise ProviderError(f"non-JSON response during {purpose}")
        choices = payload.get("choices") or []
        if not choices:
            raise ProviderError(f"response contained no choices during {purpose}")

        choice = choices[0]
        finish = choice.get("finish_reason")
        message = choice.get("message") or {}
        content = message.get("content")

        # A refusal or a truncation is reported, never used as a judgement.
        if message.get("refusal"):
            raise ProviderError(
                f"the model declined this request (purpose={purpose}): "
                f"{message['refusal']}"
            )
        if finish == "length":
            raise ProviderError(
                f"response for {purpose} hit max_tokens ({self.max_tokens}) and is "
                "truncated; refusing to use a partial judgement."
            )
        if not content or not str(content).strip():
            raise ProviderError(
                f"empty content during {purpose} (finish_reason={finish}). A "
                "reasoning model may have spent the whole budget on reasoning; "
                "raise max_tokens."
            )

        text = _unfence(str(content))
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"response for {purpose} was not valid JSON: {exc}. "
                f"First 200 chars: {text[:200]!r}"
            ) from exc
        try:
            return schema.model_validate(strip_nulls(data))
        except ValidationError as exc:
            raise ProviderError(
                f"response for {purpose} did not match {schema.__name__}: "
                f"{exc.error_count()} validation error(s); first: "
                f"{exc.errors()[0].get('loc')} {exc.errors()[0].get('msg')}"
            ) from exc

    def _record(self, payload: dict, purpose: str) -> None:
        usage = payload.get("usage") or {}
        inp = int(usage.get("prompt_tokens") or 0)
        out = int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
        # Cost is left at 0.0: per-token prices for an arbitrary proxy are not
        # known, and an invented figure is worse than an absent one.
        self.usage.add(purpose=purpose, input_tokens=inp, output_tokens=out, cost_usd=0.0)

    @staticmethod
    def _error_text(payload: Any) -> str:
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)[:400]
            return str(payload)[:400]
        return str(payload)[:400]


def _unfence(text: str) -> str:
    """Strip a ```json fence if the model added one despite instructions."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()


def from_opencode(
    *,
    config_path: str | Path | None = None,
    provider_key: str = "local",
    model: str | None = None,
    **kwargs,
) -> OpenAICompatibleProvider:
    """Build a provider from an opencode config entry."""
    base_url, api_key, chosen = load_opencode_provider(
        config_path=config_path, provider_key=provider_key, model=model
    )
    return OpenAICompatibleProvider(
        base_url=base_url,
        api_key=api_key,
        model=chosen,
        label=f"openai-compatible:{chosen}",
        **kwargs,
    )


def from_env(**kwargs) -> OpenAICompatibleProvider:
    """Build a provider from ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / ``OPENAI_MODEL``."""
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    missing = [
        name
        for name, value in (
            ("OPENAI_BASE_URL", base_url),
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise ProviderError(f"missing environment variable(s): {', '.join(missing)}")
    return OpenAICompatibleProvider(
        base_url=base_url, api_key=api_key, model=model, **kwargs
    )
