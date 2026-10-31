"""OpenAI-compatible provider: schema transformation, tiering, error handling.

Tested with an injected transport, so no network and no credentials. The
schema-transformation tests encode what a live endpoint actually demanded:
every property in ``required``, even when ``strict`` is false.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from research_harness.providers.base import ProviderError
from research_harness.providers.openai_compatible import (
    TIER_JSON_OBJECT,
    TIER_JSON_SCHEMA,
    TIER_PROMPT_ONLY,
    OpenAICompatibleProvider,
    load_opencode_provider,
    strip_nulls,
    to_strict_schema,
)


class Nested(BaseModel):
    label: str
    weight: float = 1.0


class Answer(BaseModel):
    verdict: str
    score: int = 0
    tags: list[str] = Field(default_factory=list)
    note: str | None = None
    child: Nested | None = None


def _ok(content: str, *, usage: dict | None = None, finish: str = "stop") -> tuple:
    return 200, {
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


class Transport:
    """Records request bodies and replays scripted responses."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def __call__(self, body: dict):
        self.bodies.append(body)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def provider(transport, **kw) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="https://example.invalid/v1",
        api_key="sk-not-a-real-key",
        model="test-model",
        transport=transport,
        **kw,
    )


# --------------------------------------------------------------------------
# Schema transformation
# --------------------------------------------------------------------------


def test_every_property_becomes_required():
    """What a live endpoint demanded, even with strict=false."""
    strict = to_strict_schema(Answer.model_json_schema())
    assert set(strict["required"]) == set(strict["properties"])
    assert strict["additionalProperties"] is False


def test_nested_definitions_are_transformed_too():
    strict = to_strict_schema(Answer.model_json_schema())
    nested = strict["$defs"]["Nested"]
    assert set(nested["required"]) == {"label", "weight"}
    assert nested["additionalProperties"] is False


def test_originally_optional_fields_become_nullable():
    """So the model has a legal way to say 'not applicable'."""
    props = to_strict_schema(Answer.model_json_schema())["properties"]

    # `verdict` was required: its type is untouched.
    assert props["verdict"]["type"] == "string"
    # `score` had a default: null is now allowed alongside integer.
    assert "null" in props["score"]["type"]
    assert "integer" in props["score"]["type"]
    # A $ref cannot carry a sibling type, so it is wrapped in anyOf.
    assert any(
        b.get("type") == "null" for b in props["child"]["anyOf"] if isinstance(b, dict)
    )


def test_already_nullable_fields_are_not_double_wrapped():
    before = Answer.model_json_schema()["properties"]["note"]
    after = to_strict_schema(Answer.model_json_schema())["properties"]["note"]
    nulls = [b for b in after.get("anyOf", []) if b.get("type") == "null"]
    assert len(nulls) == 1, f"null added twice to an already-optional field: {before}"


def test_transformation_does_not_mutate_the_source_schema():
    original = Answer.model_json_schema()
    snapshot = json.dumps(original, sort_keys=True)
    to_strict_schema(original)
    assert json.dumps(original, sort_keys=True) == snapshot


def test_nulls_are_stripped_so_pydantic_defaults_apply():
    """A returned null means 'nothing to say', not None."""
    data = {"verdict": "A", "score": None, "tags": None, "child": None}
    cleaned = strip_nulls(data)
    assert cleaned == {"verdict": "A"}
    parsed = Answer.model_validate(cleaned)
    assert parsed.score == 0
    assert parsed.tags == []  # a list[str] field would reject None outright


def test_nulls_are_stripped_recursively():
    cleaned = strip_nulls({"a": [{"b": None, "c": 1}], "d": {"e": None}})
    assert cleaned == {"a": [{"c": 1}], "d": {}}


# --------------------------------------------------------------------------
# Tiering
# --------------------------------------------------------------------------


def test_json_schema_tier_is_tried_first_and_remembered():
    transport = Transport(_ok('{"verdict":"A","score":3}'))
    p = provider(transport)

    assert p.tier is None
    result = p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")
    assert result.verdict == "A"
    assert p.tier == TIER_JSON_SCHEMA

    fmt = transport.bodies[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert set(fmt["json_schema"]["schema"]["required"]) == set(
        fmt["json_schema"]["schema"]["properties"]
    )

    # A second call does not re-probe.
    p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")
    assert len(transport.bodies) == 2
    assert transport.bodies[1]["response_format"]["type"] == "json_schema"


def test_a_rejected_schema_falls_back_to_json_object():
    rejection = (400, {"error": {"message": "Invalid schema for response_format"}})
    transport = Transport(rejection, _ok('{"verdict":"B"}'))
    p = provider(transport)

    result = p.complete_structured(system="sys", prompt="p", schema=Answer, purpose="t")
    assert result.verdict == "B"
    assert p.tier == TIER_JSON_OBJECT
    assert transport.bodies[1]["response_format"] == {"type": "json_object"}
    # The weaker tier has to carry the schema in the prompt instead.
    assert "JSON Schema" in transport.bodies[1]["messages"][0]["content"]
    assert transport.bodies[1]["messages"][0]["content"].startswith("sys")


def test_both_response_format_tiers_rejected_falls_back_to_prompt_only():
    rejection = (400, {"error": {"message": "response_format unsupported"}})
    transport = Transport(rejection, rejection, _ok('{"verdict":"C"}'))
    p = provider(transport)

    result = p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")
    assert result.verdict == "C"
    assert p.tier == TIER_PROMPT_ONLY
    assert "response_format" not in transport.bodies[2]


def test_all_tiers_rejected_surfaces_the_endpoint_message():
    """The endpoint's own error is the actionable information, so it wins.

    A 400 on the last (plain-completion) tier is not a response_format problem
    any more -- it is a genuinely bad request -- so the endpoint's text is
    reported rather than a generic "no mode accepted".
    """
    rejection = (400, {"error": {"message": "nope, bad schema"}})
    p = provider(Transport(rejection))
    with pytest.raises(ProviderError, match="nope, bad schema"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")


# --------------------------------------------------------------------------
# Response handling: nothing partial is ever used
# --------------------------------------------------------------------------


def test_truncated_response_is_refused():
    p = provider(Transport(_ok('{"verdict":"A"}', finish="length")))
    with pytest.raises(ProviderError, match="truncated"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="judge")
    assert p.usage.calls == 0


def test_a_refusal_is_reported_not_parsed():
    transport = Transport(
        (200, {
            "choices": [
                {"finish_reason": "stop", "message": {"refusal": "I cannot help"}}
            ],
            "usage": {},
        })
    )
    with pytest.raises(ProviderError, match="declined this request"):
        provider(transport).complete_structured(
            system="s", prompt="p", schema=Answer, purpose="judge"
        )


def test_empty_content_names_the_reasoning_budget_as_a_likely_cause():
    p = provider(Transport(_ok("   ")))
    with pytest.raises(ProviderError, match="reasoning"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")


def test_invalid_json_is_rejected_with_a_sample():
    p = provider(Transport(_ok("not json at all")))
    with pytest.raises(ProviderError, match="not valid JSON"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")


def test_schema_violation_is_rejected_rather_than_half_filled():
    p = provider(Transport(_ok('{"score": 3}')))  # `verdict` missing
    with pytest.raises(ProviderError, match="did not match Answer"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")


def test_a_markdown_fence_is_tolerated():
    p = provider(Transport(_ok('```json\n{"verdict":"A","score":2}\n```')))
    result = p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")
    assert result.verdict == "A"
    assert result.score == 2


def test_missing_choices_is_an_error():
    p = provider(Transport((200, {"choices": [], "usage": {}})))
    with pytest.raises(ProviderError, match="no choices"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")


def test_a_non_retryable_http_error_surfaces():
    p = provider(Transport((401, {"error": {"message": "bad key"}})))
    with pytest.raises(ProviderError, match="HTTP 401"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")


# --------------------------------------------------------------------------
# Retries and accounting
# --------------------------------------------------------------------------


def test_rate_limiting_is_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    transport = Transport((429, {"error": {"message": "slow down"}}), _ok('{"verdict":"A"}'))
    p = provider(transport, max_retries=2)
    assert p.complete_structured(
        system="s", prompt="p", schema=Answer, purpose="t"
    ).verdict == "A"
    assert len(transport.bodies) >= 2


def test_retries_are_bounded(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    transport = Transport((503, {"error": {"message": "down"}}))
    p = provider(transport, max_retries=2)
    with pytest.raises(ProviderError, match="HTTP 503"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")
    assert len(transport.bodies) == 3  # one attempt plus two retries


def test_usage_and_reasoning_tokens_are_tracked():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "completion_tokens_details": {"reasoning_tokens": 30},
    }
    p = provider(Transport(_ok('{"verdict":"A"}', usage=usage)))
    p.complete_structured(system="s", prompt="p", schema=Answer, purpose="judge")

    assert p.usage.input_tokens == 100
    assert p.usage.output_tokens == 50
    assert p.usage.total_tokens == 150
    assert p.reasoning_tokens == 30
    assert p.usage.per_purpose == {"judge": 1}
    # Per-token prices for an arbitrary proxy are unknown, so cost stays 0.0
    # rather than being invented.
    assert p.usage.cost_usd == 0.0


def test_the_api_key_never_appears_in_the_request_body():
    transport = Transport(_ok('{"verdict":"A"}'))
    p = provider(transport)
    p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")
    assert "sk-not-a-real-key" not in json.dumps(transport.bodies[0])


def test_no_sampling_parameters_are_sent():
    transport = Transport(_ok('{"verdict":"A"}'))
    provider(transport).complete_structured(
        system="s", prompt="p", schema=Answer, purpose="t"
    )
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in transport.bodies[0]


# --------------------------------------------------------------------------
# Credential loading
# --------------------------------------------------------------------------


def _config(tmp_path, payload: dict):
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_opencode_config_is_read(tmp_path):
    path = _config(
        tmp_path,
        {
            "provider": {
                "local": {
                    "options": {"baseURL": "https://x/v1/", "apiKey": "sk-abc"},
                    "models": {"gpt-test": {"name": "GPT"}},
                }
            }
        },
    )
    base_url, key, model = load_opencode_provider(
        config_path=path, provider_key="local", model="gpt-test"
    )
    assert base_url == "https://x/v1"  # trailing slash normalised
    assert key == "sk-abc"
    assert model == "gpt-test"


def test_the_first_declared_model_is_the_default(tmp_path):
    path = _config(
        tmp_path,
        {
            "provider": {
                "local": {
                    "options": {"baseURL": "https://x/v1", "apiKey": "k"},
                    "models": {"first": {}, "second": {}},
                }
            }
        },
    )
    assert load_opencode_provider(config_path=path)[2] == "first"


def test_an_undeclared_model_is_refused(tmp_path):
    path = _config(
        tmp_path,
        {
            "provider": {
                "local": {
                    "options": {"baseURL": "https://x/v1", "apiKey": "k"},
                    "models": {"only-this": {}},
                }
            }
        },
    )
    with pytest.raises(ProviderError, match="not declared"):
        load_opencode_provider(config_path=path, model="something-else")


def test_an_unknown_provider_key_lists_what_exists(tmp_path):
    path = _config(tmp_path, {"provider": {"local": {}, "other": {}}})
    with pytest.raises(ProviderError, match="available"):
        load_opencode_provider(config_path=path, provider_key="missing")


def test_a_provider_without_credentials_is_refused(tmp_path):
    path = _config(
        tmp_path, {"provider": {"local": {"options": {"baseURL": "https://x"}}}}
    )
    with pytest.raises(ProviderError, match="missing baseURL or apiKey"):
        load_opencode_provider(config_path=path)


def test_a_missing_config_file_is_reported(tmp_path):
    with pytest.raises(ProviderError, match="not found"):
        load_opencode_provider(config_path=tmp_path / "nope.json")


def test_malformed_config_is_reported(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProviderError, match="could not parse"):
        load_opencode_provider(config_path=path)
