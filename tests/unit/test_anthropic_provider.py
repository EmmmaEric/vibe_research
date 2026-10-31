"""AnthropicProvider logic, tested with an injected stub client.

No network and no credentials: a stub stands in for the SDK client, so the
provider's own behaviour -- request shaping, refusal and truncation handling,
schema validation, usage accounting, and the legacy fallback path -- is really
exercised rather than merely written.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from research_harness.providers.anthropic_provider import (
    PRICING,
    AnthropicProvider,
)
from research_harness.providers.base import ProviderError


class Answer(BaseModel):
    verdict: str
    score: int


class _Usage:
    def __init__(self, inp: int = 1000, out: int = 500) -> None:
        self.input_tokens = inp
        self.output_tokens = out


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(
        self,
        *,
        parsed=None,
        text: str | None = None,
        stop_reason: str = "end_turn",
        stop_details=None,
        usage=None,
    ) -> None:
        self.parsed_output = parsed
        self.content = [_Block(text)] if text is not None else []
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.usage = usage or _Usage()


class _Messages:
    """Stub messages namespace. ``supports_parse=False`` mimics an older SDK."""

    def __init__(self, response, *, supports_parse: bool = True, raiser=None) -> None:
        self._response = response
        self._raiser = raiser
        self.calls: list[dict] = []
        if not supports_parse:
            # How the provider detects a legacy SDK: no usable `parse`.
            self.parse = None

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._raiser:
            raise self._raiser
        return self._response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raiser:
            raise self._raiser
        return self._response


class _Client:
    def __init__(self, response, *, supports_parse: bool = True, raiser=None) -> None:
        self.messages = _Messages(
            response, supports_parse=supports_parse, raiser=raiser
        )


def provider(client, **kw) -> AnthropicProvider:
    return AnthropicProvider(client=client, **kw)


# --------------------------------------------------------------------------
# Request shaping
# --------------------------------------------------------------------------


def test_request_sends_effort_and_no_sampling_parameters():
    """Sampling params are rejected on current models; effort is the dial."""
    p = provider(_Client(_Response(parsed=Answer(verdict="A", score=3))))
    req = p.build_request(system="sys", prompt="ask")

    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "medium"}
    assert req["system"] == "sys"
    assert req["messages"] == [{"role": "user", "content": "ask"}]
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in req


def test_effort_and_model_are_configurable_and_reported_in_provider_id():
    p = provider(
        _Client(_Response(parsed=Answer(verdict="A", score=1))),
        model="claude-sonnet-5-5",
        effort="high",
    )
    assert p.build_request(system="s", prompt="p")["output_config"]["effort"] == "high"
    assert p.provider_id == "anthropic:claude-sonnet-5-5:effort=high"
    assert p.is_simulated is False


def test_parse_path_passes_the_schema_as_output_format():
    client = _Client(_Response(parsed=Answer(verdict="B", score=5)))
    p = provider(client)
    result = p.complete_structured(
        system="s", prompt="p", schema=Answer, purpose="t"
    )
    assert isinstance(result, Answer)
    assert result.verdict == "B"
    assert client.messages.calls[0]["output_format"] is Answer


# --------------------------------------------------------------------------
# Refusal and truncation are reported, never used
# --------------------------------------------------------------------------


def test_refusal_is_raised_and_not_retried_with_a_weakened_prompt():
    class _Details:
        category = "cyber"

    client = _Client(
        _Response(
            parsed=Answer(verdict="A", score=3),
            stop_reason="refusal",
            stop_details=_Details(),
        )
    )
    p = provider(client)
    with pytest.raises(ProviderError, match="declined this request"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="judge")

    assert len(client.messages.calls) == 1, "a refusal must not trigger a retry"
    assert p.usage.calls == 0, "a refused call must not be counted as a judgement"


def test_refusal_message_names_the_purpose_and_category():
    class _Details:
        category = "bio"

    p = provider(
        _Client(
            _Response(
                parsed=Answer(verdict="A", score=1),
                stop_reason="refusal",
                stop_details=_Details(),
            )
        )
    )
    with pytest.raises(ProviderError) as exc:
        p.complete_structured(
            system="s", prompt="p", schema=Answer, purpose="skeptic.critique"
        )
    assert "skeptic.critique" in str(exc.value)
    assert "bio" in str(exc.value)


def test_truncated_response_is_refused_rather_than_partially_used():
    p = provider(
        _Client(
            _Response(parsed=Answer(verdict="A", score=3), stop_reason="max_tokens")
        )
    )
    with pytest.raises(ProviderError, match="truncated"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")
    assert p.usage.calls == 0


def test_missing_parsed_output_is_an_error_not_a_blank_object():
    p = provider(_Client(_Response(parsed=None)))
    with pytest.raises(ProviderError, match="no parsed_output"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")


# --------------------------------------------------------------------------
# Legacy JSON-schema fallback
# --------------------------------------------------------------------------


def test_falls_back_to_json_schema_mode_when_parse_is_unavailable():
    payload = json.dumps({"verdict": "A", "score": 4})
    client = _Client(_Response(text=payload), supports_parse=False)
    p = provider(client)

    result = p.complete_structured(
        system="s", prompt="p", schema=Answer, purpose="t"
    )
    assert result == Answer(verdict="A", score=4)

    sent = client.messages.calls[0]
    fmt = sent["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"].keys() == {"verdict", "score"}
    # The effort setting survives the fallback.
    assert sent["output_config"]["effort"] == "medium"


def test_fallback_rejects_malformed_json():
    client = _Client(_Response(text="not json at all"), supports_parse=False)
    with pytest.raises(ProviderError, match="failed schema validation"):
        provider(client).complete_structured(
            system="s", prompt="p", schema=Answer, purpose="t"
        )


def test_fallback_rejects_json_that_violates_the_schema():
    client = _Client(
        _Response(text=json.dumps({"verdict": "A"})), supports_parse=False
    )
    with pytest.raises(ProviderError, match="failed schema validation"):
        provider(client).complete_structured(
            system="s", prompt="p", schema=Answer, purpose="t"
        )


def test_fallback_errors_when_there_is_no_text_block():
    client = _Client(_Response(), supports_parse=False)
    with pytest.raises(ProviderError, match="no text block"):
        provider(client).complete_structured(
            system="s", prompt="p", schema=Answer, purpose="t"
        )


# --------------------------------------------------------------------------
# Usage accounting drives the budget, so it has to be right
# --------------------------------------------------------------------------


def test_usage_and_cost_are_accumulated_per_call():
    client = _Client(
        _Response(parsed=Answer(verdict="A", score=3), usage=_Usage(1_000_000, 100_000))
    )
    p = provider(client, model="claude-opus-5-5")
    p.complete_structured(system="s", prompt="p", schema=Answer, purpose="judge")

    in_rate, out_rate = PRICING["claude-opus-5-5"]
    expected = in_rate + 0.1 * out_rate

    assert p.usage.input_tokens == 1_000_000
    assert p.usage.output_tokens == 100_000
    assert p.usage.total_tokens == 1_100_000
    assert p.usage.calls == 1
    assert p.usage.cost_usd == pytest.approx(expected)
    assert p.usage.per_purpose == {"judge": 1}


def test_usage_accumulates_across_calls_and_purposes():
    p = provider(_Client(_Response(parsed=Answer(verdict="A", score=3))))
    for purpose in ("judge", "judge", "designer"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose=purpose)
    assert p.usage.calls == 3
    assert p.usage.per_purpose == {"judge": 2, "designer": 1}


def test_unknown_model_costs_nothing_rather_than_crashing():
    """A model missing from the price table must not break a run."""
    p = provider(
        _Client(_Response(parsed=Answer(verdict="A", score=3))),
        model="claude-future-9",
    )
    p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")
    assert p.usage.cost_usd == 0.0
    assert p.usage.total_tokens > 0


def test_missing_usage_block_is_tolerated():
    resp = _Response(parsed=Answer(verdict="A", score=3))
    resp.usage = None
    p = provider(_Client(resp))
    p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")
    assert p.usage.total_tokens == 0
    assert p.usage.calls == 1


# --------------------------------------------------------------------------
# SDK exception translation
# --------------------------------------------------------------------------


def test_sdk_status_error_becomes_a_provider_error():
    class _StatusError(Exception):
        def __init__(self) -> None:
            super().__init__("rate limited")
            self.status_code = 429

    class _Sdk:
        APIStatusError = _StatusError
        APIConnectionError = ConnectionError

    client = _Client(_Response(), raiser=_StatusError())
    p = AnthropicProvider(client=client, sdk_module=_Sdk)
    with pytest.raises(ProviderError, match="429"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="judge")


def test_sdk_connection_error_becomes_a_provider_error():
    class _Sdk:
        APIStatusError = RuntimeError
        APIConnectionError = ConnectionError

    client = _Client(_Response(), raiser=ConnectionError("dns failure"))
    p = AnthropicProvider(client=client, sdk_module=_Sdk)
    with pytest.raises(ProviderError, match="could not reach"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="judge")


def test_absent_sdk_exception_types_do_not_break_the_handlers():
    """With no sdk_module, the except clauses must still be valid."""
    client = _Client(_Response(), raiser=ValueError("unexpected"))
    p = AnthropicProvider(client=client)  # sdk_module is None
    with pytest.raises(ValueError, match="unexpected"):
        p.complete_structured(system="s", prompt="p", schema=Answer, purpose="t")


# --------------------------------------------------------------------------
# Integration with the judge
# --------------------------------------------------------------------------


def test_llm_judge_drives_this_provider_end_to_end():
    """The real LLMJudgeAdapter against a stubbed model, offline."""
    from research_harness.evaluation.judge_adapter import (
        LLMJudgeAdapter,
        _LLMPairResponse,
    )
    from research_harness.evaluation.rubric import build_rubric

    from ..conftest import make_contract, make_evidence, make_proposal

    contract = make_contract()
    rubric = build_rubric(contract)
    client = _Client(
        _Response(
            parsed=_LLMPairResponse(verdict="B", rationale="B has better evidence")
        )
    )
    judge = LLMJudgeAdapter(provider(client))

    verdict = judge.compare_pair(
        goal=contract,
        rubric=rubric,
        proposal_a=make_proposal("PA"),
        proposal_b=make_proposal("PB"),
        evidence=[make_evidence("E001")],
    )
    assert verdict.verdict == "B"
    assert verdict.winner_id == "PB"
    assert "anthropic" in verdict.judge_id

    # The prompt the judge sent must not leak candidate identities.
    prompt = client.messages.calls[0]["messages"][0]["content"]
    assert "PA" not in prompt and "PB" not in prompt


def test_llm_judge_coerces_an_invalid_verdict_to_abstention():
    from research_harness.evaluation.judge_adapter import (
        LLMJudgeAdapter,
        _LLMPairResponse,
    )
    from research_harness.evaluation.rubric import build_rubric

    from ..conftest import make_contract, make_proposal

    contract = make_contract()
    judge = LLMJudgeAdapter(
        provider(
            _Client(
                _Response(parsed=_LLMPairResponse(verdict="definitely A!", rationale=""))
            )
        )
    )
    verdict = judge.compare_pair(
        goal=contract,
        rubric=build_rubric(contract),
        proposal_a=make_proposal("PA"),
        proposal_b=make_proposal("PB"),
        evidence=[],
    )
    assert verdict.verdict == "insufficient_evidence"


def test_llm_judge_discards_invented_evidence_ids():
    """A judge citing a source that does not exist must not have it recorded."""
    from research_harness.evaluation.judge_adapter import (
        LLMJudgeAdapter,
        _LLMConstraint,
        _LLMDimScore,
        _LLMScoringResponse,
    )
    from research_harness.evaluation.rubric import build_rubric

    from ..conftest import make_contract, make_evidence, make_proposal

    contract = make_contract()
    rubric = build_rubric(contract)
    response = _LLMScoringResponse(
        dimension_scores=[
            _LLMDimScore(
                dimension_key=d.key,
                raw=4,
                justification="fine",
                evidence_ids=["E001", "E999_fabricated"],
            )
            for d in rubric.dimensions
        ],
        constraint_assessments=[
            _LLMConstraint(
                constraint_key="ghost_constraint", status="PASS", rationale="n/a"
            )
        ],
    )
    judge = LLMJudgeAdapter(provider(_Client(_Response(parsed=response))))
    scoring = judge.score_rubric(
        goal=contract,
        rubric=rubric,
        proposal=make_proposal("PA"),
        evidence=[make_evidence("E001")],
    )
    for s in scoring.dimension_scores:
        assert s.evidence_ids == ["E001"], "fabricated evidence id was not dropped"
        assert s.score_source == "llm_judge"
    # A constraint not declared in the contract is ignored.
    assert scoring.constraint_assessments == []


def test_llm_judge_maps_unknown_dimensions_to_unknown_not_zero():
    from research_harness.evaluation.judge_adapter import (
        LLMJudgeAdapter,
        _LLMScoringResponse,
    )
    from research_harness.evaluation.rubric import build_rubric

    from ..conftest import make_contract, make_proposal

    contract = make_contract()
    rubric = build_rubric(contract)
    # The model returned nothing at all.
    judge = LLMJudgeAdapter(
        provider(_Client(_Response(parsed=_LLMScoringResponse())))
    )
    scoring = judge.score_rubric(
        goal=contract, rubric=rubric, proposal=make_proposal("PA"), evidence=[]
    )
    assert len(scoring.dimension_scores) == len(rubric.dimensions)
    for s in scoring.dimension_scores:
        assert s.status == "UNKNOWN"
        assert s.raw is None, "a missing score must not become a 0"
