"""Tests for the LLM wrapper."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from app import config, schemas
from app.llm import LLM, LLMError, Usage, shared


# --- helpers ----------------------------------------------------------------


class _Block:
    def __init__(self, block_type, **kw):
        self.type = block_type
        for k, v in kw.items():
            setattr(self, k, v)
        self.id = id(self)


def _make_llm():
    """Create an LLM instance bypassing __init__ (which requires credentials)."""
    llm = LLM.__new__(LLM)
    llm._client = MagicMock()
    llm.max_retries = 3
    return llm


def _usage_resp():
    resp = MagicMock()
    resp.usage = MagicMock(
        input_tokens=10, output_tokens=5,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )
    return resp


# --- Usage ----------------------------------------------------------------


class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert u.in_tok == 0
        assert u.cost_usd == 0.0
        assert u.calls == 0

    def test_add_response(self):
        u = Usage()
        resp = MagicMock()
        resp.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=20, cache_read_input_tokens=80,
        )
        u.add_response(config.MODEL_MAIN, resp)
        assert u.in_tok == 100
        assert u.out_tok == 50
        assert u.cache_write == 20
        assert u.cache_read == 80
        assert u.calls == 1
        assert u.cost_usd > 0

    def test_add_embedding(self):
        u = Usage()
        u.add_embedding(1_000_000)
        assert u.embed_tok == 1_000_000
        assert u.cost_usd == pytest.approx(0.02)

    def test_merge(self):
        a = Usage(in_tok=10, cost_usd=0.1)
        b = Usage(out_tok=20, cost_usd=0.2)
        a.merge(b)
        assert a.in_tok == 10
        assert a.out_tok == 20
        assert a.cost_usd == pytest.approx(0.3)

    def test_cache_hit_ratio(self):
        u = Usage(cache_read=70, cache_write=20, in_tok=10)
        assert u.cache_hit_ratio == pytest.approx(70 / 100)

    def test_cache_hit_ratio_zero_total(self):
        u = Usage()
        assert u.cache_hit_ratio == 0.0

    def test_as_dict(self):
        u = Usage(in_tok=10, out_tok=5, calls=1)
        d = u.as_dict()
        assert d["in_tok"] == 10
        assert d["calls"] == 1
        assert "cache_hit_ratio" in d


# --- LLM -------------------------------------------------------------------


class TestLLMCall:
    def test_call_succeeds(self):
        llm = _make_llm()
        llm._client.messages.create.return_value = _usage_resp()

        usage = Usage()
        r = llm.call(model="m", messages=[], max_tokens=10, usage=usage)
        assert r is not None
        assert usage.in_tok == 10

    def test_retries_on_503(self):
        llm = _make_llm()
        llm.max_retries = 3

        fail = anthropic.APIStatusError(
            message="unavailable",
            response=MagicMock(status_code=503),
            body={},
        )
        ok_resp = _usage_resp()
        llm._client.messages.create.side_effect = [fail, fail, ok_resp]

        usage = Usage()
        r = llm.call(model="m", messages=[], max_tokens=10, usage=usage)
        assert r is ok_resp
        assert llm._client.messages.create.call_count == 3

    def test_no_retry_on_400(self):
        llm = _make_llm()
        llm.max_retries = 3

        err = anthropic.APIStatusError(
            message="bad request",
            response=MagicMock(status_code=400),
            body={},
        )
        llm._client.messages.create.side_effect = err

        with pytest.raises(LLMError, match="400"):
            llm.call(model="m", messages=[], max_tokens=10)

        assert llm._client.messages.create.call_count == 1


class TestLLMStructured:
    def test_parses_tool_use(self):
        llm = _make_llm()
        block = MagicMock()
        block.type = "tool_use"
        block.name = "emit_findings"
        block.input = {"findings": [{"file": "x.py", "line": 1,
                                      "category": "security", "severity": "high",
                                      "message": "m", "evidence": "e"}]}

        resp = MagicMock()
        resp.content = [block]
        llm.call = MagicMock(return_value=resp)

        result = llm.structured(
            model="m",
            tool=schemas.EMIT_FINDINGS_TOOL,
            messages=[],
            max_tokens=100,
        )
        assert result["findings"][0]["file"] == "x.py"

    def test_raises_if_no_tool_use_block(self):
        llm = _make_llm()
        resp = MagicMock()
        resp.content = [_Block("text", text="sorry")]
        resp.stop_reason = "end_turn"
        llm.call = MagicMock(return_value=resp)

        with pytest.raises(LLMError, match="no tool_use"):
            llm.structured(
                model="m", tool=schemas.EMIT_FINDINGS_TOOL,
                messages=[], max_tokens=100,
            )


class TestLLMPreflight:
    def test_ok(self):
        llm = _make_llm()
        llm._client.messages.create.return_value = _usage_resp()

        ok, msg = llm.preflight("m")
        assert ok is True
        assert msg == "ok"

    def test_failure(self):
        llm = _make_llm()
        llm._client.messages.create.side_effect = anthropic.APIStatusError(
            message="server error",
            response=MagicMock(status_code=500),
            body={},
        )

        ok, msg = llm.preflight("m")
        assert ok is False
        assert msg  # non-empty error message


class TestShared:
    def test_shared_returns_same_instance(self):
        import app.llm as llm_mod
        old = llm_mod._shared
        llm_mod._shared = None
        try:
            a = shared()
            b = shared()
            assert a is b
        finally:
            llm_mod._shared = old


# --- TestLLMConnectionRetry -------------------------------------------------


class TestLLMConnectionRetry:
    def test_retries_on_connection_error(self):
        llm = _make_llm()
        llm.max_retries = 3

        mock_request = MagicMock()
        fail = anthropic.APIConnectionError(message="network down", request=mock_request)
        ok = _usage_resp()
        llm._client.messages.create.side_effect = [fail, ok]

        usage = Usage()
        r = llm.call(model="m", messages=[], max_tokens=10, usage=usage)
        assert r is ok
        assert llm._client.messages.create.call_count == 2

    def test_exhausts_retries_then_raises(self):
        llm = _make_llm()
        llm.max_retries = 2

        llm._client.messages.create.side_effect = anthropic.APIStatusError(
            message="still down", response=MagicMock(status_code=503), body={},
        )

        with pytest.raises(LLMError, match="exhausted"):
            llm.call(model="m", messages=[], max_tokens=10)

        assert llm._client.messages.create.call_count == 2

    def test_429_retried(self):
        llm = _make_llm()
        llm.max_retries = 3

        fail = anthropic.APIStatusError(
            message="rate limited", response=MagicMock(status_code=429), body={},
        )
        ok = _usage_resp()
        llm._client.messages.create.side_effect = [fail, ok]

        usage = Usage()
        r = llm.call(model="m", messages=[], max_tokens=10, usage=usage)
        assert r is ok
        assert llm._client.messages.create.call_count == 2

    def test_529_retried(self):
        llm = _make_llm()
        llm.max_retries = 2

        fail = anthropic.APIStatusError(
            message="overloaded", response=MagicMock(status_code=529), body={},
        )
        ok = _usage_resp()
        llm._client.messages.create.side_effect = [fail, ok]

        usage = Usage()
        r = llm.call(model="m", messages=[], max_tokens=10, usage=usage)
        assert r is ok


# --- TestUsageCostCalculation ----------------------------------------------


class TestUsageCostCalculation:
    def test_sonnet_cost(self):
        """Verify the cost formula against known prices."""
        u = Usage()
        resp = MagicMock()
        resp.usage = MagicMock(
            input_tokens=1_000_000,
            output_tokens=500_000,
            cache_creation_input_tokens=100_000,
            cache_read_input_tokens=900_000,
        )
        u.add_response(config.MODEL_MAIN, resp)
        # Sonnet: $3/M inp, $15/M out, cache_write 1.25x, cache_read 0.1x
        expected = (
            1_000_000 * 3.00
            + 500_000 * 15.00
            + 100_000 * 3.00 * 1.25
            + 900_000 * 3.00 * 0.10
        ) / 1_000_000
        assert u.cost_usd == pytest.approx(expected)

    def test_zero_tokens_zero_cost(self):
        u = Usage()
        assert u.cost_usd == 0.0

    def test_merge_preserves_all_fields(self):
        a = Usage(in_tok=1, out_tok=2, cache_write=3, cache_read=4,
                  embed_tok=5, cost_usd=0.1, calls=2)
        b = Usage(in_tok=10, out_tok=20, cache_write=30, cache_read=40,
                  embed_tok=50, cost_usd=0.2, calls=3)
        a.merge(b)
        assert a.in_tok == 11
        assert a.out_tok == 22
        assert a.cache_write == 33
        assert a.cache_read == 44
        assert a.embed_tok == 55
        assert a.cost_usd == pytest.approx(0.3)
        assert a.calls == 5

    def test_cache_hit_ratio_formula(self):
        """cache_read / (cache_read + cache_write + in_tok)"""
        u = Usage(cache_read=70, cache_write=20, in_tok=10)
        expected = 70 / (70 + 20 + 10)
        assert u.cache_hit_ratio == pytest.approx(expected)

    def test_as_dict_rounds_cost(self):
        u = Usage(cost_usd=0.123456789)
        d = u.as_dict()
        assert d["cost_usd"] == 0.123457  # rounded to 6 places
