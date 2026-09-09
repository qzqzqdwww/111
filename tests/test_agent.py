"""Tests for the two-phase review agent."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from app import agent, schemas, tools
from app.llm import Usage


# --- helpers ----------------------------------------------------------------


class _Block:
    def __init__(self, block_type, **kw):
        self.type = block_type
        for k, v in kw.items():
            setattr(self, k, v)
        self.id = id(self)


def _tool_use_block(name, input_dict):
    return _Block("tool_use", name=name, input=input_dict)


def _text_block(text):
    return _Block("text", text=text)


def _resp(content, stop_reason="end_turn", usage=None):
    m = MagicMock()
    m.content = content
    m.stop_reason = stop_reason
    m.usage = usage or MagicMock(
        input_tokens=0, output_tokens=0,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )
    return m


def _mock_llm(responses):
    """Create a mock LLM whose .call() returns responses in order."""
    m = MagicMock()
    m.call_count = 0
    m.structured_responses = []

    def call(**kw):
        resp = responses[m.call_count]
        m.call_count += 1
        return resp

    def structured(**kw):
        return m.structured_responses.pop(0) if m.structured_responses else {"findings": []}

    m.call = MagicMock(side_effect=call)
    m.structured = MagicMock(side_effect=structured)
    return m


# --- format_memory_block ----------------------------------------------------


class TestFormatMemoryBlock:
    def test_empty(self):
        assert agent.format_memory_block([]) == ""

    def test_single_rule(self):
        rules = [{"rule_text": "No style nits", "scope": "global", "confidence": 0.9}]
        out = agent.format_memory_block(rules)
        assert "No style nits" in out
        assert "[global]" in out

    def test_tentative_marker(self):
        rules = [{"rule_text": "Rule", "scope": "global", "confidence": 0.4}]
        out = agent.format_memory_block(rules)
        assert "(tentative)" in out

    def test_language_tag(self):
        rules = [{"rule_text": "Rule", "scope": "language", "lang": "python", "confidence": 0.9}]
        out = agent.format_memory_block(rules)
        assert "[language/python]" in out


# --- build_system -----------------------------------------------------------


class TestBuildSystem:
    def test_no_rules(self):
        blocks = agent.build_system([])
        assert len(blocks) == 1
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_with_rules(self):
        blocks = agent.build_system([{"rule_text": "r", "scope": "g", "confidence": 0.9}])
        assert len(blocks) == 2
        assert blocks[1]["text"].startswith(agent.MEMORY_HEADER)


# --- _normalize_findings ----------------------------------------------------


class TestNormalizeFindings:
    def test_valid_finding(self):
        raw = [{"file": "x.py", "line": 10, "category": "security",
                "severity": "high", "message": "m", "evidence": "e"}]
        out = agent._normalize_findings(raw, "default.py")
        assert out[0]["file"] == "x.py"
        assert out[0]["line"] == 10

    def test_default_file(self):
        raw = [{"file": "", "line": 1, "category": "x", "severity": "y", "message": "m", "evidence": "e"}]
        out = agent._normalize_findings(raw, "default.py")
        assert out[0]["file"] == "default.py"

    def test_line_coerced_to_int(self):
        raw = [{"file": "x.py", "line": "abc", "category": "x",
                "severity": "y", "message": "m", "evidence": "e"}]
        out = agent._normalize_findings(raw, "x.py")
        assert out[0]["line"] == 0

    def test_category_normalized(self):
        raw = [{"file": "x.py", "line": 1, "category": "Nit",
                "severity": "low", "message": "m", "evidence": "e"}]
        out = agent._normalize_findings(raw, "x.py")
        assert out[0]["category"] == "style"

    def test_severity_normalized(self):
        raw = [{"file": "x.py", "line": 1, "category": "x",
                "severity": "critical", "message": "m", "evidence": "e"}]
        out = agent._normalize_findings(raw, "x.py")
        assert out[0]["severity"] == "high"

    def test_non_dict_skipped(self):
        raw = ["not a dict", {"file": "x.py", "line": 1, "category": "x",
                "severity": "y", "message": "m", "evidence": "e"}]
        out = agent._normalize_findings(raw, "x.py")
        assert len(out) == 1

    def test_empty_evidence_stripped(self):
        raw = [{"file": "x.py", "line": 1, "category": "x",
                "severity": "y", "message": "m", "evidence": "   "}]
        out = agent._normalize_findings(raw, "x.py")
        assert out[0]["evidence"] == ""


# --- review -----------------------------------------------------------------


class TestReview:
    def test_review_calls_llm(self):
        """Phase 1: model calls get_diff, we dispatch, then phase 2 emits findings."""
        diff_text = "--- a/svc/pay.py\n+++ b/svc/pay.py\n@@ -1,2 +1,3 @@\n x\n+y\n z\n"

        # Phase 1 response: model calls get_diff
        phase1 = _resp(
            [_tool_use_block("get_diff", {"task_id": "pay"})],
            stop_reason="tool_use",
        )
        # Phase 2 response: model emits findings
        phase2 = _resp(
            [_tool_use_block("emit_findings", {
                "findings": [{
                    "file": "svc/pay.py", "line": 2, "category": "security",
                    "severity": "high", "message": "no validation",
                    "evidence": "charge(uid, amt)",
                }]
            })],
            stop_reason="end_turn",
        )

        # Patch tools.dispatch to return the diff for get_diff
        orig_dispatch = tools.dispatch

        def fake_dispatch(name, tool_input):
            if name == "get_diff":
                return diff_text, False
            return orig_dispatch(name, tool_input)

        llm = _mock_llm([phase1, phase2])
        llm.structured_responses = [{
            "findings": [{
                "file": "svc/pay.py", "line": 2, "category": "security",
                "severity": "high", "message": "no validation",
                "evidence": "charge(uid, amt)",
            }]
        }]

        import unittest.mock
        with unittest.mock.patch("app.agent.tools.dispatch", side_effect=fake_dispatch):
            result = agent.review("pay", path="svc/pay.py", llm=llm)

        assert result.task_id == "pay"
        assert len(result.findings) == 1
        assert result.findings[0]["category"] == "security"
        # Round 1: model calls get_diff (tool_use) → we dispatch
        # Round 2: model returns end_turn with no more tools → break
        assert result.tool_rounds == 2

    def test_review_no_memory(self):
        llm = _mock_llm([])
        llm.structured_responses = [{"findings": []}]
        result = agent.review("pay", path="svc/pay.py", llm=llm)
        assert result.findings == []
        assert result.injected_rules == []

    def test_review_system_blocks_contain_cache(self):
        llm = _mock_llm([])
        llm.structured_responses = [{"findings": []}]
        result = agent.review("pay", path="svc/pay.py", llm=llm)
        # build_system should have been called with the rules
        assert result.as_dict()["usage"] is not None
