"""Tests for the distillation module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app import schemas
from app.config import SINGLE_EVIDENCE_CAP
from app.distill import (
    FeedbackAction,
    DistillResult,
    _coerce_rule,
    _render_feedback,
    actions_from_payload,
    distill,
)
from app.llm import Usage
from app.memory import Rule


class TestRenderFeedback:
    def test_empty(self):
        assert _render_feedback([]) == "(no actions)"

    def test_deletions_first(self):
        actions = [
            FeedbackAction("delete", {"category": "style", "severity": "low"}),
            FeedbackAction("keep", {"category": "security", "severity": "high"}),
        ]
        out = _render_feedback(actions)
        first_line = out.split("\n")[0]
        assert "DELETED" in first_line

    def test_edit_includes_note(self):
        actions = [
            FeedbackAction(
                "edit",
                {"category": "style", "severity": "low"},
                replacement={"category": "style", "severity": "high"},
                note="make it high",
            ),
        ]
        out = _render_feedback(actions)
        assert "EDITED" in out
        assert "reviewer note: 'make it high'" in out


class TestActionsFromPayload:
    def test_keep_is_default(self):
        findings = [
            {"category": "security", "severity": "high", "message": "a"},
            {"category": "style", "severity": "low", "message": "b"},
        ]
        payload = [{"index": 1, "action": "delete", "note": "nit"}]
        actions = actions_from_payload(findings, payload)
        assert actions[0].action == "keep"
        assert actions[1].action == "delete"
        assert actions[1].note == "nit"

    def test_invalid_action_defaults_to_keep(self):
        findings = [{"category": "x", "severity": "y", "message": "z"}]
        payload = [{"index": 0, "action": "bogus"}]
        actions = actions_from_payload(findings, payload)
        assert actions[0].action == "keep"


class TestCoerceRule:
    def test_minimal_valid(self):
        rule = _coerce_rule(
            {"rule_text": "Do the thing", "scope": "global", "is_meta": False},
            lang="python",
            area="svc/pay.py",
        )
        assert rule is not None
        assert rule.rule_text == "Do the thing"
        assert rule.scope == "global"

    def test_empty_text_dropped(self):
        assert _coerce_rule({"rule_text": "   "}, lang="python", area="svc/x.py") is None
        assert _coerce_rule({}, lang="python", area="svc/x.py") is None

    def test_invalid_scope_defaults_to_language(self):
        rule = _coerce_rule(
            {"rule_text": "Rule", "scope": "bogus"},
            lang="python",
            area="svc/pay.py",
        )
        assert rule is not None
        assert rule.scope == "language"

    def test_language_rule_gets_lang_tag(self):
        rule = _coerce_rule(
            {"rule_text": "Rule", "scope": "language"},
            lang="python",
            area="svc/pay.py",
        )
        assert rule.lang == "python"

    def test_module_rule_gets_area(self):
        rule = _coerce_rule(
            {"rule_text": "Rule", "scope": "module"},
            lang="python",
            area="svc/pay.py",
        )
        assert rule.area == "svc/pay.py"

    def test_global_rule_no_lang_or_area(self):
        rule = _coerce_rule(
            {"rule_text": "Rule", "scope": "global"},
            lang="python",
            area="svc/pay.py",
        )
        assert rule.lang is None
        assert rule.area is None

    def test_confidence_capped_for_single_evidence(self):
        rule = _coerce_rule(
            {"rule_text": "Rule", "scope": "global", "evidence_count": 1,
             "confidence": 0.99},
            lang="python",
            area="svc/x.py",
        )
        assert rule is not None
        assert rule.confidence <= SINGLE_EVIDENCE_CAP

    def test_forbid_category_needs_cat(self):
        rule = _coerce_rule(
            {
                "rule_text": "Rule",
                "scope": "global",
                "assertion": {"kind": "forbid_category", "category": "", "severity": "", "field": ""},
            },
            lang="python",
            area="svc/x.py",
        )
        assert rule is not None
        assert rule.assert_kind == "none"

    def test_require_field_needs_field(self):
        rule = _coerce_rule(
            {
                "rule_text": "Rule",
                "scope": "global",
                "assertion": {"kind": "require_field", "category": "", "severity": "", "field": ""},
            },
            lang="python",
            area="svc/x.py",
        )
        assert rule is not None
        assert rule.assert_kind == "none"


# --- helpers for integration tests ------------------------------------------


def _mock_llm_for_distill(rules_payload):
    m = MagicMock()
    def _structured(**kw):
        usage = kw.get("usage")
        if usage is not None:
            usage.calls += 1
        return {"rules": rules_payload}
    m.structured = MagicMock(side_effect=_structured)
    return m


# --- TestDistillResult -----------------------------------------------------


class TestDistillResult:
    def test_empty_as_dict(self):
        dr = DistillResult()
        d = dr.as_dict()
        assert d["rules"] == []
        assert d["feedback_id"] is None
        assert d["t_distill"] == 0.0
        assert d["actions"] == {}

    def test_as_dict_with_values(self):
        usage = Usage(in_tok=10, cost_usd=0.01)
        dr = DistillResult(
            rules=[{"rule_text": "r", "scope": "global", "id": 1, "action": "inserted"}],
            feedback_id=5,
            usage=usage,
            t_distill=0.3,
            actions={"keep": 1, "delete": 2, "edit": 0},
        )
        d = dr.as_dict()
        assert len(d["rules"]) == 1
        assert d["feedback_id"] == 5
        assert d["t_distill"] == 0.3
        assert d["usage"]["in_tok"] == 10


# --- TestDistill -----------------------------------------------------------


class TestDistill:
    def test_empty_actions_returns_early(self, conn, fake_embed):
        llm = _mock_llm_for_distill([])
        usage = Usage()
        result = distill(
            conn, task_id="pay", actions=[], language="python",
            area="svc/pay.py", llm=llm, usage=usage,
        )
        assert result.rules == []
        assert result.feedback_id is None
        assert usage.calls == 0

    def test_returns_rules_from_llm(self, conn, fake_embed):
        rules_payload = [{
            "rule_text": "No style nits",
            "scope": "global",
            "is_meta": True,
            "assertion": {"kind": "forbid_category", "category": "style",
                          "severity": "", "field": ""},
            "evidence_count": 1,
            "confidence": 0.5,
            "tags": ["style"],
        }]
        llm = _mock_llm_for_distill(rules_payload)
        usage = Usage()
        actions = [
            FeedbackAction("delete", {"category": "style", "severity": "low"},
                           note="no nits"),
        ]
        result = distill(
            conn, task_id="pay", actions=actions, language="python",
            area="svc/pay.py", llm=llm, usage=usage,
        )
        assert len(result.rules) == 1
        assert result.rules[0]["rule_text"] == "No style nits"
        assert result.rules[0]["scope"] == "global"
        assert result.feedback_id is not None
        assert usage.calls == 1

    def test_records_feedback_event(self, conn, fake_embed):
        llm = _mock_llm_for_distill([{
            "rule_text": "Rule", "scope": "global", "is_meta": False,
            "assertion": {"kind": "none", "category": "", "severity": "", "field": ""},
            "evidence_count": 1, "confidence": 0.5, "tags": [],
        }])
        actions = [FeedbackAction("keep", {"category": "x", "severity": "y", "message": "m"})]
        result = distill(
            conn, task_id="pay", actions=actions, language="python",
            area="svc/pay.py", llm=llm,
        )
        assert result.feedback_id is not None
        assert result.feedback_id > 0
        row = conn.execute(
            "SELECT * FROM feedback_events WHERE id=?", (result.feedback_id,)
        ).fetchone()
        assert row is not None
        assert row["task_id"] == "pay"

    def test_dedup_same_rule_text(self, conn, fake_embed):
        rules_payload = [
            {"rule_text": "Same rule", "scope": "global", "is_meta": False,
             "assertion": {"kind": "none", "category": "", "severity": "", "field": ""},
             "evidence_count": 1, "confidence": 0.5, "tags": []},
            {"rule_text": "Same rule", "scope": "global", "is_meta": False,
             "assertion": {"kind": "none", "category": "", "severity": "", "field": ""},
             "evidence_count": 1, "confidence": 0.5, "tags": []},
        ]
        llm = _mock_llm_for_distill(rules_payload)
        actions = [
            FeedbackAction("delete", {"category": "x", "severity": "y", "message": "m"}),
            FeedbackAction("delete", {"category": "x", "severity": "y", "message": "m2"}),
        ]
        result = distill(
            conn, task_id="pay", actions=actions, language="python",
            area="svc/pay.py", llm=llm,
        )
        texts = [r["rule_text"] for r in result.rules]
        assert texts.count("Same rule") == 1

    def test_drops_malformed_rules(self, conn, fake_embed):
        rules_payload = [
            {"rule_text": "", "scope": "global", "is_meta": False,
             "assertion": {"kind": "none", "category": "", "severity": "", "field": ""},
             "evidence_count": 1, "confidence": 0.5, "tags": []},
            {"rule_text": "   ", "scope": "global", "is_meta": False,
             "assertion": {"kind": "none", "category": "", "severity": "", "field": ""},
             "evidence_count": 1, "confidence": 0.5, "tags": []},
            {"rule_text": "Good rule", "scope": "global", "is_meta": False,
             "assertion": {"kind": "none", "category": "", "severity": "", "field": ""},
             "evidence_count": 1, "confidence": 0.5, "tags": []},
        ]
        llm = _mock_llm_for_distill(rules_payload)
        actions = [
            FeedbackAction("delete", {"category": "x", "severity": "y", "message": "m"}),
        ]
        result = distill(
            conn, task_id="pay", actions=actions, language="python",
            area="svc/pay.py", llm=llm,
        )
        assert len(result.rules) == 1
        assert result.rules[0]["rule_text"] == "Good rule"

    def test_max_rules_cap(self, conn, fake_embed):
        rules_payload = [
            {"rule_text": f"Rule {i}", "scope": "global", "is_meta": False,
             "assertion": {"kind": "none", "category": "", "severity": "", "field": ""},
             "evidence_count": 1, "confidence": 0.5, "tags": []}
            for i in range(10)
        ]
        llm = _mock_llm_for_distill(rules_payload)
        actions = [
            FeedbackAction("delete", {"category": "x", "severity": "y", "message": "m"})
            for _ in range(10)
        ]
        result = distill(
            conn, task_id="pay", actions=actions, language="python",
            area="svc/pay.py", llm=llm,
        )
        assert len(result.rules) <= 3

    def test_actions_counted(self, conn, fake_embed):
        llm = _mock_llm_for_distill([])
        actions = [
            FeedbackAction("keep", {}),
            FeedbackAction("delete", {}),
            FeedbackAction("delete", {}),
            FeedbackAction("edit", {}, replacement={}),
        ]
        result = distill(
            conn, task_id="pay", actions=actions, language="python",
            area="svc/pay.py", llm=llm,
        )
        assert result.actions == {"keep": 1, "delete": 2, "edit": 1}

    def test_persist_false_skips_feedback_record(self, conn, fake_embed):
        llm = _mock_llm_for_distill([{
            "rule_text": "Rule", "scope": "global", "is_meta": False,
            "assertion": {"kind": "none", "category": "", "severity": "", "field": ""},
            "evidence_count": 1, "confidence": 0.5, "tags": [],
        }])
        actions = [FeedbackAction("keep", {})]
        result = distill(
            conn, task_id="pay", actions=actions, language="python",
            area="svc/pay.py", llm=llm, persist=False,
        )
        assert result.feedback_id is None
