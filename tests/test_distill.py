"""Tests for the distillation module."""

from __future__ import annotations

import sqlite3

import pytest

from app import schemas
from app.config import SINGLE_EVIDENCE_CAP
from app.distill import (
    FeedbackAction,
    _coerce_rule,
    _render_feedback,
    actions_from_payload,
)
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
