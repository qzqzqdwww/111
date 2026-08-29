"""Tests for the SQLite memory store."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app import config, memory
from app.memory import Rule, clamp_confidence


class TestClampConfidence:
    def test_normal_range(self):
        assert clamp_confidence(0.7, 5) == 0.7

    def test_clamped_to_one(self):
        assert clamp_confidence(1.5, 5) == 1.0

    def test_clamped_to_zero(self):
        assert clamp_confidence(-0.1, 5) == 0.0

    def test_single_evidence_cap(self):
        assert clamp_confidence(0.99, 1) == config.SINGLE_EVIDENCE_CAP
        assert clamp_confidence(0.99, 2) == config.SINGLE_EVIDENCE_CAP

    def test_three_evidence_not_capped(self):
        assert clamp_confidence(0.99, 3) == 0.99


class TestRuleDataclass:
    def test_defaults(self):
        r = Rule(rule_text="test")
        assert r.scope == "global"
        assert r.is_meta is False
        assert r.confidence == 0.5
        assert r.tags == []
        assert r.evidence_count == 1

    def test_tags_none_becomes_empty(self):
        r = Rule(rule_text="test", tags=None)
        assert r.tags == []


class TestRuleCRUD:
    def test_add_and_list(self, conn, fake_embed):
        rule = Rule(
            rule_text="No style nits",
            scope="global",
            is_meta=True,
            assert_kind="forbid_category",
            assert_cat="style",
            evidence_count=3,
            confidence=0.9,
        )
        rid, action = memory.add_rule(conn, rule)
        assert rid > 0
        assert action == "inserted"

        rules = memory.list_rules(conn)
        assert len(rules) == 1
        assert rules[0]["rule_text"] == "No style nits"

    def test_distinct_rules_not_merged(self, conn, fake_embed):
        r1 = Rule(rule_text="r1", scope="global", confidence=0.5)
        r2 = Rule(rule_text="r2", scope="global", confidence=0.5)
        id1, action1 = memory.add_rule(conn, r1)
        id2, action2 = memory.add_rule(conn, r2)
        # Different texts -> different vectors -> not merged
        assert action1 == "inserted"
        assert action2 == "inserted"
        rules = memory.list_rules(conn)
        assert len(rules) == 2


class TestRetrieval:
    def test_empty_store(self, conn):
        rules, toks = memory.retrieve(conn, query_text="anything", lang="python")
        assert rules == []
        assert toks == 0

    def test_meta_rules_unconditional(self, conn, fake_embed):
        meta = Rule(rule_text="Always include evidence", scope="global", is_meta=True)
        other = Rule(
            rule_text="Python validation at high severity",
            scope="language",
            lang="python",
        )
        memory.add_rule(conn, meta)
        memory.add_rule(conn, other)

        # Go task should get meta but NOT the Python-only rule
        rules, _ = memory.retrieve(conn, query_text="Go review", lang="go", area="x.go")
        texts = [r["rule_text"] for r in rules]
        assert "Always include evidence" in texts
        assert "Python validation" not in texts

    def test_language_gating(self, conn, fake_embed):
        py_rule = Rule(
            rule_text="Python style", scope="language", lang="python"
        )
        go_rule = Rule(rule_text="Go style", scope="language", lang="go")
        memory.add_rule(conn, py_rule)
        memory.add_rule(conn, go_rule)

        rules, _ = memory.retrieve(conn, query_text="python", lang="python", area="x.py")
        texts = [r["rule_text"] for r in rules]
        assert "Python style" in texts
        assert "Go style" not in texts

    def test_top_k_limit(self, conn, fake_embed):
        for i in range(10):
            memory.add_rule(
                conn,
                Rule(rule_text=f"Rule {i}", scope="global"),
            )
        rules, _ = memory.retrieve(conn, query_text="test", lang="python", top_k=3)
        assert len(rules) <= 3


class TestStats:
    def test_empty_store(self, conn):
        s = memory.stats(conn)
        assert s["total"] == 0
        assert s["active"] == 0

    def test_with_rules(self, conn, fake_embed):
        memory.add_rule(conn, Rule(rule_text="r1", scope="global"))
        memory.add_rule(conn, Rule(rule_text="r2", scope="global"))
        s = memory.stats(conn)
        assert s["total"] == 2
        assert s["active"] == 2
