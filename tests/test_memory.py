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


# --- TestRowToDict ---------------------------------------------------------


class TestRowToDict:
    def test_basic_row(self, conn, fake_embed):
        row = conn.execute("SELECT * FROM rules WHERE 0").fetchone()
        # Just verify the function doesn't crash on a real row shape
        # by creating a rule and reading it back
        rid, _ = memory.add_rule(
            conn, memory.Rule(rule_text="test", scope="global")
        )
        row = conn.execute("SELECT * FROM rules WHERE id=?", (rid,)).fetchone()
        d = memory.row_to_dict(row)
        assert d["rule_text"] == "test"
        assert d["scope"] == "global"
        assert d["is_meta"] is False
        assert d["tags"] == []

    def test_json_tags_parsed(self, conn, fake_embed):
        rid, _ = memory.add_rule(
            conn, memory.Rule(rule_text="test", scope="global", tags=["a", "b"])
        )
        row = conn.execute("SELECT * FROM rules WHERE id=?", (rid,)).fetchone()
        d = memory.row_to_dict(row)
        assert d["tags"] == ["a", "b"]

    def test_embedding_removed(self, conn, fake_embed):
        rid, _ = memory.add_rule(
            conn, memory.Rule(rule_text="test", scope="global")
        )
        row = conn.execute("SELECT * FROM rules WHERE id=?", (rid,)).fetchone()
        d = memory.row_to_dict(row)
        assert "embedding" not in d


# --- TestRecordFeedback ----------------------------------------------------


class TestRecordFeedback:
    def test_inserts_row(self, conn):
        fid = memory.record_feedback(conn, "pay", {"action": "delete", "index": 0})
        assert fid > 0
        row = conn.execute(
            "SELECT * FROM feedback_events WHERE id=?", (fid,)
        ).fetchone()
        assert row["task_id"] == "pay"
        raw = __import__("json").loads(row["raw_json"])
        assert raw["action"] == "delete"

    def test_created_at_set(self, conn):
        fid = memory.record_feedback(conn, "billing", {})
        row = conn.execute(
            "SELECT * FROM feedback_events WHERE id=?", (fid,)
        ).fetchone()
        assert row["created_at"] != ""
        assert "T" in row["created_at"]  # ISO format


# --- TestAddRuleMerge ------------------------------------------------------


class TestAddRuleMerge:
    def test_near_duplicate_merges(self, conn, fake_embed):
        r1 = memory.Rule(rule_text="No style nits", scope="global", is_meta=True,
                         evidence_count=2, confidence=0.5)
        id1, action1 = memory.add_rule(conn, r1)
        assert action1 == "inserted"

        r2 = memory.Rule(rule_text="No style nits", scope="global", is_meta=True,
                         evidence_count=1, confidence=0.5)
        id2, action2 = memory.add_rule(conn, r2)
        assert action2 == "merged"
        assert id2 == id1  # same rule id

        row = conn.execute("SELECT * FROM rules WHERE id=?", (id1,)).fetchone()
        # evidence_count should have grown
        assert int(row["evidence_count"]) >= 3
        # confidence should have grown
        assert float(row["confidence"]) > 0.5

    def test_distinct_rules_not_merged(self, conn, fake_embed):
        r1 = memory.Rule(rule_text="Rule alpha", scope="global")
        r2 = memory.Rule(rule_text="Rule beta", scope="global")
        id1, a1 = memory.add_rule(conn, r1)
        id2, a2 = memory.add_rule(conn, r2)
        assert a1 == "inserted"
        assert a2 == "inserted"
        assert id1 != id2


# --- TestAddRuleSupersede --------------------------------------------------


class TestAddRuleSupersede:
    def test_conflicting_rule_supersedes(self, conn, fake_embed):
        # First rule: require security findings
        r1 = memory.Rule(
            rule_text="Report security at high severity",
            scope="global",
            is_meta=False,
            assert_kind="require_severity_for_category",
            assert_cat="security",
            assert_sev="high",
            evidence_count=3,
            confidence=0.8,
        )
        id1, action1 = memory.add_rule(conn, r1)
        assert action1 == "inserted"

        # Second rule: forbid security category (direct conflict)
        r2 = memory.Rule(
            rule_text="Never report security findings",
            scope="global",
            is_meta=False,
            assert_kind="forbid_category",
            assert_cat="security",
            evidence_count=1,
            confidence=0.5,
        )
        id2, action2 = memory.add_rule(conn, r2)
        assert action2 == "superseded"

        # Old rule should be marked superseded
        old = conn.execute("SELECT * FROM rules WHERE id=?", (id1,)).fetchone()
        assert old["status"] == "superseded"
        assert old["superseded_by"] == id2

        # New rule is active
        new = conn.execute("SELECT * FROM rules WHERE id=?", (id2,)).fetchone()
        assert new["status"] == "active"


# --- TestRecordCompliance --------------------------------------------------


class TestRecordCompliance:
    def test_records_ok_and_fail(self, conn, fake_embed):
        rid, _ = memory.add_rule(
            conn, memory.Rule(rule_text="r", scope="global")
        )
        memory.record_compliance(conn, [(rid, True), (rid, False)])
        row = conn.execute(
            "SELECT * FROM rules WHERE id=?", (rid,)
        ).fetchone()
        assert row["applied_total"] == 2
        assert row["applied_ok"] == 1

    def test_empty_results_noop(self, conn, fake_embed):
        rid, _ = memory.add_rule(
            conn, memory.Rule(rule_text="r", scope="global")
        )
        memory.record_compliance(conn, [])
        row = conn.execute(
            "SELECT * FROM rules WHERE id=?", (rid,)
        ).fetchone()
        assert row["applied_total"] == 0


# --- TestScopeAllows -------------------------------------------------------


class TestScopeAllows:
    def test_global_always_allowed(self, conn, fake_embed):
        r = memory.Rule(rule_text="global rule", scope="global")
        memory.add_rule(conn, r)
        row = conn.execute("SELECT * FROM rules").fetchone()
        assert memory._scope_allows(row, "go", "x.go") is True
        assert memory._scope_allows(row, "python", "x.py") is True

    def test_language_gated(self, conn, fake_embed):
        r = memory.Rule(rule_text="py rule", scope="language", lang="python")
        memory.add_rule(conn, r)
        row = conn.execute("SELECT * FROM rules").fetchone()
        assert memory._scope_allows(row, "python", "x.py") is True
        assert memory._scope_allows(row, "go", "x.go") is False
        assert memory._scope_allows(row, None, None) is False

    def test_module_gated(self, conn, fake_embed):
        r = memory.Rule(rule_text="mod rule", scope="module",
                        lang="python", area="svc/pay.py")
        memory.add_rule(conn, r)
        row = conn.execute("SELECT * FROM rules").fetchone()
        assert memory._scope_allows(row, "python", "svc/pay.py") is True
        assert memory._scope_allows(row, "python", "svc/auth.py") is False
        assert memory._scope_allows(row, "go", "svc/pay.py") is False

    def test_module_prefix_match(self, conn, fake_embed):
        r = memory.Rule(rule_text="mod rule", scope="module",
                        lang="python", area="svc/")
        memory.add_rule(conn, r)
        row = conn.execute("SELECT * FROM rules").fetchone()
        assert memory._scope_allows(row, "python", "svc/pay.py") is True
        assert memory._scope_allows(row, "python", "svc/billing.py") is True


# --- TestRecencyWeight -----------------------------------------------------


class TestRecencyWeight:
    def test_fresh_rule_full_weight(self):
        w = memory._recency_weight(memory._now(), half_life_days=45.0)
        assert w >= 0.9  # fresh -> near 1.0

    def test_old_rule_decayed(self):
        import datetime
        old_ts = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            - datetime.timedelta(days=200)
        ).isoformat(timespec="seconds")
        w = memory._recency_weight(old_ts, half_life_days=45.0)
        assert w < 0.7  # 200 days >> 45-day half-life

    def test_invalid_timestamp_defaults_to_one(self):
        assert memory._recency_weight("not-a-date") == 1.0

    def test_never_reaches_zero(self):
        import datetime
        very_old = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            - datetime.timedelta(days=3650)
        ).isoformat(timespec="seconds")
        w = memory._recency_weight(very_old, half_life_days=45.0)
        assert w > 0.0


# --- TestRetrieveScoring ---------------------------------------------------


class TestRetrieveScoring:
    def test_retrieve_returns_similarity_scores(self, conn, fake_embed):
        r = memory.Rule(rule_text="Python validation", scope="language", lang="python")
        memory.add_rule(conn, r)
        rules, _ = memory.retrieve(
            conn, query_text="python validation review", lang="python"
        )
        assert len(rules) == 1
        assert "_similarity" in rules[0]
        assert "_score" in rules[0]

    def test_retrieve_meta_rules_no_similarity(self, conn, fake_embed):
        r = memory.Rule(rule_text="Always include evidence", scope="global",
                        is_meta=True)
        memory.add_rule(conn, r)
        rules, _ = memory.retrieve(
            conn, query_text="anything", lang="go"
        )
        assert len(rules) == 1
        assert rules[0]["_similarity"] is None
        assert rules[0]["_score"] is None

    def test_retrieve_meta_comes_first(self, conn, fake_embed):
        meta = memory.Rule(rule_text="Meta rule", scope="global", is_meta=True)
        other = memory.Rule(rule_text="Python rule", scope="language", lang="python")
        memory.add_rule(conn, meta)
        memory.add_rule(conn, other)
        rules, _ = memory.retrieve(
            conn, query_text="python", lang="python"
        )
        assert len(rules) == 2
        assert rules[0]["rule_text"] == "Meta rule"

    def test_retrieve_increments_hits(self, conn, fake_embed):
        r = memory.Rule(rule_text="Rule", scope="global")
        rid, _ = memory.add_rule(conn, r)
        assert conn.execute(
            "SELECT hits FROM rules WHERE id=?", (rid,)
        ).fetchone()["hits"] == 0
        memory.retrieve(conn, query_text="test", lang="python")
        assert conn.execute(
            "SELECT hits FROM rules WHERE id=?", (rid,)
        ).fetchone()["hits"] == 1
