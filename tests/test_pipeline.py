"""Tests for the pipeline orchestration layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app import catalog, memory, metrics, pipeline
from app.llm import Usage


# --- helpers ----------------------------------------------------------------


def _mock_llm(review_findings=None, structured_return=None):
    """Create a mock LLM for pipeline tests."""
    m = MagicMock()
    review_findings = review_findings or [{
        "file": "svc/pay.py", "line": 2, "category": "security",
        "severity": "high", "message": "no validation",
        "evidence": "charge(uid, amt)",
    }]
    structured_return = structured_return or {"findings": review_findings}

    def call(**kw):
        resp = MagicMock()
        resp.content = []
        resp.stop_reason = "end_turn"
        resp.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        return resp

    m.call = MagicMock(side_effect=call)
    m.structured = MagicMock(return_value=structured_return)
    m.preflight = MagicMock(return_value=(True, "ok"))
    return m


def _fake_embed(text):
    """Deterministic fake embedding."""
    dim = 1536
    vec = np.zeros(dim, dtype=np.float32)
    for i, b in enumerate(text.encode()):
        idx = b % dim
        vec[idx] += (b - 128) / 128.0
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        vec[0] = 1.0
        return vec, len(text.split())
    return (vec / norm).astype(np.float32), len(text.split())


# --- review_task ------------------------------------------------------------


class TestReviewTask:
    def test_returns_payload(self):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            res = pipeline.review_task(conn, "pay", memory_on=False, llm=llm)
        finally:
            conn.close()

        assert res["task"]["task_id"] == "pay"
        assert "findings" in res
        assert "cost" in res
        assert "timing" in res
        assert "usage" in res
        assert res["memory_on"] is False

    def test_memory_off_no_rules_injected(self):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            res = pipeline.review_task(conn, "pay", memory_on=False, llm=llm)
        finally:
            conn.close()

        assert res["injected_rules"] == []

    def test_memory_on_injects_rules(self):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            res = pipeline.review_task(conn, "pay", memory_on=True, llm=llm)
        finally:
            conn.close()

        assert res["memory_on"] is True
        # With no rules in the store, nothing is injected
        assert res["injected_rules"] == []

    def test_persists_metrics(self):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            pipeline.review_task(conn, "pay", memory_on=True, llm=llm)
            row = conn.execute("SELECT COUNT(*) AS n FROM turns").fetchone()
            assert row["n"] == 1
        finally:
            conn.close()

    def test_task_metadata(self):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            res = pipeline.review_task(conn, "pay", memory_on=False, llm=llm)
        finally:
            conn.close()

        task = res["task"]
        assert task["lang"] == "python"
        assert task["path"] == "svc/pay.py"
        assert task["change_kind"] == "new-feature"

    def test_unknown_task_raises(self):
        conn = memory.connect(":memory:")
        try:
            with pytest.raises(KeyError, match="unknown task"):
                pipeline.review_task(conn, "nonexistent", memory_on=False)
        finally:
            conn.close()


# --- apply_feedback ---------------------------------------------------------


class TestApplyFeedback:
    def test_returns_distill_result(self):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        llm.structured = MagicMock(return_value={
            "rules": [{
                "rule_text": "No style nits",
                "scope": "global",
                "is_meta": True,
                "assertion": {"kind": "forbid_category", "category": "style",
                              "severity": "", "field": ""},
                "evidence_count": 1,
                "confidence": 0.5,
                "tags": [],
            }]
        })
        try:
            findings = [{
                "file": "svc/pay.py", "line": 2, "category": "style",
                "severity": "low", "message": "naming",
                "evidence": "x",
            }]
            payload = [{"index": 0, "action": "delete", "note": "no nits"}]
            with patch("app.memory.embed") as mock_embed:
                mock_embed.embed_one = MagicMock(side_effect=_fake_embed)
                mock_embed.to_blob = MagicMock(return_value=b"\x00" * 6144)
                res = pipeline.apply_feedback(
                    conn, "pay", findings=findings, payload=payload, llm=llm,
                )
        finally:
            conn.close()

        assert "rules" in res
        assert "cost" in res
        assert res["task_id"] == "pay"

    def test_no_changes_returns_empty_rules(self):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            findings = [{
                "file": "svc/pay.py", "line": 2, "category": "security",
                "severity": "high", "message": "m", "evidence": "e",
            }]
            payload = [{"index": 0, "action": "keep"}]
            res = pipeline.apply_feedback(
                conn, "pay", findings=findings, payload=payload, llm=llm,
            )
        finally:
            conn.close()

        assert res["rules"] == []


# --- preflight --------------------------------------------------------------


class TestPreflight:
    def test_returns_status(self):
        llm = _mock_llm()
        llm.preflight = MagicMock(return_value=(True, "ok"))
        res = pipeline.preflight(llm=llm)
        assert "ok" in res
        assert "models" in res
        assert "main" in res["models"]
        assert "cheap" in res["models"]


# --- TestReviewTaskMemoryOn ------------------------------------------------


class TestReviewTaskMemoryOn:
    def test_rules_injected_from_store(self, fake_embed):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            # Seed a rule into the store
            rid, _ = memory.add_rule(
                conn,
                memory.Rule(
                    rule_text="Include evidence in findings",
                    scope="global",
                    is_meta=True,
                    assert_kind="require_field",
                    assert_field="evidence",
                    evidence_count=3,
                    confidence=0.9,
                ),
            )
            res = pipeline.review_task(conn, "pay", memory_on=True, llm=llm)
            assert res["memory_on"] is True
            # The seeded rule should appear in injected_rules
            injected_texts = [r["rule_text"] for r in res["injected_rules"]]
            assert "Include evidence in findings" in injected_texts
        finally:
            conn.close()

    def test_cost_breakdown(self, fake_embed):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            res = pipeline.review_task(conn, "pay", memory_on=True, llm=llm)
            cost = res["cost"]
            assert "memory_usd" in cost
            assert "generation_usd" in cost
            assert "total_usd" in cost
            # total = memory + generation (approximately)
            assert cost["total_usd"] == pytest.approx(
                cost["memory_usd"] + cost["generation_usd"], abs=0.001
            )
        finally:
            conn.close()

    def test_timing_breakdown(self, fake_embed):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            res = pipeline.review_task(conn, "pay", memory_on=True, llm=llm)
            t = res["timing"]
            assert "retrieval" in t
            assert "generation" in t
            assert "user_visible" in t
            # user_visible = retrieval + generation when memory is on
            assert t["user_visible"] == pytest.approx(
                t["retrieval"] + t["generation"]
            )
        finally:
            conn.close()

    def test_usage_accumulated(self, fake_embed):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            res = pipeline.review_task(conn, "pay", memory_on=True, llm=llm)
            u = res["usage"]
            assert "in_tok" in u
            assert "out_tok" in u
            assert "cache_read" in u
            assert "cache_write" in u
            assert "cache_hit_ratio" in u
        finally:
            conn.close()

    def test_no_rules_means_empty_injection(self, fake_embed):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            # No rules seeded, memory ON
            res = pipeline.review_task(conn, "pay", memory_on=True, llm=llm)
            assert res["injected_rules"] == []
        finally:
            conn.close()


# --- TestApplyFeedbackEdgeCases --------------------------------------------


class TestApplyFeedbackEdgeCases:
    def test_no_actions_returns_empty(self):
        conn = memory.connect(":memory:")
        llm = _mock_llm()
        try:
            findings = [{"file": "x.py", "line": 1, "category": "security",
                          "severity": "high", "message": "m", "evidence": "e"}]
            # payload with only "keep" actions -> no deletions/edits
            payload = [{"index": 0, "action": "keep"}]
            res = pipeline.apply_feedback(
                conn, "pay", findings=findings, payload=payload, llm=llm,
            )
            assert res["rules"] == []
            assert res["cost"]["memory_usd"] == 0.0
        finally:
            conn.close()
