"""Tests for the metrics module."""

from __future__ import annotations

import sqlite3

import pytest

from app import metrics, memory
from app.llm import Usage


@pytest.fixture()
def conn():
    c = memory.connect(":memory:")
    yield c
    c.close()


class TestRecordTurn:
    def test_inserts_row(self, conn):
        uid = Usage(in_tok=10, out_tok=5, cost_usd=0.01)
        tid = metrics.record_turn(
            conn, task_id="pay", memory_on=True, phase="review",
            t_retrieval=0.1, t_generation=0.5, t_distill=0.0,
            t_user_visible=0.6, usage=uid, memory_cost_usd=0.001,
            n_rules_injected=2, n_findings=3,
        )
        assert tid > 0
        row = conn.execute("SELECT * FROM turns WHERE id=?", (tid,)).fetchone()
        assert row["task_id"] == "pay"
        assert row["n_findings"] == 3
        assert row["n_rules_injected"] == 2


class TestSummary:
    def test_empty(self, conn):
        s = metrics.summary(conn)
        assert s["review"]["memory_on"]["turns"] == 0
        assert s["review"]["memory_off"]["turns"] == 0
        assert s["feedback"]["turns"] == 0
        assert s["totals"]["total_cost_usd"] == 0

    def test_review_turns_grouped(self, conn):
        uid = Usage(in_tok=10, out_tok=5, cost_usd=0.01)
        metrics.record_turn(
            conn, task_id="pay", memory_on=True, phase="review",
            t_retrieval=0.1, t_generation=0.5, t_distill=0.0,
            t_user_visible=0.6, usage=uid, memory_cost_usd=0.001,
            n_rules_injected=2, n_findings=3,
        )
        metrics.record_turn(
            conn, task_id="pay", memory_on=False, phase="review",
            t_retrieval=0.0, t_generation=0.4, t_distill=0.0,
            t_user_visible=0.4, usage=uid, memory_cost_usd=0.0,
            n_rules_injected=0, n_findings=2,
        )
        s = metrics.summary(conn)
        assert s["review"]["memory_on"]["turns"] == 1
        assert s["review"]["memory_off"]["turns"] == 1
        assert s["review"]["memory_on"]["avg_findings"] == 3.0
        assert s["review"]["memory_off"]["avg_findings"] == 2.0

    def test_feedback_turns(self, conn):
        uid = Usage(in_tok=10, out_tok=5, cost_usd=0.002)
        metrics.record_turn(
            conn, task_id="pay", memory_on=True, phase="feedback",
            t_retrieval=0.0, t_generation=0.0, t_distill=0.3,
            t_user_visible=0.0, usage=uid, memory_cost_usd=0.002,
            n_rules_injected=0, n_findings=3,
        )
        s = metrics.summary(conn)
        assert s["feedback"]["turns"] == 1
        assert s["feedback"]["avg_distill"] == pytest.approx(0.3)

    def test_totals(self, conn):
        uid = Usage(in_tok=100, out_tok=50, cache_write=20, cache_read=80,
                    embed_tok=10, cost_usd=0.01)
        metrics.record_turn(
            conn, task_id="pay", memory_on=True, phase="review",
            t_retrieval=0.1, t_generation=0.5, t_distill=0.0,
            t_user_visible=0.6, usage=uid, memory_cost_usd=0.001,
            n_rules_injected=2, n_findings=3,
        )
        s = metrics.summary(conn)
        assert s["totals"]["total_cost_usd"] == pytest.approx(0.01)
        assert s["totals"]["memory_cost_usd"] == pytest.approx(0.001)
        assert s["totals"]["cache_read"] == 80
        assert s["totals"]["cache_write"] == 20

    def test_cache_hit_ratio(self, conn):
        uid = Usage(cache_read=70, cache_write=20, in_tok=10, cost_usd=0.01)
        metrics.record_turn(
            conn, task_id="pay", memory_on=True, phase="review",
            t_retrieval=0.0, t_generation=0.0, t_distill=0.0,
            t_user_visible=0.0, usage=uid, memory_cost_usd=0.0,
            n_rules_injected=0, n_findings=1,
        )
        s = metrics.summary(conn)
        assert s["totals"]["cache_hit_ratio"] == pytest.approx(70 / 100)

    def test_memory_cost_share(self, conn):
        uid = Usage(in_tok=10, cost_usd=0.05)
        metrics.record_turn(
            conn, task_id="pay", memory_on=True, phase="review",
            t_retrieval=0.0, t_generation=0.0, t_distill=0.0,
            t_user_visible=0.0, usage=uid, memory_cost_usd=0.01,
            n_rules_injected=1, n_findings=1,
        )
        s = metrics.summary(conn)
        assert s["totals"]["memory_cost_share"] == pytest.approx(0.01 / 0.05)


class TestRecentTurns:
    def test_returns_limit(self, conn):
        uid = Usage(cost_usd=0.01)
        for i in range(5):
            metrics.record_turn(
                conn, task_id="pay", memory_on=True, phase="review",
                t_retrieval=0.0, t_generation=0.0, t_distill=0.0,
                t_user_visible=0.0, usage=uid, memory_cost_usd=0.0,
                n_rules_injected=0, n_findings=1,
            )
        recent = metrics.recent_turns(conn, limit=3)
        assert len(recent) == 3
