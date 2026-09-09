"""Tests for the FastAPI web layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app import catalog, memory, pipeline
from app.llm import Usage
from app.web import app


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    return TestClient(app)


def _mock_llm(findings=None):
    m = MagicMock()
    findings = findings or [{
        "file": "svc/pay.py", "line": 2, "category": "security",
        "severity": "high", "message": "no validation",
        "evidence": "charge(uid, amt)",
    }]

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
    m.structured = MagicMock(return_value={"findings": findings})
    return m


class TestHealth:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True


class TestTasks:
    def test_tasks(self, client):
        r = client.get("/api/tasks")
        assert r.status_code == 200
        data = r.json()
        assert "tasks" in data
        ids = [t["task_id"] for t in data["tasks"]]
        assert "pay" in ids


class TestReview:
    def test_review_returns_findings(self, client):
        llm = _mock_llm()
        with patch("app.web.pipeline") as mock_pipe:
            mock_pipe.review_task.return_value = {
                "task": {"task_id": "pay", "path": "svc/pay.py",
                         "lang": "python", "area": "svc/pay.py",
                         "change_kind": "new-feature", "label": "test"},
                "findings": [],
                "injected_rules": [],
                "usage": {"in_tok": 100, "out_tok": 50,
                          "cache_write": 0, "cache_read": 0,
                          "cache_hit_ratio": 0.0},
                "timing": {"retrieval": 0.0, "generation": 0.0,
                           "user_visible": 0.0},
                "cost": {"memory_usd": 0.0, "generation_usd": 0.0,
                         "total_usd": 0.0},
                "memory_on": True,
            }
            r = client.post("/api/review", json={
                "task_id": "pay", "memory_on": True,
            })
        assert r.status_code == 200
        data = r.json()
        assert data["task"]["task_id"] == "pay"

    def test_review_unknown_task(self, client):
        r = client.post("/api/review", json={
            "task_id": "nonexistent", "memory_on": True,
        })
        assert r.status_code == 404


class TestFeedback:
    def test_feedback_no_changes(self, client):
        with patch("app.web.pipeline") as mock_pipe:
            mock_pipe.apply_feedback.return_value = {
                "rules": [],
                "actions": {"keep": 1, "delete": 0, "edit": 0},
                "t_distill": 0.0,
                "cost": {"memory_usd": 0.0},
                "note": "no deletions or edits",
            }
            r = client.post("/api/feedback", json={
                "task_id": "pay",
                "findings": [{"file": "x.py", "line": 1, "category": "security",
                               "severity": "high", "message": "m", "evidence": "e"}],
                "actions": [{"index": 0, "action": "keep"}],
            })
        assert r.status_code == 200
        data = r.json()
        assert data["rules"] == []

    def test_feedback_with_actions(self, client):
        with patch("app.web.pipeline") as mock_pipe:
            mock_pipe.apply_feedback.return_value = {
                "rules": [{"rule_text": "No style", "scope": "global",
                           "confidence": 0.9}],
                "actions": {"keep": 0, "delete": 1, "edit": 0},
                "t_distill": 0.1,
                "cost": {"memory_usd": 0.001},
            }
            r = client.post("/api/feedback", json={
                "task_id": "pay",
                "findings": [{"file": "x.py", "line": 1, "category": "style",
                               "severity": "low", "message": "m", "evidence": "e"}],
                "actions": [{"index": 0, "action": "delete", "note": "no nits"}],
            })
        assert r.status_code == 200


class TestRules:
    def test_list_rules(self, client):
        with patch("app.web.memory") as mock_mem:
            mock_mem.list_rules.return_value = []
            mock_mem.stats.return_value = {"active": 0, "meta": 0,
                                            "superseded": 0, "needs_review": 0,
                                            "avg_conf": 0.0}
            r = client.get("/api/rules")
        assert r.status_code == 200

    def test_reset_rules(self, client):
        with patch("app.web.db") as mock_db:
            mock_conn = MagicMock()
            mock_db.return_value = mock_conn
            r = client.delete("/api/rules")
        assert r.status_code == 200


class TestMetrics:
    def test_metrics(self, client):
        with patch("app.web.metrics") as mock_met:
            mock_met.summary.return_value = {
                "review": {"memory_on": {}, "memory_off": {}},
                "feedback": {},
                "totals": {},
            }
            mock_met.recent_turns.return_value = []
            r = client.get("/api/metrics")
        assert r.status_code == 200


class TestEvalAB:
    def test_run_ab(self, client):
        with patch("eval.run_ab.run") as mock_run:
            mock_run.return_value = {
                "task_ids": ["billing", "auth", "worker"],
                "repeats": 1,
                "seeded_rules": [],
                "off": {"rows": [], "totals": {"compliance_rate": 0.5, "passed": 1, "counted": 2, "leaks": 0,
                                                "avg_user_visible_s": 1.0, "avg_retrieval_s": 0.1,
                                                "total_cost_usd": 0.01, "memory_cost_usd": 0.001,
                                                "cache_read": 100, "cache_write": 10,
                                                "avg_findings": 2.0}},
                "on": {"rows": [], "totals": {"compliance_rate": 0.8, "passed": 4, "counted": 5, "leaks": 0,
                                               "avg_user_visible_s": 1.5, "avg_retrieval_s": 0.2,
                                               "total_cost_usd": 0.02, "memory_cost_usd": 0.002,
                                               "cache_read": 150, "cache_write": 20,
                                               "avg_findings": 3.0}},
                "per_rule": [],
                "judge_usage": {"calls": 0, "cost_usd": 0.0},
                "wall_clock_s": 5.0,
            }
            r = client.post("/api/eval/ab", json={"repeats": 1})
        assert r.status_code == 200


class TestIndex:
    def test_serves_static(self, client):
        r = client.get("/")
        assert r.status_code == 200
