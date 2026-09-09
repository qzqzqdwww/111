"""Tests for the task catalog."""

from __future__ import annotations

import pytest

from app import catalog


class TestCatalog:
    def test_ids_returns_all(self):
        ids = catalog.ids()
        assert set(ids) == {"pay", "billing", "auth", "worker"}

    def test_get_known(self):
        t = catalog.get("pay")
        assert t.task_id == "pay"
        assert t.lang == "python"
        assert t.change_kind == "new-feature"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="unknown task"):
            catalog.get("does-not-exist")

    @pytest.mark.parametrize("task_id", ["pay", "billing", "auth", "worker"])
    def test_query_text_shape(self, task_id):
        t = catalog.get(task_id)
        qt = t.query_text
        assert task_id in qt or t.path in qt
        assert t.lang in qt
        assert t.change_kind in qt

    def test_all_tasks_have_required_fields(self):
        for task_id in catalog.ids():
            t = catalog.get(task_id)
            assert t.task_id
            assert t.path
            assert t.lang
            assert t.change_kind
            assert t.query_text
            assert t.area

    def test_pay_path(self):
        t = catalog.get("pay")
        assert "pay" in t.path

    def test_auth_path(self):
        t = catalog.get("auth")
        assert "auth" in t.path

    def test_billing_path(self):
        t = catalog.get("billing")
        assert "billing" in t.path

    def test_worker_go_lang(self):
        t = catalog.get("worker")
        assert t.lang == "go"

    def test_worker_has_query_text(self):
        t = catalog.get("worker")
        assert "go" in t.query_text or "worker" in t.query_text

    def test_task_area_matches_path(self):
        for task_id in catalog.ids():
            t = catalog.get(task_id)
            assert t.area in t.path or t.path in t.area
