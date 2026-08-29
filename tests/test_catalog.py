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
