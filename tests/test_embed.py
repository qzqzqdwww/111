"""Tests for the embedding module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app import config
from app.embed import cosine, embed, embed_one, from_blob, to_blob


# --- embed -----------------------------------------------------------------


class TestEmbed:
    def test_empty_input(self):
        mat, tokens = embed([])
        assert mat.shape == (0, config.EMBED_DIM)
        assert tokens == 0

    def test_single_text(self, monkeypatch):
        fake_rows = [[0.1] * config.EMBED_DIM]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [{"embedding": r} for r in fake_rows],
            "usage": {"prompt_tokens": 5},
        }
        import httpx
        monkeypatch.setattr(httpx, "post", lambda *a, **kw: resp)

        mat, tokens = embed(["hello"])
        assert mat.shape == (1, config.EMBED_DIM)
        assert tokens == 5

    def test_dimension_mismatch_raises(self, monkeypatch):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [{"embedding": [0.1] * 10}],
            "usage": {"prompt_tokens": 1},
        }
        import httpx
        monkeypatch.setattr(httpx, "post", lambda *a, **kw: resp)

        with pytest.raises(Exception, match="expected dim"):
            embed(["hello"])

    def test_count_mismatch_raises(self, monkeypatch):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [],  # empty when we asked for 1
            "usage": {"prompt_tokens": 1},
        }
        import httpx
        monkeypatch.setattr(httpx, "post", lambda *a, **kw: resp)

        with pytest.raises(Exception, match="asked for 1"):
            embed(["hello"])

    def test_non_retryable_error(self, monkeypatch):
        resp = MagicMock()
        resp.status_code = 401
        resp.text = "unauthorized"
        import httpx
        monkeypatch.setattr(httpx, "post", lambda *a, **kw: resp)

        with pytest.raises(Exception, match="embeddings failed"):
            embed(["hello"])

    def test_retries_then_succeeds(self, monkeypatch):
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        fail_resp.text = "unavail"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {
            "data": [{"embedding": [0.1] * config.EMBED_DIM}],
            "usage": {"prompt_tokens": 1},
        }

        import httpx
        calls = [fail_resp, ok_resp]
        monkeypatch.setattr(httpx, "post", lambda *a, **kw: calls.pop(0))

        mat, tokens = embed(["hello"])
        assert mat.shape == (1, config.EMBED_DIM)


class TestEmbedOne:
    def test_returns_vector_and_tokens(self, monkeypatch):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [{"embedding": [0.2] * config.EMBED_DIM}],
            "usage": {"prompt_tokens": 3},
        }
        import httpx
        monkeypatch.setattr(httpx, "post", lambda *a, **kw: resp)

        vec, tokens = embed_one("test text")
        assert vec.shape == (config.EMBED_DIM,)
        assert tokens == 3


# --- serialization ---------------------------------------------------------


class TestBlob:
    def test_roundtrip(self):
        v = np.random.randn(config.EMBED_DIM).astype(np.float32)
        blob = to_blob(v)
        restored = from_blob(blob)
        np.testing.assert_array_equal(restored, v)

    def test_to_blob_float32(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        blob = to_blob(v)
        assert isinstance(blob, bytes)


# --- cosine ----------------------------------------------------------------


class TestCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine(v, v.reshape(1, -1))[0] == pytest.approx(1.0)

    def test_orthogonal(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        m = np.array([[0.0, 1.0]], dtype=np.float32)
        assert cosine(v, m)[0] == pytest.approx(0.0)

    def test_empty_matrix(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        m = np.zeros((0, 2), dtype=np.float32)
        result = cosine(v, m)
        assert result.shape == (0,)

    def test_against_matrix(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        m = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sims = cosine(v, m)
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)
