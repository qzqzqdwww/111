"""Shared pytest fixtures."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from app import memory as memory_mod


@pytest.fixture()
def conn():
    """In-memory SQLite store, freshly initialized for each test."""
    c = memory_mod.connect(":memory:")
    yield c
    c.close()


def _fake_embed(text: str, dim: int = None) -> tuple[np.ndarray, int]:
    """Deterministic fake embedding with good separation between texts.

    Uses a positional hash: each byte of the MD5 digest sets a different
    dimension, so different texts scatter across the hypersphere and cosine
    similarity between distinct texts stays well below the dedup threshold.
    """
    dim = dim or memory_mod.config.EMBED_DIM
    h = hashlib.md5(text.encode()).digest()
    v = np.zeros(dim, dtype=np.float32)
    # Map each hash byte to a dimension, scaled to [-1, 1]
    for i, b in enumerate(h):
        idx = b % dim
        v[idx] += (b - 128) / 128.0
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        v[0] = 1.0
        return v, len(text.split())
    return (v / norm).astype(np.float32), len(text.split())


@pytest.fixture()
def fake_embed(monkeypatch):
    """Patch ``embed.embed_one`` with a deterministic fake."""
    def _fake(text: str):
        return _fake_embed(text)
    monkeypatch.setattr(memory_mod.embed, "embed_one", _fake)
    return _fake
