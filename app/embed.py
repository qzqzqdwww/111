"""Embeddings via the proxy's OpenAI-compatible /v1/embeddings route.

The Anthropic SDK has no embeddings endpoint, so this goes over raw httpx to
the same base URL. Verified: text-embedding-3-small returns 200 with dim=1536;
~1.5s for one input, ~2.5s for a batch of ten. Batch whenever possible -- ten
inputs cost barely more wall-clock than one.

Vectors are L2-normalized on the way out, so cosine similarity is a plain dot
product everywhere downstream.
"""

from __future__ import annotations

import random
import time

import httpx
import numpy as np

from . import config

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 529}


class EmbedError(RuntimeError):
    pass


def _endpoint() -> str:
    base = (config.BASE_URL or "https://api.anthropic.com").rstrip("/")
    return f"{base}/v1/embeddings"


def embed(
    texts: list[str], *, max_retries: int = 3, timeout: float = 40.0
) -> tuple[np.ndarray, int]:
    """Embed a batch of texts.

    Returns (matrix of shape (len(texts), EMBED_DIM) float32, prompt_tokens).
    Rows are L2-normalized. Empty input returns an empty matrix and 0 tokens.
    """
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32), 0
    config.require_credentials()

    payload = {"model": config.MODEL_EMBED, "input": texts}
    headers = {
        "authorization": f"Bearer {config.AUTH_TOKEN}",
        "content-type": "application/json",
    }

    last: Exception | str | None = None
    for attempt in range(max_retries):
        try:
            r = httpx.post(_endpoint(), headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                body = r.json()
                rows = [d["embedding"] for d in body["data"]]
                if len(rows) != len(texts):
                    raise EmbedError(
                        f"asked for {len(texts)} embeddings, got {len(rows)}"
                    )
                mat = np.asarray(rows, dtype=np.float32)
                if mat.shape[1] != config.EMBED_DIM:
                    raise EmbedError(
                        f"expected dim {config.EMBED_DIM}, got {mat.shape[1]}"
                    )
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                # Guard against a zero vector making the row NaN.
                norms[norms == 0] = 1.0
                usage = body.get("usage") or {}
                tokens = int(
                    usage.get("prompt_tokens") or usage.get("total_tokens") or 0
                )
                return (mat / norms).astype(np.float32), tokens
            last = f"{r.status_code} {r.text[:160]}"
            if r.status_code not in _RETRY_STATUS:
                raise EmbedError(f"embeddings failed: {last}")
        except httpx.HTTPError as e:
            last = e
        if attempt < max_retries - 1:
            time.sleep(min(2**attempt + random.random(), 8.0))
    raise EmbedError(f"embeddings failed after {max_retries} attempts: {last}")


def embed_one(text: str) -> tuple[np.ndarray, int]:
    """Embed a single string. Returns (1-D vector of EMBED_DIM, tokens)."""
    mat, tokens = embed([text])
    return mat[0], tokens


# --- serialization for SQLite BLOB columns --------------------------------


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one normalized vector against normalized rows.

    Both sides are already unit-length, so this is just a dot product.
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ np.asarray(query, dtype=np.float32)
