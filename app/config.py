"""Central configuration: models, pricing, paths, credentials.

Every model id and price lives here so cost accounting has exactly one source
of truth. Model ids are pinned to the dated aliases that were verified working
against the proxy -- see MODEL_* notes below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "memory.db"
FIXTURE_REPO = ROOT / "fixtures" / "repo"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- credentials -----------------------------------------------------------

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL") or None
# The SDK accepts either variable. We read both so the project works whether
# the operator set an api key or an auth token.
AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get(
    "ANTHROPIC_API_KEY"
)


def require_credentials() -> None:
    """Fail fast with an actionable message rather than a 401 deep in a call."""
    if not AUTH_TOKEN:
        raise RuntimeError(
            "No credential found. Set ANTHROPIC_AUTH_TOKEN (or ANTHROPIC_API_KEY) "
            "in the environment or in a .env file. See .env.example."
        )


# --- models ----------------------------------------------------------------
#
# Verified against the proxy at build time:
#   claude-sonnet-4-6            -> OK
#   claude-haiku-4-5-20251001    -> OK
#   claude-haiku-4-5 (bare)      -> 503 model_not_found  (do not use)
#   claude-3-5-haiku-20241022    -> 400 upstream_error   (do not use)
#
# Bare `claude-haiku-4-5` fails on this proxy, so the dated alias is required.

MODEL_MAIN = "claude-sonnet-4-6"
MODEL_CHEAP = "claude-haiku-4-5-20251001"
MODEL_EMBED = "text-embedding-3-small"
EMBED_DIM = 1536


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    inp: float
    out: float

    def cost(
        self,
        in_tok: int = 0,
        out_tok: int = 0,
        cache_write: int = 0,
        cache_read: int = 0,
    ) -> float:
        """Cost in USD. Cache writes bill ~1.25x input, reads ~0.1x input."""
        return (
            in_tok * self.inp
            + out_tok * self.out
            + cache_write * self.inp * 1.25
            + cache_read * self.inp * 0.10
        ) / 1_000_000


PRICES: dict[str, Price] = {
    "claude-sonnet-4-6": Price(3.00, 15.00),
    "claude-haiku-4-5-20251001": Price(1.00, 5.00),
    "claude-opus-5": Price(5.00, 25.00),
    "claude-opus-4-8": Price(5.00, 25.00),
}

# text-embedding-3-small: $0.02 per 1M tokens.
EMBED_PRICE_PER_1M = 0.02


def price_for(model: str) -> Price:
    """Price lookup that degrades to the main model's price rather than KeyError."""
    return PRICES.get(model, PRICES[MODEL_MAIN])


def embed_cost(tokens: int) -> float:
    return tokens * EMBED_PRICE_PER_1M / 1_000_000


# --- agent tuning ----------------------------------------------------------

# Phase-1 ceiling. Measured: the model settles after ~3 tool calls but needs one
# further round to emit its closing non-tool turn, so a cap of 4 always tripped
# the guard (hit_round_cap=True) even on a clean run. 6 leaves headroom while
# still bounding a runaway loop.
MAX_TOOL_ROUNDS = 6
RETRIEVAL_TOP_K = 5  # non-meta rules injected per turn
DEDUP_SAME = 0.90  # cosine >= this  -> same rule, merge evidence
DEDUP_REVIEW = 0.75  # cosine in [REVIEW, SAME) -> flag for human review
SINGLE_EVIDENCE_CAP = 0.6  # hard confidence cap when evidence_count <= 2
LOW_CONFIDENCE = 0.5  # below this, downweight at injection time
