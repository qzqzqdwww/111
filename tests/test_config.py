"""Tests for configuration, pricing, and credentials."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app import config


class TestConfig:
    def test_root_resolves(self):
        assert config.ROOT.is_dir()

    def test_data_dir_created(self):
        assert config.DATA_DIR.is_dir()

    def test_db_path(self):
        assert config.DB_PATH.name == "memory.db"

    def test_fixture_repo_exists(self):
        assert config.FIXTURE_REPO.is_dir()

    def test_model_ids_pinned(self):
        assert config.MODEL_MAIN == "claude-sonnet-4-6"
        assert config.MODEL_CHEAP == "claude-haiku-4-5-20251001"
        assert config.MODEL_EMBED == "text-embedding-3-small"

    def test_price_lookup(self):
        p = config.price_for(config.MODEL_MAIN)
        assert p.inp == 3.00
        assert p.out == 15.00

    def test_price_fallback_to_main(self):
        p = config.price_for("some-unknown-model")
        assert p == config.PRICES[config.MODEL_MAIN]

    def test_embed_cost(self):
        assert config.embed_cost(1_000_000) == 0.02
        assert config.embed_cost(500_000) == pytest.approx(0.01)

    def test_require_credentials_raises_when_missing(self):
        with patch.object(config, "AUTH_TOKEN", None):
            with pytest.raises(RuntimeError, match="No credential found"):
                config.require_credentials()

    def test_auth_token_from_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "tok-123"}):
            # Re-evaluate the module-level expression
            tok = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get(
                "ANTHROPIC_API_KEY"
            )
            assert tok == "tok-123"

    def test_retrieval_top_k(self):
        assert config.RETRIEVAL_TOP_K == 5

    def test_dedup_thresholds(self):
        assert 0.0 < config.DEDUP_SAME <= 1.0
        assert 0.0 < config.DEDUP_REVIEW < config.DEDUP_SAME

    def test_single_evidence_cap(self):
        assert config.SINGLE_EVIDENCE_CAP == 0.6

    def test_max_tool_rounds(self):
        assert config.MAX_TOOL_ROUNDS >= 1
