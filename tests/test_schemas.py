"""Tests for closed-vocabulary schemas and normalization helpers."""

from __future__ import annotations

import pytest

from app import schemas


class TestNormalizeCategory:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("correctness", "correctness"),
            ("Correctness", "correctness"),
            ("error-handling", "error-handling"),
            ("Error Handling", "error-handling"),
            ("code-quality", "style"),
            ("nit", "style"),
            ("docs", "style"),
            ("bug", "correctness"),
            ("logging", "observability"),
            ("resilience", "reliability"),
            ("", "correctness"),
            ("totally-unknown", "correctness"),
        ],
    )
    def test_known_and_aliases(self, raw, expected):
        assert schemas.normalize_category(raw) == expected


class TestNormalizeSeverity:
    def test_literal(self):
        assert schemas.normalize_severity("high") == "high"

    def test_aliases(self):
        assert schemas.normalize_severity("critical") == "high"
        assert schemas.normalize_severity("major") == "high"
        assert schemas.normalize_severity("minor") == "low"
        assert schemas.normalize_severity("info") == "low"

    def test_default(self):
        assert schemas.normalize_severity("") == "medium"
        assert schemas.normalize_severity("bogus") == "medium"


class TestClosedVocabularies:
    def test_categories_exhaustive(self):
        assert set(schemas.CATEGORIES) == {
            "correctness",
            "error-handling",
            "security",
            "validation",
            "reliability",
            "observability",
            "performance",
            "style",
        }

    def test_severities_exhaustive(self):
        assert schemas.SEVERITIES == ["low", "medium", "high"]

    def test_assert_kinds_exhaustive(self):
        assert schemas.ASSERT_KINDS == [
            "forbid_category",
            "forbid_category_below_severity",
            "require_severity_for_category",
            "require_field",
            "none",
        ]

    def test_findings_schema_requires_all_fields(self):
        schema = schemas.FINDINGS_SCHEMA["properties"]["findings"]["items"]
        required = set(schema["required"])
        assert required == {"file", "line", "category", "severity", "message", "evidence"}

    def test_emit_findings_tool_shape(self):
        tool = schemas.EMIT_FINDINGS_TOOL
        assert tool["name"] == "emit_findings"
        assert tool["input_schema"] == schemas.FINDINGS_SCHEMA
