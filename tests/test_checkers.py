"""Tests for the compliance checkers in eval/checkers.py."""

from __future__ import annotations

import pytest

from eval import checkers


class TestForbidCategory:
    def test_pass_when_absent(self):
        verdict = checkers._check_forbid_category(
            {"id": 1, "rule_text": "no style", "assert_kind": "forbid_category", "assert_cat": "style"},
            [{"category": "security"}],
        )
        assert verdict.outcome == checkers.PASS

    def test_fail_when_present(self):
        verdict = checkers._check_forbid_category(
            {"id": 1, "rule_text": "no style", "assert_kind": "forbid_category", "assert_cat": "style"},
            [{"category": "style", "severity": "low"}],
        )
        assert verdict.outcome == checkers.FAIL
        assert verdict.n_relevant == 1

    def test_vacuous_not_used_here(self):
        verdict = checkers._check_forbid_category(
            {"id": 1, "rule_text": "no style", "assert_kind": "forbid_category", "assert_cat": "style"},
            [],
        )
        assert verdict.outcome == checkers.PASS
        assert verdict.n_relevant == 0


class TestRequireSeverity:
    def test_pass_when_all_high(self):
        verdict = checkers._check_require_severity(
            {"id": 1, "rule_text": "high val", "assert_kind": "require_severity_for_category", "assert_cat": "validation", "assert_sev": "high"},
            [{"category": "validation", "severity": "high"}],
        )
        assert verdict.outcome == checkers.PASS

    def test_fail_when_below_floor(self):
        verdict = checkers._check_require_severity(
            {"id": 1, "rule_text": "high val", "assert_kind": "require_severity_for_category", "assert_cat": "validation", "assert_sev": "high"},
            [{"category": "validation", "severity": "medium"}],
        )
        assert verdict.outcome == checkers.FAIL

    def test_vacuous_when_no_relevant_findings(self):
        verdict = checkers._check_require_severity(
            {"id": 1, "rule_text": "high val", "assert_kind": "require_severity_for_category", "assert_cat": "validation", "assert_sev": "high"},
            [{"category": "security", "severity": "high"}],
        )
        assert verdict.outcome == checkers.VACUOUS


class TestRequireField:
    def test_pass_when_all_populated(self):
        verdict = checkers._check_require_field(
            {"id": 1, "rule_text": "need evidence", "assert_kind": "require_field", "assert_field": "evidence"},
            [{"evidence": "some code", "category": "x"}],
        )
        assert verdict.outcome == checkers.PASS

    def test_fail_when_field_missing(self):
        verdict = checkers._check_require_field(
            {"id": 1, "rule_text": "need evidence", "assert_kind": "require_field", "assert_field": "evidence"},
            [{"category": "x"}],
        )
        assert verdict.outcome == checkers.FAIL
        assert verdict.n_relevant == 1

    def test_vacuous_on_empty_findings(self):
        verdict = checkers._check_require_field(
            {"id": 1, "rule_text": "need evidence", "assert_kind": "require_field", "assert_field": "evidence"},
            [],
        )
        assert verdict.outcome == checkers.VACUOUS


class TestComplianceRate:
    def test_all_pass(self):
        verdicts = [
            checkers.Verdict(1, "r", "k", checkers.PASS, ""),
            checkers.Verdict(2, "r", "k", checkers.PASS, ""),
        ]
        rate, passed, counted = checkers.compliance_rate(verdicts)
        assert rate == 1.0
        assert passed == 2
        assert counted == 2

    def test_vacuous_excluded(self):
        verdicts = [
            checkers.Verdict(1, "r", "k", checkers.PASS, ""),
            checkers.Verdict(2, "r", "k", checkers.VACUOUS, ""),
        ]
        rate, passed, counted = checkers.compliance_rate(verdicts)
        assert rate == 1.0
        assert counted == 1

    def test_empty(self):
        rate, passed, counted = checkers.compliance_rate([])
        assert rate == 0.0
        assert passed == 0
        assert counted == 0
