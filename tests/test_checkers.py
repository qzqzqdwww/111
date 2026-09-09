"""Tests for the compliance checkers in eval/checkers.py."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest

from eval import checkers


# --- checker functions ------------------------------------------------------


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

    def test_multiple_findings_counts_all(self):
        verdict = checkers._check_forbid_category(
            {"id": 1, "rule_text": "no style", "assert_kind": "forbid_category", "assert_cat": "style"},
            [{"category": "style"}, {"category": "style"}, {"category": "security"}],
        )
        assert verdict.outcome == checkers.FAIL
        assert verdict.n_relevant == 2

    def test_empty_rule_text_still_works(self):
        verdict = checkers._check_forbid_category(
            {"id": 1, "rule_text": "", "assert_kind": "forbid_category", "assert_cat": "security"},
            [{"category": "security"}],
        )
        assert verdict.outcome == checkers.FAIL


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

    def test_mixed_severity_fails(self):
        verdict = checkers._check_require_severity(
            {"id": 1, "rule_text": "high val", "assert_kind": "require_severity_for_category", "assert_cat": "security", "assert_sev": "high"},
            [{"category": "security", "severity": "high"}, {"category": "security", "severity": "low"}],
        )
        assert verdict.outcome == checkers.FAIL
        assert verdict.n_relevant == 2


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

    def test_empty_string_field_counts_as_missing(self):
        verdict = checkers._check_require_field(
            {"id": 1, "rule_text": "need evidence", "assert_kind": "require_field", "assert_field": "evidence"},
            [{"evidence": "", "category": "x"}],
        )
        assert verdict.outcome == checkers.FAIL

    def test_non_dict_skipped(self):
        verdict = checkers._check_require_field(
            {"id": 1, "rule_text": "need evidence", "assert_kind": "require_field", "assert_field": "evidence"},
            ["not-a-dict", {"evidence": "x", "category": "y"}],
        )
        assert verdict.outcome == checkers.PASS
        assert verdict.n_relevant == 1
        assert verdict.n_relevant == 1


# --- compliance_rate --------------------------------------------------------


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

    def test_mixed_outcomes(self):
        verdicts = [
            checkers.Verdict(1, "r", "k", checkers.PASS, ""),
            checkers.Verdict(2, "r", "k", checkers.FAIL, ""),
            checkers.Verdict(3, "r", "k", checkers.VACUOUS, ""),
        ]
        rate, passed, counted = checkers.compliance_rate(verdicts)
        assert rate == 0.5
        assert passed == 1
        assert counted == 2

    def test_all_fail(self):
        verdicts = [
            checkers.Verdict(1, "r", "k", checkers.FAIL, ""),
            checkers.Verdict(2, "r", "k", checkers.FAIL, ""),
        ]
        rate, passed, counted = checkers.compliance_rate(verdicts)
        assert rate == 0.0
        assert passed == 0
        assert counted == 2


# --- judge LLM path ---------------------------------------------------------


class TestCheckWithJudge:
    def test_judge_returns_pass(self):
        rule = {
            "id": 1, "rule_text": "Always include evidence",
            "assert_kind": "none", "assert_field": "evidence",
        }
        findings = [{"file": "x.py", "line": 1, "category": "security",
                      "severity": "high", "message": "m", "evidence": "e"}]

        mock_llm = MagicMock()
        mock_llm.structured.return_value = {"passed": True, "reason": "ok"}
        mock_usage = MagicMock()

        verdict = checkers._check_with_judge(rule, findings, llm=mock_llm, usage=mock_usage)
        assert verdict.outcome == checkers.PASS
        assert verdict.subjective is True

    def test_judge_returns_fail(self):
        rule = {
            "id": 1, "rule_text": "Always include evidence",
            "assert_kind": "none", "assert_field": "evidence",
        }
        findings = [{"file": "x.py", "line": 1, "category": "security",
                      "severity": "high", "message": "m", "evidence": ""}]

        mock_llm = MagicMock()
        mock_llm.structured.return_value = {"passed": False, "reason": "missing evidence"}
        mock_usage = MagicMock()

        verdict = checkers._check_with_judge(rule, findings, llm=mock_llm, usage=mock_usage)
        assert verdict.outcome == checkers.FAIL
        assert verdict.n_relevant == 1
        assert verdict.subjective is True

    def test_judge_error_falls_back(self):
        rule = {
            "id": 1, "rule_text": "Always include evidence",
            "assert_kind": "none", "assert_field": "evidence",
        }
        findings = [{"file": "x.py", "line": 1, "category": "security",
                      "severity": "high", "message": "m", "evidence": "e"}]

        mock_llm = MagicMock()
        mock_llm.structured.side_effect = Exception("api error")
        mock_usage = MagicMock()

        verdict = checkers._check_with_judge(rule, findings, llm=mock_llm, usage=mock_usage)
        # Should not raise; falls back to VACUOUS with subjective=True
        assert verdict.outcome == checkers.VACUOUS
        assert verdict.subjective is True

    def test_judge_passes_reason_in_detail(self):
        rule = {
            "id": 1, "rule_text": "Check for X",
            "assert_kind": "none", "assert_field": "evidence",
        }
        findings = [{"file": "x.py", "line": 1, "category": "security",
                      "severity": "high", "message": "m", "evidence": "e"}]

        mock_llm = MagicMock()
        mock_llm.structured.return_value = {"passed": True, "reason": "findings match"}
        mock_usage = MagicMock()

        verdict = checkers._check_with_judge(rule, findings, llm=mock_llm, usage=mock_usage)
        assert "findings match" in verdict.detail


# --- Verdict dataclass -------------------------------------------------------


class TestVerdict:
    def test_create(self):
        v = checkers.Verdict(1, "r", "k", checkers.PASS, "")
        assert v.rule_id == 1
        assert v.outcome == checkers.PASS

    def test_outcomes_are_strings(self):
        assert checkers.PASS == "pass"
        assert checkers.FAIL == "fail"
        assert checkers.VACUOUS == "vacuous"
