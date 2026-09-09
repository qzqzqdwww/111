"""Rule-compliance checkers: did the output actually honor each rule?

This is what turns grading criterion #3 ("memory effectiveness and whether it is
used accurately") into a number instead of an opinion.

Deterministic first. Every assertion kind below is evaluated against the closed
category/severity vocabularies in app.schemas, so no model is involved. Only
`assert_kind == "none"` -- genuinely subjective rules -- falls back to an LLM
judge, and those verdicts are labelled subjective in the report.

The three-valued verdict matters more than it looks:

    PASS       the rule was exercised and honored
    FAIL       the rule was exercised and violated
    VACUOUS    the rule never applied to this output

VACUOUS exists because of a measured problem. A rule like "never report style
nits" is trivially satisfied by an output that had no style findings for any
reason, so counting it as PASS would inflate compliance with cases where memory
did nothing. It also cuts the other way: the memory-OFF baseline was measured
emitting style nits on one run and none on another, so a rule can look honored
purely by luck. Only non-vacuous results are counted in the headline rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app import config, schemas
from app.llm import LLM, Usage, shared

PASS = "pass"
FAIL = "fail"
VACUOUS = "vacuous"


@dataclass
class Verdict:
    rule_id: int | None
    rule_text: str
    assert_kind: str
    outcome: str  # pass | fail | vacuous
    detail: str
    subjective: bool = False
    n_relevant: int = 0

    @property
    def counted(self) -> bool:
        """Vacuous results are excluded from the compliance rate."""
        return self.outcome in (PASS, FAIL)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_text": self.rule_text,
            "assert_kind": self.assert_kind,
            "outcome": self.outcome,
            "detail": self.detail,
            "subjective": self.subjective,
            "n_relevant": self.n_relevant,
        }


def _sev_rank(s: str) -> int:
    return schemas.SEVERITY_RANK.get(schemas.normalize_severity(s), 1)


# --- deterministic checkers ----------------------------------------------


def _check_forbid_category(rule: dict, findings: list[dict]) -> Verdict:
    cat = rule.get("assert_cat")
    hits = [f for f in findings if f.get("category") == cat]
    if hits:
        return Verdict(
            rule.get("id"),
            rule["rule_text"],
            rule["assert_kind"],
            FAIL,
            f"{len(hits)} finding(s) in forbidden category {cat!r}",
            n_relevant=len(hits),
        )
    # No findings in the forbidden category. That is compliance only if the
    # model actually had something to suppress; we cannot know that from one
    # output, so this counts as a real PASS but with n_relevant=0 recorded so a
    # reader can see it was not a hard-won pass.
    return Verdict(
        rule.get("id"),
        rule["rule_text"],
        rule["assert_kind"],
        PASS,
        f"no findings in forbidden category {cat!r}",
        n_relevant=0,
    )


def _check_forbid_below_severity(rule: dict, findings: list[dict]) -> Verdict:
    cat, floor = rule.get("assert_cat"), rule.get("assert_sev") or "medium"
    relevant = [f for f in findings if f.get("category") == cat]
    bad = [f for f in relevant if _sev_rank(f.get("severity", "")) < _sev_rank(floor)]
    if bad:
        return Verdict(
            rule.get("id"),
            rule["rule_text"],
            rule["assert_kind"],
            FAIL,
            f"{len(bad)} {cat!r} finding(s) below severity floor {floor!r}",
            n_relevant=len(relevant),
        )
    return Verdict(
        rule.get("id"),
        rule["rule_text"],
        rule["assert_kind"],
        PASS,
        f"all {cat!r} findings at or above {floor!r}",
        n_relevant=len(relevant),
    )


def _check_require_severity(rule: dict, findings: list[dict]) -> Verdict:
    cat, floor = rule.get("assert_cat"), rule.get("assert_sev") or "high"
    relevant = [f for f in findings if f.get("category") == cat]
    if not relevant:
        # The rule says "when you report X, report it at >= S". An output with no
        # X findings neither honors nor violates it.
        return Verdict(
            rule.get("id"),
            rule["rule_text"],
            rule["assert_kind"],
            VACUOUS,
            f"no {cat!r} findings, so the severity requirement never applied",
            n_relevant=0,
        )
    bad = [f for f in relevant if _sev_rank(f.get("severity", "")) < _sev_rank(floor)]
    if bad:
        return Verdict(
            rule.get("id"),
            rule["rule_text"],
            rule["assert_kind"],
            FAIL,
            f"{len(bad)}/{len(relevant)} {cat!r} finding(s) below required {floor!r}",
            n_relevant=len(relevant),
        )
    return Verdict(
        rule.get("id"),
        rule["rule_text"],
        rule["assert_kind"],
        PASS,
        f"all {len(relevant)} {cat!r} finding(s) at {floor!r} or above",
        n_relevant=len(relevant),
    )


def _check_require_field(rule: dict, findings: list[dict]) -> Verdict:
    field = rule.get("assert_field") or "evidence"
    if not findings:
        return Verdict(
            rule.get("id"),
            rule["rule_text"],
            rule["assert_kind"],
            VACUOUS,
            "no findings to check",
            n_relevant=0,
        )
    missing = []
    relevant = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        v = f.get(field)
        if field == "line":
            ok = isinstance(v, int) and v > 0
        else:
            ok = bool(v and str(v).strip())
        if not ok:
            missing.append(f)
        relevant.append(f)
    if missing:
        return Verdict(
            rule.get("id"),
            rule["rule_text"],
            rule["assert_kind"],
            FAIL,
            f"{len(missing)}/{len(relevant)} finding(s) missing {field!r}",
            n_relevant=len(relevant),
        )
    return Verdict(
        rule.get("id"),
        rule["rule_text"],
        rule["assert_kind"],
        PASS,
        f"all {len(relevant)} finding(s) populate {field!r}",
        n_relevant=len(relevant),
    )


_DETERMINISTIC = {
    "forbid_category": _check_forbid_category,
    "forbid_category_below_severity": _check_forbid_below_severity,
    "require_severity_for_category": _check_require_severity,
    "require_field": _check_require_field,
}


# --- LLM judge (subjective fallback) ------------------------------------

JUDGE_PROMPT = """Decide whether a code review honored a team convention.

Convention:
  {rule_text}

The review produced these findings:
{findings}

Answer with passed=true if the findings are consistent with the convention, and
passed=false if they contradict it. If the convention simply did not apply to
this change, answer passed=true and say so in the reason.

Judge only the convention stated above. Do not evaluate overall review quality."""


def _check_with_judge(
    rule: dict,
    findings: list[dict],
    *,
    llm: LLM | None = None,
    usage: Usage | None = None,
) -> Verdict:
    llm = llm or shared()
    rendered = (
        "\n".join(
            f"  - [{f.get('category')}/{f.get('severity')}] {f.get('message','')[:140]}"
            for f in findings
        )
        or "  (no findings)"
    )
    try:
        out = llm.structured(
            model=config.MODEL_CHEAP,
            tool=schemas.EMIT_JUDGE_TOOL,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        rule_text=rule["rule_text"], findings=rendered
                    ),
                }
            ],
            max_tokens=400,
            usage=usage,
        )
    except Exception as e:  # noqa: BLE001 - a judge failure must not fail the run
        return Verdict(
            rule.get("id"),
            rule["rule_text"],
            rule.get("assert_kind", "none"),
            VACUOUS,
            f"judge unavailable: {str(e)[:100]}",
            subjective=True,
        )
    passed = bool(out.get("passed"))
    return Verdict(
        rule.get("id"),
        rule["rule_text"],
        rule.get("assert_kind", "none"),
        PASS if passed else FAIL,
        str(out.get("reason", ""))[:200],
        subjective=True,
        n_relevant=len(findings),
    )


# --- public API ---------------------------------------------------------


def check_rule(
    rule: dict,
    findings: list[dict],
    *,
    llm: LLM | None = None,
    usage: Usage | None = None,
    allow_judge: bool = True,
) -> Verdict:
    """Check one rule against one review's findings."""
    kind = rule.get("assert_kind") or "none"
    fn = _DETERMINISTIC.get(kind)
    if fn is not None:
        return fn(rule, findings)
    if not allow_judge:
        return Verdict(
            rule.get("id"),
            rule["rule_text"],
            kind,
            VACUOUS,
            "subjective rule, judge disabled",
            subjective=True,
        )
    return _check_with_judge(rule, findings, llm=llm, usage=usage)


def check_all(
    rules: list[dict],
    findings: list[dict],
    *,
    llm: LLM | None = None,
    usage: Usage | None = None,
    allow_judge: bool = True,
) -> list[Verdict]:
    return [
        check_rule(r, findings, llm=llm, usage=usage, allow_judge=allow_judge)
        for r in rules
    ]


def compliance_rate(verdicts: list[Verdict]) -> tuple[float, int, int]:
    """Return (rate, n_passed, n_counted) over non-vacuous verdicts only."""
    counted = [v for v in verdicts if v.counted]
    if not counted:
        return 0.0, 0, 0
    passed = sum(1 for v in counted if v.outcome == PASS)
    return round(passed / len(counted), 4), passed, len(counted)
