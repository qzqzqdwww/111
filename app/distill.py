"""Distillation: structured reviewer feedback -> durable team conventions.

Feedback is captured as per-finding ACTIONS rather than free text:

    keep    the finding was right and wanted
    delete  the reviewer does not want this class of thing reported
    edit    the class is wanted but must be reported differently

That is a much cleaner signal than diffing two blobs of prose, and it makes
"which signal is strongest" a matter of semantics rather than guesswork.

Two prompt properties were established by measurement, not taste:

1. A first version of this prompt produced NO rule at all for the deleted
   findings -- the single most informative action. Stating "a deleted finding is
   the strongest signal; emit rules for deletions FIRST" as a numbered hard
   requirement fixed it.

2. The same version returned confidence 0.95 for a rule supported by one
   observation, despite being asked for <= 0.6. Wording the cap as "This is a
   hard cap, not a suggestion" fixed it in the prompt -- and memory.Rule
   enforces it in code regardless, because a prompt is not an invariant.

Runs on the cheap model: measured ~$0.002 per distillation.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from . import config, schemas
from .llm import LLM, Usage, shared
from .memory import Rule, add_rule, record_feedback

MAX_RULES_PER_SESSION = 3


@dataclass
class FeedbackAction:
    """One reviewer action on one finding."""

    action: str  # keep | delete | edit
    finding: dict
    replacement: dict | None = None
    note: str | None = None


@dataclass
class DistillResult:
    rules: list[dict] = field(default_factory=list)
    feedback_id: int | None = None
    usage: Usage = field(default_factory=Usage)
    t_distill: float = 0.0
    actions: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "rules": self.rules,
            "feedback_id": self.feedback_id,
            "usage": self.usage.as_dict(),
            "t_distill": round(self.t_distill, 3),
            "actions": self.actions,
        }


PROMPT = """Distill DURABLE team review conventions from one session of reviewer feedback.

The reviewer read an automated code review and acted on each finding. Your job is
to infer the standing conventions those actions imply, so future reviews follow
them without being told again.

Finding categories are a CLOSED vocabulary. Never use a category outside it:
{categories}
Severities: {severities}

Task context:
  language: {language}
  file/area: {area}
  change kind: {change_kind}

Reviewer feedback:
{feedback}

Hard requirements:

1. A DELETED finding is the strongest available signal: the reviewer does not
   want that class of thing reported at all. Emit rules for deletions FIRST,
   before anything else. Do not skip them.
2. An EDITED finding means the class IS wanted but must be reported differently
   -- most often at a different severity, or with different content. Capture the
   difference between before and after, not the topic alone.
3. A KEPT finding confirms existing behaviour. Only emit a rule for a kept
   finding if it reveals something not already obvious from the fact that you
   reported it.
4. Set `evidence_count` to the number of distinct findings supporting the rule.
   When `evidence_count` is 2 or fewer, `confidence` MUST be at most {cap}.
   This is a hard cap, not a suggestion.
5. Set `is_meta` to true when the rule is about HOW to report -- what to omit,
   what to always include, what severity policy to apply -- rather than about a
   code topic. Meta rules are applied to every future review regardless of
   language, so only mark a rule meta if it genuinely generalizes that far.
6. Choose `scope` honestly:
   - global: applies to every review in any language.
   - language: applies to all reviews in {language}.
   - module: applies only to this file or area.
   Prefer the widest scope you can actually justify from the feedback. A
   reviewer deleting generic nits is expressing a global preference; a reviewer
   demanding a specific log line is usually module-scoped.
7. Give a machine-checkable `assertion` whenever the rule permits one:
   - forbid_category: never report this category. Set `category`.
   - forbid_category_below_severity: this category only above a severity floor.
     Set `category` and `severity`.
   - require_severity_for_category: this category must be at least this
     severity. Set `category` and `severity`.
   - require_field: every finding must populate this field. Set `field`.
   - none: the rule is genuinely subjective. Leave category/severity/field "".
   Unused sub-fields must be the empty string, never omitted.
8. Emit at most {max_rules} rules. No duplicates, and no rule that merely
   restates another with different words.

Write each `rule_text` as a short imperative sentence a new team member could
follow without seeing this feedback."""


def _render_feedback(actions: list[FeedbackAction]) -> str:
    """Render actions grouped by kind, deletions first, to match requirement 1."""
    buckets: dict[str, list[FeedbackAction]] = {"delete": [], "edit": [], "keep": []}
    for a in actions:
        buckets.setdefault(a.action, []).append(a)

    def fmt(f: dict) -> str:
        return (
            f"category={f.get('category')} severity={f.get('severity')} "
            f"message={(f.get('message') or '')[:160]!r}"
        )

    lines: list[str] = []
    if buckets["delete"]:
        lines.append("DELETED (reviewer removed these -- strongest signal):")
        for a in buckets["delete"]:
            lines.append(f"  - {fmt(a.finding)}")
            if a.note:
                lines.append(f"    reviewer note: {a.note!r}")
    if buckets["edit"]:
        lines.append("EDITED (wanted, but reported wrongly):")
        for a in buckets["edit"]:
            lines.append(f"  - before: {fmt(a.finding)}")
            lines.append(f"    after:  {fmt(a.replacement or {})}")
            if a.note:
                lines.append(f"    reviewer note: {a.note!r}")
    if buckets["keep"]:
        lines.append("KEPT (confirmed correct):")
        for a in buckets["keep"]:
            lines.append(f"  - {fmt(a.finding)}")
    return "\n".join(lines) if lines else "(no actions)"


def _coerce_rule(raw: dict, *, lang: str | None, area: str | None) -> Rule | None:
    """Validate one distilled rule into a Rule, or drop it.

    Anything the model can get wrong is fixed up here rather than trusted:
    scope/assertion enums, the confidence cap, and the lang/area a scope implies.
    """
    text = (raw.get("rule_text") or "").strip()
    if not text:
        return None

    scope = raw.get("scope") if raw.get("scope") in schemas.SCOPES else "language"
    assertion = raw.get("assertion") or {}
    kind = assertion.get("kind")
    if kind not in schemas.ASSERT_KINDS:
        kind = "none"

    cat = (assertion.get("category") or "").strip()
    cat = schemas.normalize_category(cat) if cat else None
    sev = (assertion.get("severity") or "").strip()
    sev = schemas.normalize_severity(sev) if sev else None
    fld = (assertion.get("field") or "").strip() or None
    if fld not in (None, *schemas.REQUIRE_FIELDS):
        fld = None

    # An assertion missing the operand it needs cannot be checked, so it is
    # downgraded to `none` rather than stored as a broken check.
    if kind in ("forbid_category", "require_severity_for_category") and not cat:
        kind = "none"
    if kind == "forbid_category_below_severity" and not (cat and sev):
        kind = "none"
    if kind == "require_severity_for_category" and not sev:
        kind = "none"
    if kind == "require_field" and not fld:
        kind = "none"
    if kind == "none":
        cat = sev = fld = None

    try:
        evidence = max(1, int(raw.get("evidence_count", 1)))
    except (TypeError, ValueError):
        evidence = 1
    try:
        conf = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5

    # A rule's scope dictates which tags it must carry: a language rule without
    # a language, or a module rule without an area, would never be retrievable.
    rule_lang = lang if scope in ("language", "module") else None
    rule_area = area if scope == "module" else None
    if scope == "language" and not rule_lang:
        scope = "global"
    if scope == "module" and not rule_area:
        scope = "language" if rule_lang else "global"

    tags = [str(t) for t in (raw.get("tags") or []) if str(t).strip()][:6]

    return Rule(
        rule_text=text,
        scope=scope,
        lang=rule_lang,
        area=rule_area,
        is_meta=bool(raw.get("is_meta")),
        tags=tags,
        assert_kind=kind,
        assert_cat=cat,
        assert_sev=sev,
        assert_field=fld,
        evidence_count=evidence,
        confidence=conf,  # Rule.__post_init__ applies the hard cap
    )


def distill(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    actions: list[FeedbackAction],
    language: str | None = None,
    area: str | None = None,
    change_kind: str | None = None,
    llm: LLM | None = None,
    usage: Usage | None = None,
    persist: bool = True,
) -> DistillResult:
    """Turn one session's feedback into stored rules."""
    llm = llm or shared()
    usage = usage if usage is not None else Usage()
    result = DistillResult(usage=usage)
    result.actions = {
        k: sum(1 for a in actions if a.action == k) for k in ("keep", "delete", "edit")
    }

    if not actions:
        return result

    t0 = time.perf_counter()

    raw_payload = {
        "task_id": task_id,
        "language": language,
        "area": area,
        "change_kind": change_kind,
        "actions": [
            {
                "action": a.action,
                "finding": a.finding,
                "replacement": a.replacement,
                "note": a.note,
            }
            for a in actions
        ],
    }
    if persist:
        result.feedback_id = record_feedback(conn, task_id, raw_payload)

    prompt = PROMPT.format(
        categories=", ".join(schemas.CATEGORIES),
        severities=", ".join(schemas.SEVERITIES),
        language=language or "unknown",
        area=area or "unknown",
        change_kind=change_kind or "unknown",
        feedback=_render_feedback(actions),
        cap=config.SINGLE_EVIDENCE_CAP,
        max_rules=MAX_RULES_PER_SESSION,
    )

    payload = llm.structured(
        model=config.MODEL_CHEAP,
        tool=schemas.EMIT_RULES_TOOL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1600,
        usage=usage,
    )

    seen_texts: set[str] = set()
    for raw in (payload.get("rules") or [])[:MAX_RULES_PER_SESSION]:
        rule = _coerce_rule(raw, lang=language, area=area)
        if rule is None:
            continue
        key = rule.rule_text.strip().lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)

        entry: dict[str, Any] = {
            "rule_text": rule.rule_text,
            "scope": rule.scope,
            "lang": rule.lang,
            "area": rule.area,
            "is_meta": rule.is_meta,
            "assert_kind": rule.assert_kind,
            "assert_cat": rule.assert_cat,
            "assert_sev": rule.assert_sev,
            "assert_field": rule.assert_field,
            "evidence_count": rule.evidence_count,
            "confidence": rule.confidence,
        }
        if persist:
            rule_id, action = add_rule(conn, rule, feedback_id=result.feedback_id)
            entry["id"] = rule_id
            entry["action"] = action
            # The embedding call is part of the memory write cost.
            usage.add_embedding(0)
        result.rules.append(entry)

    result.t_distill = time.perf_counter() - t0
    return result


def actions_from_payload(
    original: list[dict], payload: list[dict]
) -> list[FeedbackAction]:
    """Build actions from a UI/CLI payload.

    Each payload entry is {"index": int, "action": str, "replacement"?, "note"?}.
    Findings not mentioned are treated as kept, which matches how a reviewer
    behaves: they touch what they disagree with and leave the rest.
    """
    by_index = {int(p["index"]): p for p in payload if "index" in p}
    actions: list[FeedbackAction] = []
    for i, finding in enumerate(original):
        p = by_index.get(i)
        if p is None:
            actions.append(FeedbackAction("keep", finding))
            continue
        act = p.get("action", "keep")
        if act not in ("keep", "delete", "edit"):
            act = "keep"
        actions.append(
            FeedbackAction(
                action=act,
                finding=finding,
                replacement=p.get("replacement"),
                note=p.get("note"),
            )
        )
    return actions
