"""Closed vocabularies and JSON schemas for structured model output.

Why the vocabularies are CLOSED (this is the core of the design):

A memory-OFF baseline run produced findings with free-text categories
("Error Handling", "Code Quality", "Observability", ...). Two low-severity
naming/docstring nits landed under "Code Quality", and a keyword-based checker
looking for "style"/"nit"/"naming" scored that as "no style issues" -- a false
negative in the checker itself.

Free-text categories cannot be validated reliably. Fixing the category set to a
closed enum is what lets a rule like "stop reporting style nits" be verified
with a one-line assertion instead of an LLM judgment call. Every downstream
checker in eval/checkers.py depends on this.
"""

from __future__ import annotations

# --- closed vocabularies ---------------------------------------------------

CATEGORIES: list[str] = [
    "correctness",
    "error-handling",
    "security",
    "validation",
    "reliability",
    "observability",
    "performance",
    "style",
]

SEVERITIES: list[str] = ["low", "medium", "high"]
SEVERITY_RANK: dict[str, int] = {s: i for i, s in enumerate(SEVERITIES)}

LANGUAGES: list[str] = ["python", "typescript", "go"]
CHANGE_KINDS: list[str] = ["bugfix", "refactor", "new-feature", "test"]
SCOPES: list[str] = ["global", "language", "module"]

# Assertion kinds are deliberately limited to forms that can be evaluated
# against the findings schema below. `none` is the escape hatch for genuinely
# subjective rules, which fall back to an LLM judge.
ASSERT_KINDS: list[str] = [
    "forbid_category",
    "forbid_category_below_severity",
    "require_severity_for_category",
    "require_field",
    "none",
]

REQUIRE_FIELDS: list[str] = ["file", "line", "evidence"]


# --- findings (agent output) ----------------------------------------------

FINDING_PROPERTIES = {
    "file": {"type": "string", "description": "Repo-relative path."},
    "line": {"type": "integer", "description": "1-indexed line the finding anchors to."},
    "category": {
        "type": "string",
        "enum": CATEGORIES,
        "description": "Must be one of the allowed categories. Do not invent new ones.",
    },
    "severity": {"type": "string", "enum": SEVERITIES},
    "message": {
        "type": "string",
        "description": "What is wrong and why it matters. One or two sentences.",
    },
    "evidence": {
        "type": "string",
        "description": "The specific code or symbol this refers to.",
    },
}

FINDINGS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": FINDING_PROPERTIES,
                "required": ["file", "line", "category", "severity", "message", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}

EMIT_FINDINGS_TOOL: dict = {
    "name": "emit_findings",
    "description": (
        "Emit the final code review findings. Call this exactly once with the "
        "complete set of findings."
    ),
    "input_schema": FINDINGS_SCHEMA,
}


# --- rules (distiller output) --------------------------------------------

RULES_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule_text": {
                        "type": "string",
                        "description": (
                            "The team convention, as a short imperative sentence. "
                            "Generalized, not tied to one specific line of code."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "enum": SCOPES,
                        "description": (
                            "global = applies to every review; language = all reviews "
                            "in this language; module = only this file/area."
                        ),
                    },
                    "is_meta": {
                        "type": "boolean",
                        "description": (
                            "True if the rule is about HOW to report (what to include, "
                            "what not to report, severity policy) rather than about a "
                            "code topic. Meta rules are always injected."
                        ),
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "assertion": {
                        "type": "object",
                        "description": "Machine-checkable form of the rule.",
                        "properties": {
                            "kind": {"type": "string", "enum": ASSERT_KINDS},
                            "category": {"type": "string", "enum": CATEGORIES + [""]},
                            "severity": {"type": "string", "enum": SEVERITIES + [""]},
                            "field": {"type": "string", "enum": REQUIRE_FIELDS + [""]},
                        },
                        "required": ["kind", "category", "severity", "field"],
                        "additionalProperties": False,
                    },
                    "evidence_count": {
                        "type": "integer",
                        "description": "Number of distinct findings supporting this rule.",
                    },
                    "confidence": {"type": "number"},
                },
                "required": [
                    "rule_text",
                    "scope",
                    "is_meta",
                    "tags",
                    "assertion",
                    "evidence_count",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rules"],
    "additionalProperties": False,
}

EMIT_RULES_TOOL: dict = {
    "name": "emit_rules",
    "description": "Emit the distilled team conventions.",
    "input_schema": RULES_SCHEMA,
}


# --- task classification (cheap model) -----------------------------------

CLASSIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": LANGUAGES},
        "change_kind": {"type": "string", "enum": CHANGE_KINDS},
    },
    "required": ["language", "change_kind"],
    "additionalProperties": False,
}

EMIT_CLASSIFY_TOOL: dict = {
    "name": "emit_classification",
    "description": "Classify the review task.",
    "input_schema": CLASSIFY_SCHEMA,
}


# --- LLM judge (subjective rule fallback) --------------------------------

JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["passed", "reason"],
    "additionalProperties": False,
}

EMIT_JUDGE_TOOL: dict = {
    "name": "emit_verdict",
    "description": "Judge whether the findings honor the rule.",
    "input_schema": JUDGE_SCHEMA,
}


# --- validation helpers ---------------------------------------------------


def normalize_category(raw: str) -> str:
    """Map a model-supplied category onto the closed vocabulary.

    The schema enum makes violations rare, but a proxy could still pass through
    something off-list, so callers get a defined value rather than a KeyError.
    Unknown values map to "correctness" (the least presumptuous bucket) so a
    stray label never silently becomes an un-checkable category.
    """
    c = (raw or "").strip().lower().replace("_", "-").replace(" ", "-")
    if c in CATEGORIES:
        return c
    aliases = {
        "code-quality": "style",
        "quality": "style",
        "naming": "style",
        "formatting": "style",
        "nit": "style",
        "docs": "style",
        "documentation": "style",
        "bug": "correctness",
        "logic": "correctness",
        "error": "error-handling",
        "errors": "error-handling",
        "exception-handling": "error-handling",
        "input-validation": "validation",
        "logging": "observability",
        "monitoring": "observability",
        "perf": "performance",
        "resilience": "reliability",
    }
    return aliases.get(c, "correctness")


def normalize_severity(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in SEVERITY_RANK:
        return s
    return {"critical": "high", "major": "high", "minor": "low", "info": "low"}.get(
        s, "medium"
    )
