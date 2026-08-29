"""Task sequences and the seeded rule set used by the A/B harness.

Why the headline A/B uses SEEDED rules rather than organically distilled ones:

The memory-OFF baseline is not deterministic. Measured across runs of the same
task it emitted two style findings once and none another time. So an organic
run (review -> feedback -> distill -> review) sometimes produces no deletion
signal at all, and a rule's apparent compliance can swing on what the baseline
happened to emit rather than on whether memory worked.

Seeding a fixed rule set fixes the independent variable. Both arms face the
identical rules, so the difference in compliance is attributable to injection.
The organic loop is still exercised end to end -- see `organic_sequence()` and
the CLI demo -- but it is a demonstration, not the measurement.
"""

from __future__ import annotations

from app.memory import Rule

# Rules planted before the A/B. Every one carries a machine-checkable assertion
# so no LLM judge is needed for the headline number.
#
# evidence_count=3 is deliberate: these stand in for conventions the team has
# expressed more than once, so they are allowed above the single-evidence
# confidence cap. A rule at the cap (<=0.6) would be marked "(tentative)" in the
# prompt, which is not what we want to measure here.
SEED_RULES: list[Rule] = [
    Rule(
        rule_text=(
            "Never report style issues such as naming conventions, missing "
            "docstrings, or absent type annotations. The team does not review these."
        ),
        scope="global",
        is_meta=True,  # about HOW to report -> always injected, never ranked
        tags=["reporting", "style"],
        assert_kind="forbid_category",
        assert_cat="style",
        evidence_count=3,
        confidence=0.9,
    ),
    Rule(
        rule_text=(
            "Every finding must quote the specific code or symbol it refers to in "
            "its evidence field."
        ),
        scope="global",
        is_meta=True,
        tags=["reporting", "evidence"],
        assert_kind="require_field",
        assert_field="evidence",
        evidence_count=3,
        confidence=0.9,
    ),
    Rule(
        rule_text=(
            "In Python services, report missing input validation at high severity: "
            "unvalidated values reaching an outbound request are treated as serious."
        ),
        scope="language",
        lang="python",
        tags=["validation", "python"],
        assert_kind="require_severity_for_category",
        assert_cat="validation",
        assert_sev="high",
        evidence_count=3,
        confidence=0.85,
    ),
    # Cross-language control. This must NOT be injected into any Python task.
    # Its assertion is checkable so a leak would show up as a real FAIL rather
    # than merely a stray line in the prompt.
    Rule(
        rule_text=(
            "In Go code, report every unchecked error return at high severity."
        ),
        scope="language",
        lang="go",
        tags=["error-handling", "go"],
        assert_kind="require_severity_for_category",
        assert_cat="error-handling",
        assert_sev="high",
        evidence_count=3,
        confidence=0.85,
    ),
]

# The A/B task sequence. `expect_leak_free` marks the cross-language control:
# python-scoped rules must not appear when reviewing the Go file.
AB_TASKS: list[str] = ["billing", "auth", "worker"]

# Rules keyed by the language they are scoped to, for leak assertions.
PYTHON_ONLY_TAGS = {"python"}
GO_ONLY_TAGS = {"go"}


def organic_sequence() -> list[dict]:
    """The full learn-then-apply loop, for demonstration.

    Step 1 reviews pay.py and feeds back reviewer actions; steps 2-3 check that
    what was learned carries to a similar file and does not leak to Go.
    """
    return [
        {
            "task_id": "pay",
            "give_feedback": True,
            "note": "review, then delete style nits and raise observability severity",
        },
        {
            "task_id": "billing",
            "give_feedback": False,
            "note": "similar python task -- learned rules should be retrieved",
        },
        {
            "task_id": "worker",
            "give_feedback": False,
            "note": "cross-language control -- python rules must not fire",
        },
    ]


def feedback_payload_for(findings: list[dict]) -> list[dict]:
    """Simulate a reviewer's actions on a set of findings.

    Deletes style nits, raises observability to high, keeps everything else.
    Mirrors what a real reviewer does in the web UI.
    """
    payload: list[dict] = []
    for i, f in enumerate(findings):
        cat = f.get("category")
        if cat == "style":
            payload.append(
                {
                    "index": i,
                    "action": "delete",
                    "note": "we do not review naming or docstrings",
                }
            )
        elif cat == "observability":
            payload.append(
                {
                    "index": i,
                    "action": "edit",
                    "replacement": {
                        **f,
                        "severity": "high",
                        "message": (
                            "Payment operations must log user id and amount at the "
                            "service boundary"
                        ),
                    },
                    "note": "logging gaps on payment paths are high severity",
                }
            )
    return payload
