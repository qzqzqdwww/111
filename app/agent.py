"""Two-phase code review agent.

Why two phases: forced `tool_choice` is the only reliable way to get
schema-shaped JSON out of this proxy (`output_config.format` came back
markdown-fenced), but pinning tool_choice to `emit_findings` also strips the
model's ability to call any *other* tool. So context gathering and structured
emission cannot happen in the same call.

  phase 1  free tool loop     tools=[get_diff, read_file, grep_repo, run_linter]
                              loop tool_use -> tool_result until end_turn
  phase 2  forced emission    tools=[emit_findings], tool_choice=that tool

System prompt layout matters for cost. It is split into two blocks:

  block 0  stable reviewer instructions   <- cache_control: ephemeral
  block 1  retrieved memory rules         <- volatile, AFTER the breakpoint

Measured: with rules changing every call, cache_read stays ~2970 tokens and
only ~1280 are reprocessed. Folding memory into block 0 would invalidate the
cache on every single turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import config, schemas, tools
from .llm import LLM, Usage, shared

# This block is the cached prefix, and its LENGTH is load-bearing.
#
# Measured: with a ~471-token system prompt the whole prefix (system + tools)
# came to ~1020 input tokens and NOTHING cached -- cache_creation_input_tokens
# stayed 0 across every call, with no error. sonnet-4-6 requires a ~1024-token
# minimum cacheable prefix and silently declines below it. The text below is
# sized to clear that threshold comfortably, and every paragraph is real review
# guidance rather than padding, so the tokens buy output quality as well as a
# cache entry. Do not trim this without re-checking cache_read in metrics.
REVIEWER_SYSTEM = """You are a senior code reviewer performing pre-merge review.

Your job is to find the defects that matter in a change and report them
precisely enough that the author can act on each one without asking you a
follow-up question. You are reviewing code that is about to ship, so a missed
defect is expensive and a fabricated one wastes the author's time.

## How to investigate

Start by reading the diff. Never review a diff in isolation: read the
surrounding file before judging any hunk. A line that looks wrong on its own is
often correct in context, and a line that looks harmless on its own is often
wrong given how its callers use it. When the diff touches a function, read the
whole function and the code immediately around it.

Use grep to find call sites and definitions when the change references a symbol
you cannot see in the file you have read. Do not guess what a function does from
its name. If a change alters a function's contract -- its return type, the
exceptions it can raise, the conditions under which it returns nothing -- find
the callers and check whether they still hold up.

Run the linter on the files you review. Issues the linter already reports are
already visible to the author through the existing toolchain, so repeating them
adds noise without adding information. Prefer to spend your attention on the
defects static analysis cannot see: wrong logic, missing failure handling, unsafe
assumptions about external systems, and gaps between what the code does and what
its callers expect.

Anchor every finding to a file and a line number you actually observed in the
file contents. Do not approximate a line number, and do not cite a line you
inferred from the diff header alone.

## What counts as a finding

Report a problem when it could plausibly cause incorrect behavior, data loss, a
security exposure, an unhandled failure, or a production incident. Concretely,
that includes:

- Logic that produces the wrong result for a realistic input, including boundary
  values, empty collections, zero, and negative numbers.
- Unhandled failure modes when talking to anything external: network errors,
  timeouts, non-2xx responses, malformed or truncated payloads, and missing keys
  in a response body.
- Missing validation where untrusted input crosses a trust boundary, especially
  for values that are used in a financial calculation, a query, a path, or an
  outbound request.
- Resource problems: unbounded retries, unbounded reads, connections or file
  handles that are not released on the failure path.
- Concurrency problems: shared state mutated without synchronization, a
  read-modify-write that is not atomic, and retries that are not idempotent.
- Security problems: weak or inappropriate cryptographic primitives, secrets
  reachable from a log or an error message, and authorization that is checked in
  one path but not another.
- Absent observability at a point where a failure would otherwise be silent, so
  that an operator would have no way to tell the operation failed.

Do not report speculation. If you cannot point at the specific code that makes a
problem possible, you do not have a finding yet -- go read more of the file.

## How to judge severity

- high: can cause incorrect results, data loss, a security exposure, or an
  outage. Also use high when the code silently swallows a failure that operators
  or callers genuinely need to know about.
- medium: a real defect with bounded impact, or a gap that will cause trouble
  under load, in an edge case, or during an incident.
- low: worth mentioning but not blocking the merge.

Judge severity by consequence, not by how easy the fix is. A one-line fix for a
silent data-corruption bug is still high.

## How to report

One finding per distinct problem. Do not split a single problem across several
entries, and do not bundle two unrelated problems into one entry. If the same
root cause produces two separate failure modes, report the root cause once.

State what is wrong and why it matters, in one or two sentences. Skip preamble
and do not restate the code back to the author. Assume the reader is a competent
engineer who has the file open.

Pick the single most accurate category from the allowed set, and never invent a
category outside it. When a problem could fit two categories, choose the one
that describes the consequence rather than the mechanism.

Quote the specific code or symbol as evidence, so the author can locate the
problem without searching. Evidence should be the shortest fragment that
identifies the site unambiguously.

Be accurate over exhaustive. A short list of real defects is worth more than a
long list padded with maybes, and a list that includes one fabricated finding
makes the author distrust the rest of it."""

MEMORY_HEADER = """Your team has established the following review conventions,
learned from how this team's reviewers have edited your previous output. These
are binding: apply every one of them to the findings you emit.
"""

EMIT_INSTRUCTION = (
    "Now emit your final findings by calling emit_findings. Apply every team "
    "convention listed in your instructions. Use only the allowed categories."
)


@dataclass
class ReviewResult:
    task_id: str
    findings: list[dict]
    injected_rules: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    t_retrieval: float = 0.0
    t_generation: float = 0.0
    t_total: float = 0.0
    tool_rounds: int = 0
    tool_calls: list[str] = field(default_factory=list)
    hit_round_cap: bool = False

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "findings": self.findings,
            "injected_rules": [
                {"id": r.get("id"), "rule_text": r.get("rule_text")}
                for r in self.injected_rules
            ],
            "usage": self.usage.as_dict(),
            "timing": {
                "retrieval": round(self.t_retrieval, 3),
                "generation": round(self.t_generation, 3),
                "total": round(self.t_total, 3),
            },
            "tool_rounds": self.tool_rounds,
            "tool_calls": self.tool_calls,
            "hit_round_cap": self.hit_round_cap,
        }


def format_memory_block(rules: list[dict]) -> str:
    """Render retrieved rules compactly. One line each keeps the volatile
    portion of the prompt small, since it is reprocessed every turn."""
    if not rules:
        return ""
    lines = [MEMORY_HEADER]
    for r in rules:
        scope = r.get("scope", "global")
        conf = float(r.get("confidence", 0.5))
        tag = f"{scope}"
        if r.get("lang"):
            tag += f"/{r['lang']}"
        marker = " (tentative)" if conf < config.LOW_CONFIDENCE else ""
        lines.append(f"- [{tag}]{marker} {r['rule_text']}")
    return "\n".join(lines)


def build_system(rules: list[dict]) -> list[dict]:
    """Stable cached block first, volatile memory block after the breakpoint."""
    blocks: list[dict] = [
        {
            "type": "text",
            "text": REVIEWER_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    memory = format_memory_block(rules)
    if memory:
        blocks.append({"type": "text", "text": memory})
    return blocks


def _normalize_findings(raw: list[dict], default_file: str) -> list[dict]:
    """Coerce model output onto the closed vocabularies.

    The schema enum makes drift rare, but normalizing here means every
    downstream checker can trust the category/severity values.
    """
    out: list[dict] = []
    for f in raw or []:
        if not isinstance(f, dict):
            continue
        line = f.get("line")
        try:
            line = int(line)
        except (TypeError, ValueError):
            line = 0
        out.append(
            {
                "file": (f.get("file") or default_file).strip(),
                "line": line,
                "category": schemas.normalize_category(f.get("category", "")),
                "severity": schemas.normalize_severity(f.get("severity", "")),
                "message": (f.get("message") or "").strip(),
                "evidence": (f.get("evidence") or "").strip(),
            }
        )
    return out


def review(
    task_id: str,
    *,
    path: str,
    rules: list[dict] | None = None,
    llm: LLM | None = None,
    usage: Usage | None = None,
    max_rounds: int | None = None,
) -> ReviewResult:
    """Run one review. `rules` are pre-retrieved memories to inject (empty = OFF).

    Retrieval is the caller's job (see pipeline.review_task) so this function
    stays usable as the memory-OFF baseline with no special casing.
    """
    llm = llm or shared()
    rules = rules or []
    usage = usage if usage is not None else Usage()
    rounds_cap = max_rounds if max_rounds is not None else config.MAX_TOOL_ROUNDS

    result = ReviewResult(task_id=task_id, findings=[], injected_rules=rules, usage=usage)
    t0 = time.perf_counter()
    system = build_system(rules)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Review the change to `{path}` (task id: {task_id}).\n"
                f"Start by calling get_diff with task_id={task_id!r}, then read the "
                f"file for context before judging."
            ),
        }
    ]

    # --- phase 1: free tool loop -----------------------------------------
    resp = None
    settled = False
    try:
        for _ in range(rounds_cap):
            resp = llm.call(
                model=config.MODEL_MAIN,
                usage=usage,
                max_tokens=2000,
                system=system,
                tools=tools.TOOL_SPECS,
                messages=messages,
            )
            result.tool_rounds += 1
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                settled = False  # nothing pending; phase 2 appends this turn itself
                break
            messages.append({"role": "assistant", "content": resp.content})
            results_block = []
            for b in uses:
                result.tool_calls.append(b.name)
                text, is_error = tools.dispatch(b.name, dict(b.input))
                results_block.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": text,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results_block})
            settled = True
        else:
            # Ran out of rounds while the model still wanted tools. Its last calls
            # were answered above, so the transcript is well-formed -- but the model
            # never signalled it was done gathering context.
            result.hit_round_cap = True
    except Exception:
        # If the model call fails (network error, etc.), fall through to phase 2
        # with whatever context we have. The structured call will surface the
        # error cleanly rather than crashing the whole review.
        pass

    # --- phase 2: forced structured emission ------------------------------
    if resp is not None and not settled and resp.content:
        messages.append({"role": "assistant", "content": resp.content})
    messages.append({"role": "user", "content": EMIT_INSTRUCTION})

    payload = llm.structured(
        model=config.MODEL_MAIN,
        tool=schemas.EMIT_FINDINGS_TOOL,
        system=system,
        messages=messages,
        max_tokens=4000,
        usage=usage,
    )
    result.findings = _normalize_findings(payload.get("findings", []), default_file=path)

    result.t_generation = time.perf_counter() - t0
    result.t_total = result.t_generation
    return result
