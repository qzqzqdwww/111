"""Preset tools the review agent can call during phase 1.

All four operate on a real fixture repo under `fixtures/`, not on stubbed
strings -- the agent genuinely reads files and greps for symbols. `run_linter`
is a deterministic rule-based checker rather than a real linter subprocess
(no linters are installed here), but it returns realistic output and exists for
a design reason: it makes "don't report what the linter already catches" a
learnable convention.

Path handling: every path from the model is untrusted. `_resolve` confines all
reads to the fixture repo root and rejects traversal.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from . import config

DIFF_DIR = config.ROOT / "fixtures" / "diffs"


class ToolError(RuntimeError):
    """Returned to the model as an is_error tool_result, not raised to the user."""


def _resolve(path: str) -> Path:
    """Resolve a model-supplied path, confined to the fixture repo."""
    root = config.FIXTURE_REPO.resolve()
    candidate = (root / path.lstrip("/\\")).resolve()
    if candidate != root and root not in candidate.parents:
        raise ToolError(f"path escapes the repo root: {path!r}")
    return candidate


# --- tool implementations -------------------------------------------------


def get_diff(task_id: str) -> str:
    """Return the diff under review for a task."""
    f = DIFF_DIR / f"{task_id}.diff"
    if not f.exists():
        available = sorted(p.stem for p in DIFF_DIR.glob("*.diff"))
        raise ToolError(f"no diff for task {task_id!r}. Available: {available}")
    return f.read_text(encoding="utf-8")


def read_file(path: str, start: int | None = None, end: int | None = None) -> str:
    """Read a repo file, optionally a line range. Output is line-numbered so the
    model can cite accurate line numbers (rules often require file:line)."""
    f = _resolve(path)
    if not f.exists():
        raise ToolError(f"no such file: {path!r}")
    lines = f.read_text(encoding="utf-8").splitlines()
    lo = max(1, start or 1)
    hi = min(len(lines), end or len(lines))
    return "\n".join(f"{i:>4} | {lines[i - 1]}" for i in range(lo, hi + 1))


def grep_repo(pattern: str, glob: str = "**/*") -> str:
    """Search the repo for a regex. Returns path:line:text hits."""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"bad regex {pattern!r}: {e}") from e
    root = config.FIXTURE_REPO
    hits: list[str] = []
    for f in sorted(root.glob(glob)):
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                rel = f.relative_to(root).as_posix()
                hits.append(f"{rel}:{i}: {line.strip()}")
                if len(hits) >= 50:
                    return "\n".join(hits) + "\n... (truncated at 50 hits)"
    return "\n".join(hits) if hits else "(no matches)"


# Deterministic checks standing in for a linter. Each is (regex, code, message).
_LINT_RULES: list[tuple[str, str, str]] = [
    (r"\bmd5\b", "S324", "use of insecure hash function md5"),
    (r"requests\.(get|post|put|delete)\((?![^)]*timeout)", "B113", "request without timeout"),
    (r"^\s*except\s*:", "E722", "bare except"),
    (r"\bprint\(", "T201", "print found"),
    (r"[ \t]+$", "W291", "trailing whitespace"),
]


def run_linter(path: str) -> str:
    """Run the project's static checks on a file.

    Returns findings the existing toolchain already catches -- the agent can
    then avoid duplicating them.
    """
    f = _resolve(path)
    if not f.exists():
        raise ToolError(f"no such file: {path!r}")
    out: list[str] = []
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        for pattern, code, msg in _LINT_RULES:
            if re.search(pattern, line):
                out.append(f"{path}:{i}: {code} {msg}")
    if not out:
        return "(linter: no issues)"
    return "\n".join(out)


# --- registry -------------------------------------------------------------

TOOL_SPECS: list[dict] = [
    {
        "name": "get_diff",
        "description": "Get the diff under review for the current task.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a repo file with line numbers. Use this to see context around "
            "the diff before judging it. Cite the line numbers you see here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path."},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "grep_repo",
        "description": "Search the repo with a regex to find definitions or call sites.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "glob": {"type": "string", "description": "e.g. '**/*.py'"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_linter",
        "description": (
            "Run the project's existing static checks on a file. Issues reported "
            "here are already caught by the toolchain."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]

_IMPLS: dict[str, Callable[..., str]] = {
    "get_diff": get_diff,
    "read_file": read_file,
    "grep_repo": grep_repo,
    "run_linter": run_linter,
}


def dispatch(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Execute a tool call. Returns (result_text, is_error).

    Errors come back as text with is_error=True so the model can correct itself
    rather than the whole turn failing.
    """
    impl = _IMPLS.get(name)
    if impl is None:
        return f"unknown tool: {name!r}", True
    try:
        return impl(**tool_input), False
    except ToolError as e:
        return str(e), True
    except TypeError as e:
        return f"bad arguments for {name}: {e}", True
