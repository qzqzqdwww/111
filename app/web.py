"""FastAPI surface. Thin HTTP wrapper over app.pipeline.

    uvicorn app.web:app --reload

Endpoints mirror the CLI exactly -- run a review, act on findings, browse the
memory store, read metrics, run the A/B. No business logic lives here; if you
find yourself adding any, it belongs in app.pipeline so the CLI gets it too.

A single sqlite connection is shared across requests with check_same_thread
disabled, guarded by a lock: this is a demo server, and serializing the handful
of writes is simpler and safer than a pool.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import catalog, config, memory, metrics, pipeline

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Feedback-Memory Review Agent", version="1.0")

_conn: sqlite3.Connection | None = None
_lock = threading.RLock()


_db_initialized = False


def _ensure_db() -> sqlite3.Connection:
    """Create (once) the shared connection used by all handlers."""
    global _conn, _db_initialized
    with _lock:
        if _conn is None:
            _conn = memory.connect()
            _conn.execute("PRAGMA journal_mode=WAL")
            _db_initialized = True
    return _conn


def db() -> sqlite3.Connection:
    """Return the shared connection, creating it on first call.

    A single connection is shared across requests and all writes are guarded by
    ``_lock``.  This is safe only when the server runs with a single uvicorn
    worker (the default in development); multiple workers each get their own
    process and therefore their own connection, so cross-worker writes would
    not be serialized.  For production use with multiple workers, switch to
    per-request connections or a connection pool.
    """
    if _conn is None:
        _ensure_db()
    return _conn


# --- request models -------------------------------------------------------


class ReviewRequest(BaseModel):
    task_id: str = Field(..., description="a key from /api/tasks")
    memory_on: bool = True


class FeedbackItem(BaseModel):
    index: int
    action: str = Field("keep", pattern="^(keep|delete|edit)$")
    replacement: dict[str, Any] | None = None
    note: str | None = None


class FeedbackRequest(BaseModel):
    task_id: str
    findings: list[dict[str, Any]]
    actions: list[FeedbackItem]


class AbRequest(BaseModel):
    tasks: list[str] | None = None
    repeats: int = Field(1, ge=1, le=5)


# --- routes ---------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "base_url": config.BASE_URL,
        "models": {"main": config.MODEL_MAIN, "cheap": config.MODEL_CHEAP},
    }


@app.get("/api/preflight")
def preflight() -> dict[str, Any]:
    return pipeline.preflight()


@app.get("/api/tasks")
def tasks() -> dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "path": t.path,
                "lang": t.lang,
                "change_kind": t.change_kind,
                "label": t.label,
            }
            for t in catalog.TASKS.values()
        ]
    }


@app.post("/api/review")
def review(req: ReviewRequest) -> dict[str, Any]:
    if req.task_id not in catalog.TASKS:
        raise HTTPException(404, f"unknown task {req.task_id!r}")
    try:
        with _lock:
            return pipeline.review_task(db(), req.task_id, memory_on=req.memory_on)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"{type(e).__name__}: {e}") from e


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    if req.task_id not in catalog.TASKS:
        raise HTTPException(404, f"unknown task {req.task_id!r}")
    payload = [a.model_dump() for a in req.actions]
    if not any(a["action"] in ("delete", "edit") for a in payload):
        # Nothing to learn from an all-keep submission; say so rather than
        # burning a distillation call to produce zero rules.
        return {
            "rules": [],
            "actions": {"keep": len(req.findings), "delete": 0, "edit": 0},
            "t_distill": 0.0,
            "cost": {"memory_usd": 0.0},
            "note": "no deletions or edits — nothing to distill",
        }
    try:
        with _lock:
            return pipeline.apply_feedback(
                db(), req.task_id, findings=req.findings, payload=payload
            )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"{type(e).__name__}: {e}") from e


@app.get("/api/rules")
def rules(include_superseded: bool = False, lang: str | None = None) -> dict[str, Any]:
    with _lock:
        conn = db()
        items = memory.list_rules(
            conn, status=None if include_superseded else "active", lang=lang
        )
        return {"rules": items, "stats": memory.stats(conn)}


@app.get("/api/metrics")
def get_metrics(limit: int = 20) -> dict[str, Any]:
    with _lock:
        conn = db()
        return {
            "summary": metrics.summary(conn),
            "recent": metrics.recent_turns(conn, limit=limit),
        }


@app.post("/api/eval/ab")
def run_ab(req: AbRequest) -> dict[str, Any]:
    """Run the A/B in a throwaway store so it never pollutes live memory."""
    from eval.run_ab import run as run_eval

    try:
        res = run_eval(req.tasks, db=":memory:", repeats=req.repeats)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"{type(e).__name__}: {e}") from e
    # Trim the per-task findings: the UI shows aggregates, and the full payload
    # is large.
    for arm in ("off", "on"):
        for row in res[arm]["rows"]:
            row.pop("findings", None)
    return res


@app.delete("/api/rules")
def reset_rules() -> dict[str, Any]:
    """Clear learned memory (keeps recorded metrics) so a demo can start fresh."""
    with _lock:
        conn = db()
        conn.execute("DELETE FROM rule_sources")
        conn.execute("DELETE FROM rules")
        conn.execute("DELETE FROM feedback_events")
        conn.commit()
    return {"ok": True, "cleared": "rules, rule_sources, feedback_events"}


# --- static frontend ------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    f = STATIC_DIR / "index.html"
    if not f.exists():
        raise HTTPException(500, "static/index.html is missing")
    return FileResponse(str(f))
