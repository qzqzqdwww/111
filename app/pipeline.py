"""Transport-agnostic orchestration shared by the CLI and the web API.

Two entry points:

    review_task(...)    retrieve memories -> review -> persist metrics
    apply_feedback(...) distill reviewer actions -> store rules -> persist metrics

Neither knows anything about rich, FastAPI, or HTTP. Both take an open sqlite
connection so callers control transaction scope and tests can use ':memory:'.

Cost is reported in two buckets, because "is the memory itself expensive?" is a
separate question from "what did this turn cost":

    memory_cost_usd   embeddings for retrieval + the distillation call
    total_cost_usd    everything, including the main review generation
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from . import agent, catalog, config, distill, memory, metrics
from .llm import LLM, Usage, shared


def review_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    memory_on: bool = True,
    llm: LLM | None = None,
    top_k: int | None = None,
    persist_metrics: bool = True,
) -> dict[str, Any]:
    """Run one review, optionally with retrieved memory injected."""
    task = catalog.get(task_id)
    llm = llm or shared()

    retrieval_usage = Usage()
    rules: list[dict] = []
    t_retrieval = 0.0

    if memory_on:
        t0 = time.perf_counter()
        rules, embed_tok = memory.retrieve(
            conn,
            query_text=task.query_text,
            lang=task.lang,
            area=task.area,
            top_k=top_k,
        )
        retrieval_usage.add_embedding(embed_tok)
        t_retrieval = time.perf_counter() - t0

    gen_usage = Usage()
    result = agent.review(
        task.task_id, path=task.path, rules=rules, llm=llm, usage=gen_usage
    )

    combined = Usage()
    combined.merge(retrieval_usage)
    combined.merge(gen_usage)

    payload = result.as_dict()
    payload["memory_on"] = memory_on
    payload["task"] = {
        "task_id": task.task_id,
        "path": task.path,
        "lang": task.lang,
        "area": task.area,
        "change_kind": task.change_kind,
        "label": task.label,
    }
    payload["injected_rules"] = rules
    payload["usage"] = combined.as_dict()
    payload["timing"] = {
        "retrieval": round(t_retrieval, 3),
        "generation": round(result.t_generation, 3),
        # What the user waits for. Distillation happens after the response is
        # already in hand, so it is deliberately excluded here and reported
        # separately -- otherwise the latency figure would overstate the cost of
        # having memory at all.
        "user_visible": round(t_retrieval + result.t_generation, 3),
    }
    payload["cost"] = {
        "memory_usd": round(retrieval_usage.cost_usd, 8),
        "generation_usd": round(gen_usage.cost_usd, 6),
        "total_usd": round(combined.cost_usd, 6),
    }

    if persist_metrics:
        metrics.record_turn(
            conn,
            task_id=task.task_id,
            memory_on=memory_on,
            phase="review",
            t_retrieval=t_retrieval,
            t_generation=result.t_generation,
            t_distill=0.0,
            t_user_visible=t_retrieval + result.t_generation,
            usage=combined,
            memory_cost_usd=retrieval_usage.cost_usd,
            n_rules_injected=len(rules),
            n_findings=len(result.findings),
        )

    return payload


def apply_feedback(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    findings: list[dict],
    payload: list[dict],
    llm: LLM | None = None,
    persist_metrics: bool = True,
) -> dict[str, Any]:
    """Distill reviewer actions into rules.

    `payload` entries look like {"index": int, "action": "keep"|"delete"|"edit",
    "replacement": {...}?, "note": str?}. Findings not mentioned count as kept.
    """
    task = catalog.get(task_id)
    llm = llm or shared()

    actions = distill.actions_from_payload(findings, payload)
    res = distill.distill(
        conn,
        task_id=task.task_id,
        actions=actions,
        language=task.lang,
        area=task.area,
        change_kind=task.change_kind,
        llm=llm,
    )

    out = res.as_dict()
    out["task_id"] = task.task_id
    out["cost"] = {"memory_usd": round(res.usage.cost_usd, 6)}

    if persist_metrics:
        metrics.record_turn(
            conn,
            task_id=task.task_id,
            memory_on=True,
            phase="feedback",
            t_retrieval=0.0,
            t_generation=0.0,
            t_distill=res.t_distill,
            # Distillation is off the user's critical path.
            t_user_visible=0.0,
            usage=res.usage,
            memory_cost_usd=res.usage.cost_usd,
            n_rules_injected=0,
            n_findings=len(findings),
        )

    return out


def preflight(llm: LLM | None = None) -> dict[str, Any]:
    """Check both models before doing real work, so failures are legible.

    The proxy returns 503 model_not_found for unavailable ids, which is easy to
    mistake for a transient outage. Surfacing it up front avoids that confusion.
    """
    llm = llm or shared()
    out: dict[str, Any] = {"ok": True, "models": {}}
    for label, model in (("main", config.MODEL_MAIN), ("cheap", config.MODEL_CHEAP)):
        ok, msg = llm.preflight(model)
        out["models"][label] = {"model": model, "ok": ok, "detail": msg}
        if not ok:
            out["ok"] = False
    return out
