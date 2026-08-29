"""Per-turn instrumentation for the three grading criteria.

Everything here answers one of:

    memory cost      tokens and USD spent specifically on memory (embeddings for
                     retrieval, plus the distillation call) vs total spend
    latency          phase-level wall clock, with the distinct notion of
                     user_visible time -- distillation runs after the response is
                     already returned, so charging it to perceived latency would
                     misrepresent what memory costs the user
    effectiveness    rules injected per turn, plus compliance recorded by
                     eval/checkers.py against each rule's assertion

Token counts come exclusively from `response.usage`. The proxy has no
`count_tokens` endpoint (404), so there is no estimate anywhere in this codebase.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from .llm import Usage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_turn(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    memory_on: bool,
    phase: str,
    t_retrieval: float,
    t_generation: float,
    t_distill: float,
    t_user_visible: float,
    usage: Usage,
    memory_cost_usd: float,
    n_rules_injected: int,
    n_findings: int,
) -> int:
    cur = conn.execute(
        """INSERT INTO turns
             (task_id, memory_on, phase, t_retrieval, t_generation, t_distill,
              t_total, t_user_visible, in_tok, out_tok, cache_write, cache_read,
              embed_tok, memory_cost_usd, total_cost_usd, n_rules_injected,
              n_findings, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            task_id,
            int(memory_on),
            phase,
            round(t_retrieval, 4),
            round(t_generation, 4),
            round(t_distill, 4),
            round(t_retrieval + t_generation + t_distill, 4),
            round(t_user_visible, 4),
            usage.in_tok,
            usage.out_tok,
            usage.cache_write,
            usage.cache_read,
            usage.embed_tok,
            round(memory_cost_usd, 8),
            round(usage.cost_usd, 8),
            n_rules_injected,
            n_findings,
            _now(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """Aggregate review turns by memory ON/OFF, plus feedback overhead."""
    out: dict[str, Any] = {"review": {}, "feedback": {}, "totals": {}}

    for label, flag in (("memory_on", 1), ("memory_off", 0)):
        row = conn.execute(
            """SELECT
                 COUNT(*)               AS turns,
                 AVG(t_user_visible)    AS avg_user_visible,
                 AVG(t_retrieval)       AS avg_retrieval,
                 AVG(t_generation)      AS avg_generation,
                 AVG(total_cost_usd)    AS avg_cost,
                 SUM(total_cost_usd)    AS sum_cost,
                 SUM(memory_cost_usd)   AS sum_memory_cost,
                 SUM(in_tok)            AS in_tok,
                 SUM(out_tok)           AS out_tok,
                 SUM(cache_write)       AS cache_write,
                 SUM(cache_read)        AS cache_read,
                 SUM(embed_tok)         AS embed_tok,
                 AVG(n_rules_injected)  AS avg_rules,
                 AVG(n_findings)        AS avg_findings
               FROM turns WHERE phase='review' AND memory_on=?""",
            (flag,),
        ).fetchone()
        out["review"][label] = _clean_row(row)

    row = conn.execute(
        """SELECT COUNT(*) AS turns, AVG(t_distill) AS avg_distill,
                  SUM(memory_cost_usd) AS sum_memory_cost,
                  SUM(in_tok) AS in_tok, SUM(out_tok) AS out_tok
           FROM turns WHERE phase='feedback'"""
    ).fetchone()
    out["feedback"] = _clean_row(row)

    row = conn.execute(
        """SELECT COUNT(*) AS turns,
                  SUM(total_cost_usd)  AS total_cost_usd,
                  SUM(memory_cost_usd) AS memory_cost_usd,
                  SUM(cache_read)      AS cache_read,
                  SUM(cache_write)     AS cache_write,
                  SUM(in_tok)          AS in_tok
           FROM turns"""
    ).fetchone()
    totals = _clean_row(row)
    denom = (
        (totals.get("cache_read") or 0)
        + (totals.get("cache_write") or 0)
        + (totals.get("in_tok") or 0)
    )
    totals["cache_hit_ratio"] = (
        round((totals.get("cache_read") or 0) / denom, 4) if denom else 0.0
    )
    tc = totals.get("total_cost_usd") or 0.0
    totals["memory_cost_share"] = (
        round((totals.get("memory_cost_usd") or 0.0) / tc, 4) if tc else 0.0
    )
    out["totals"] = totals
    return out


def _clean_row(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    d: dict[str, Any] = {}
    for k in row.keys():
        v = row[k]
        if v is None:
            d[k] = 0
        elif isinstance(v, float):
            # Memory costs land in the 1e-7 range; rounding to 6 would show 0.
            d[k] = round(v, 8)
        else:
            d[k] = v
    return d


def recent_turns(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM turns ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]
