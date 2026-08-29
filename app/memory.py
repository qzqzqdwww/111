"""Memory store: SQLite persistence, tag-gated retrieval, dedup and conflict.

Retrieval is deliberately NOT pure vector search. Measured on 5 rules x 4 tasks:

  - Pure vector top-3 leaked across languages 4 times (a Go task retrieved a
    React rule and a Python money-validation rule).
  - The most important rule -- "don't report style nits" -- scored LOWEST and
    flattest of all rules against every task (0.170-0.225), because it
    describes reporting behaviour rather than a code topic. Pure vector search
    never retrieved it at all for the Go task.

So retrieval is two-tier:
  1. `is_meta` rules (about HOW to report) are injected unconditionally.
  2. Everything else is gated by scope/lang/area first, THEN ranked by
     cosine * confidence * recency.

Measured after the fix: 0 cross-language leaks, style rule present in 4/4 tasks.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import config, embed

SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  rule_text      TEXT    NOT NULL,
  scope          TEXT    NOT NULL,
  lang           TEXT,
  area           TEXT,
  is_meta        INTEGER NOT NULL DEFAULT 0,
  tags           TEXT    NOT NULL DEFAULT '[]',
  assert_kind    TEXT    NOT NULL DEFAULT 'none',
  assert_cat     TEXT,
  assert_sev     TEXT,
  assert_field   TEXT,
  evidence_count INTEGER NOT NULL DEFAULT 1,
  confidence     REAL    NOT NULL DEFAULT 0.5,
  hits           INTEGER NOT NULL DEFAULT 0,
  applied_ok     INTEGER NOT NULL DEFAULT 0,
  applied_total  INTEGER NOT NULL DEFAULT 0,
  status         TEXT    NOT NULL DEFAULT 'active',
  superseded_by  INTEGER REFERENCES rules(id),
  needs_review   INTEGER NOT NULL DEFAULT 0,
  embedding      BLOB,
  created_at     TEXT    NOT NULL,
  updated_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(status, is_meta);
CREATE INDEX IF NOT EXISTS idx_rules_lang   ON rules(status, lang);

CREATE TABLE IF NOT EXISTS feedback_events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id    TEXT NOT NULL,
  raw_json   TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rule_sources (
  rule_id     INTEGER NOT NULL REFERENCES rules(id),
  feedback_id INTEGER NOT NULL REFERENCES feedback_events(id),
  PRIMARY KEY (rule_id, feedback_id)
);

CREATE TABLE IF NOT EXISTS turns (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id          TEXT    NOT NULL,
  memory_on        INTEGER NOT NULL,
  phase            TEXT    NOT NULL DEFAULT 'review',
  t_retrieval      REAL    NOT NULL DEFAULT 0,
  t_generation     REAL    NOT NULL DEFAULT 0,
  t_distill        REAL    NOT NULL DEFAULT 0,
  t_total          REAL    NOT NULL DEFAULT 0,
  t_user_visible   REAL    NOT NULL DEFAULT 0,
  in_tok           INTEGER NOT NULL DEFAULT 0,
  out_tok          INTEGER NOT NULL DEFAULT 0,
  cache_write      INTEGER NOT NULL DEFAULT 0,
  cache_read       INTEGER NOT NULL DEFAULT 0,
  embed_tok        INTEGER NOT NULL DEFAULT 0,
  memory_cost_usd  REAL    NOT NULL DEFAULT 0,
  total_cost_usd   REAL    NOT NULL DEFAULT 0,
  n_rules_injected INTEGER NOT NULL DEFAULT 0,
  n_findings       INTEGER NOT NULL DEFAULT 0,
  created_at       TEXT    NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open (and initialize) the store. Pass ':memory:' for tests."""
    target = str(path) if path is not None else str(config.DB_PATH)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# --- rule dataclass -------------------------------------------------------


@dataclass
class Rule:
    rule_text: str
    scope: str = "global"
    lang: str | None = None
    area: str | None = None
    is_meta: bool = False
    tags: list[str] = None  # type: ignore[assignment]
    assert_kind: str = "none"
    assert_cat: str | None = None
    assert_sev: str | None = None
    assert_field: str | None = None
    evidence_count: int = 1
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []
        # Hard cap: a rule seen at most twice can never be high-confidence.
        # The distiller is instructed to respect this, and was measured
        # violating it (returning 0.95 on single-observation evidence), so it is
        # enforced here in code as well. Prompt instructions are not a
        # substitute for an invariant.
        self.confidence = clamp_confidence(self.confidence, self.evidence_count)


def clamp_confidence(confidence: float, evidence_count: int) -> float:
    c = max(0.0, min(1.0, float(confidence)))
    if evidence_count <= 2:
        c = min(c, config.SINGLE_EVIDENCE_CAP)
    return round(c, 3)


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    d["is_meta"] = bool(d.get("is_meta"))
    d.pop("embedding", None)  # callers never need the raw bytes
    return d


# --- writes ---------------------------------------------------------------


def record_feedback(conn: sqlite3.Connection, task_id: str, raw: dict) -> int:
    cur = conn.execute(
        "INSERT INTO feedback_events (task_id, raw_json, created_at) VALUES (?,?,?)",
        (task_id, json.dumps(raw, ensure_ascii=False), _now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def _conflicts(a: sqlite3.Row | dict, b: Rule) -> bool:
    """True when two rules make contradictory demands about one category.

    Forbidding a category while also requiring a severity for it is the case
    that actually shows up: the team first says "stop reporting X", then later
    says "report X, but only when it's high".
    """
    cat_a = (a["assert_cat"] or "").strip()
    cat_b = (b.assert_cat or "").strip()
    if not cat_a or cat_a != cat_b:
        return False
    forbid = {"forbid_category", "forbid_category_below_severity"}
    require = {"require_severity_for_category"}
    ka, kb = a["assert_kind"], b.assert_kind
    return (ka in forbid and kb in require) or (ka in require and kb in forbid)


def add_rule(
    conn: sqlite3.Connection,
    rule: Rule,
    *,
    feedback_id: int | None = None,
    vector: np.ndarray | None = None,
) -> tuple[int, str]:
    """Insert a rule, merging into a near-duplicate when one exists.

    Returns (rule_id, action) where action is 'inserted' | 'merged' | 'superseded'.

    Merging is the ONLY legitimate way confidence rises: repeated independent
    evidence for the same convention. That is what keeps a single off-hand
    comment from producing a high-confidence rule.
    """
    if vector is None:
        vector, _ = embed.embed_one(rule.rule_text)

    # Compare only against rules in the same scope/lang bucket -- a Python rule
    # and a Go rule are never the same rule however similar their wording.
    rows = conn.execute(
        """SELECT * FROM rules
           WHERE status='active' AND scope=? AND IFNULL(lang,'')=IFNULL(?,'')""",
        (rule.scope, rule.lang),
    ).fetchall()

    best_row, best_sim = None, -1.0
    for row in rows:
        if not row["embedding"]:
            continue
        sim = float(embed.from_blob(row["embedding"]) @ vector)
        if sim > best_sim:
            best_row, best_sim = row, sim

    now = _now()

    # 1. Near-duplicate -> merge evidence, recompute confidence.
    if best_row is not None and best_sim >= config.DEDUP_SAME:
        new_count = int(best_row["evidence_count"]) + max(1, rule.evidence_count)
        # Confidence grows with evidence but saturates; never reaches certainty
        # from repetition alone.
        grown = 1.0 - math.pow(0.55, new_count)
        merged = clamp_confidence(
            max(float(best_row["confidence"]), min(grown, 0.95)), new_count
        )
        conn.execute(
            """UPDATE rules SET evidence_count=?, confidence=?, updated_at=?,
                                needs_review=0
               WHERE id=?""",
            (new_count, merged, now, best_row["id"]),
        )
        if feedback_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO rule_sources (rule_id, feedback_id) VALUES (?,?)",
                (best_row["id"], feedback_id),
            )
        conn.commit()
        return int(best_row["id"]), "merged"

    # 2. Direct contradiction -> insert new, supersede old. Never silently
    #    overwrite: history stays queryable and reversible.
    superseded: sqlite3.Row | None = None
    for row in rows:
        if _conflicts(row, rule):
            superseded = row
            break

    needs_review = int(
        best_row is not None and config.DEDUP_REVIEW <= best_sim < config.DEDUP_SAME
    )
    cur = conn.execute(
        """INSERT INTO rules
             (rule_text, scope, lang, area, is_meta, tags, assert_kind, assert_cat,
              assert_sev, assert_field, evidence_count, confidence, needs_review,
              embedding, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            rule.rule_text,
            rule.scope,
            rule.lang,
            rule.area,
            int(rule.is_meta),
            json.dumps(rule.tags, ensure_ascii=False),
            rule.assert_kind,
            rule.assert_cat or None,
            rule.assert_sev or None,
            rule.assert_field or None,
            rule.evidence_count,
            rule.confidence,
            needs_review,
            embed.to_blob(vector),
            now,
            now,
        ),
    )
    new_id = int(cur.lastrowid)

    action = "inserted"
    if superseded is not None:
        conn.execute(
            "UPDATE rules SET status='superseded', superseded_by=?, updated_at=? WHERE id=?",
            (new_id, now, superseded["id"]),
        )
        action = "superseded"

    if feedback_id is not None:
        conn.execute(
            "INSERT OR IGNORE INTO rule_sources (rule_id, feedback_id) VALUES (?,?)",
            (new_id, feedback_id),
        )
    conn.commit()
    return new_id, action


# --- reads ----------------------------------------------------------------


def list_rules(
    conn: sqlite3.Connection,
    *,
    status: str | None = "active",
    lang: str | None = None,
) -> list[dict]:
    sql = "SELECT * FROM rules"
    where, params = [], []
    if status:
        where.append("status=?")
        params.append(status)
    if lang:
        where.append("(lang IS NULL OR lang=?)")
        params.append(lang)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY is_meta DESC, confidence DESC, id ASC"
    return [row_to_dict(r) for r in conn.execute(sql, params).fetchall()]


def _scope_allows(row: sqlite3.Row, lang: str | None, area: str | None) -> bool:
    """Tag gate. This is what prevents cross-language leakage."""
    scope = row["scope"]
    if scope == "global":
        return True
    if scope == "language":
        return bool(lang) and row["lang"] == lang
    if scope == "module":
        if not row["area"]:
            return False
        if row["lang"] and lang and row["lang"] != lang:
            return False
        return bool(area) and (area == row["area"] or area.startswith(row["area"]))
    return False


def _recency_weight(updated_at: str, half_life_days: float = 45.0) -> float:
    """Gentle decay so stale conventions lose out to fresh ones, without ever
    dropping to zero -- an old rule that still applies should still fire."""
    try:
        ts = datetime.fromisoformat(updated_at)
    except (TypeError, ValueError):
        return 1.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0)
    return 0.5 + 0.5 * math.pow(0.5, age_days / half_life_days)


def retrieve(
    conn: sqlite3.Connection,
    *,
    query_text: str,
    lang: str | None = None,
    area: str | None = None,
    top_k: int | None = None,
    query_vector: np.ndarray | None = None,
) -> tuple[list[dict], int]:
    """Select rules to inject for a task.

    Returns (rules, embed_tokens). Meta rules come first and are unconditional;
    the rest are tag-gated then ranked by cosine * confidence * recency.
    """
    k = config.RETRIEVAL_TOP_K if top_k is None else top_k
    rows = conn.execute("SELECT * FROM rules WHERE status='active'").fetchall()
    if not rows:
        return [], 0

    # `is_meta` exempts a rule from VECTOR RANKING, not from the scope gate.
    #
    # The flag exists because reporting-behaviour rules ("never report style
    # nits") describe how to report rather than a code topic, so they score
    # lowest and flattest against every task and cosine ranking never selects
    # them. It does NOT mean "applies everywhere": the distiller legitimately
    # produces language-scoped meta rules such as "in Python, don't report
    # missing timeouts", and injecting one of those into a Go review is a
    # cross-language leak. Measured: exempting meta rules from the gate leaked a
    # python-scoped rule into the Go task on every turn.
    #
    # So both tiers pass through _scope_allows; only ranking differs.
    allowed = [r for r in rows if _scope_allows(r, lang, area)]
    meta = [r for r in allowed if r["is_meta"]]
    candidates = [r for r in allowed if not r["is_meta"]]

    embed_tokens = 0
    ranked: list[dict] = []
    if candidates:
        if query_vector is None:
            query_vector, embed_tokens = embed.embed_one(query_text)
        scored: list[tuple[float, float, sqlite3.Row]] = []
        for row in candidates:
            if not row["embedding"]:
                continue
            sim = float(embed.from_blob(row["embedding"]) @ query_vector)
            score = sim * float(row["confidence"]) * _recency_weight(row["updated_at"])
            scored.append((score, sim, row))
        scored.sort(key=lambda t: t[0], reverse=True)
        for score, sim, row in scored[:k]:
            d = row_to_dict(row)
            d["_similarity"] = round(sim, 4)
            d["_score"] = round(score, 4)
            ranked.append(d)

    selected: list[dict] = []
    for row in meta:
        d = row_to_dict(row)
        d["_similarity"] = None  # unconditional; not ranked
        d["_score"] = None
        selected.append(d)
    selected.extend(ranked)

    if selected:
        conn.executemany(
            "UPDATE rules SET hits = hits + 1 WHERE id = ?",
            [(r["id"],) for r in selected],
        )
        conn.commit()
    return selected, embed_tokens


# --- compliance bookkeeping ----------------------------------------------


def record_compliance(
    conn: sqlite3.Connection, results: Iterable[tuple[int, bool]]
) -> None:
    """Record whether each injected rule was actually honored.

    A rule with many hits but applied_ok stuck at 0 is retrieved yet never
    obeyed -- usually a sign its wording is unusable, so it gets flagged.
    """
    payload = list(results)
    if not payload:
        return
    for rule_id, ok in payload:
        conn.execute(
            """UPDATE rules
                 SET applied_total = applied_total + 1,
                     applied_ok    = applied_ok + ?,
                     updated_at    = ?
               WHERE id = ?""",
            (1 if ok else 0, _now(), rule_id),
        )
    conn.execute(
        """UPDATE rules SET needs_review = 1
             WHERE status='active' AND applied_total >= 3 AND applied_ok = 0"""
    )
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """SELECT
             COUNT(*)                                        AS total,
             SUM(status='active')                            AS active,
             SUM(status='superseded')                        AS superseded,
             SUM(is_meta AND status='active')                AS meta,
             SUM(needs_review AND status='active')           AS needs_review,
             AVG(CASE WHEN status='active' THEN confidence END) AS avg_conf
           FROM rules"""
    ).fetchone()
    d = {k: (row[k] or 0) for k in row.keys()}
    d["avg_conf"] = round(float(d["avg_conf"] or 0), 3)
    return d
