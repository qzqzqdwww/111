"""A/B harness: does injected memory change the output?

Runs the same task sequence twice against the same seeded rule set:

    OFF  no rules injected  -> baseline compliance (what the model does anyway)
    ON   rules injected     -> compliance with memory

The OFF arm is not decoration. Measured, the baseline already cites file:line
and populates evidence unprompted, so a rule requiring those scores 100% in both
arms. Reporting ON alone would credit memory for behaviour the model had for
free. Only the ON-minus-OFF delta is evidence that memory did something.

Usage:
    python -m eval.run_ab                  # full run, fresh in-memory store
    python -m eval.run_ab --tasks billing  # single task
    python -m eval.run_ab --json out.json  # machine-readable results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from app import memory, pipeline
from app.llm import Usage, shared
from eval import cases, checkers


def _rule_rows(conn) -> list[dict]:
    """All active rules, as the checker expects them."""
    return memory.list_rules(conn, status="active")


def _relevant_rules(all_rules: list[dict], lang: str) -> list[dict]:
    """Rules that SHOULD apply to a task of this language.

    Meta and global rules always apply. Language rules apply only to their own
    language -- so a Python rule evaluated against the Go task is precisely the
    leak check.
    """
    out = []
    for r in all_rules:
        if r["is_meta"] or r["scope"] == "global":
            out.append(r)
        elif r["lang"] == lang:
            out.append(r)
    return out


def _foreign_rules(all_rules: list[dict], lang: str) -> list[dict]:
    """Rules scoped to a DIFFERENT language -- these must never be injected."""
    return [
        r
        for r in all_rules
        if not r["is_meta"] and r["scope"] != "global" and r["lang"] not in (None, lang)
    ]


def run_arm(
    conn,
    task_ids: list[str],
    *,
    memory_on: bool,
    llm,
    judge_usage: Usage,
) -> list[dict[str, Any]]:
    """Run one arm of the A/B over the task sequence."""
    all_rules = _rule_rows(conn)
    rows: list[dict[str, Any]] = []

    for task_id in task_ids:
        res = pipeline.review_task(conn, task_id, memory_on=memory_on, llm=llm)
        lang = res["task"]["lang"]
        findings = res["findings"]

        # Score against the rules that SHOULD govern this task, regardless of
        # arm. In the OFF arm nothing was injected, so this measures the
        # model's unaided baseline against the same bar.
        governing = _relevant_rules(all_rules, lang)
        verdicts = checkers.check_all(
            governing, findings, llm=llm, usage=judge_usage, allow_judge=True
        )
        rate, passed, counted = checkers.compliance_rate(verdicts)

        injected_ids = {r["id"] for r in res["injected_rules"]}
        foreign = _foreign_rules(all_rules, lang)
        leaked = [r for r in foreign if r["id"] in injected_ids]

        rows.append(
            {
                "task_id": task_id,
                "lang": lang,
                "memory_on": memory_on,
                "n_findings": len(findings),
                "n_injected": len(res["injected_rules"]),
                "n_leaked": len(leaked),
                "leaked_rules": [r["rule_text"][:60] for r in leaked],
                "compliance_rate": rate,
                "passed": passed,
                "counted": counted,
                "verdicts": [v.as_dict() for v in verdicts],
                "findings": findings,
                "categories": sorted({f["category"] for f in findings}),
                "timing": res["timing"],
                "cost": res["cost"],
                "usage": res["usage"],
            }
        )

        # Record compliance only for rules that were actually injected -- a rule
        # cannot be blamed for output it never influenced.
        if memory_on:
            memory.record_compliance(
                conn,
                [
                    (v.rule_id, v.outcome == checkers.PASS)
                    for v in verdicts
                    if v.rule_id in injected_ids and v.counted
                ],
            )
    return rows


def _per_rule_table(off_rows: list[dict], on_rows: list[dict]) -> list[dict]:
    """Per-rule OFF vs ON compliance, aggregated across tasks."""
    acc: dict[str, dict[str, Any]] = {}
    for arm, rows in (("off", off_rows), ("on", on_rows)):
        for row in rows:
            for v in row["verdicts"]:
                key = v["rule_text"]
                e = acc.setdefault(
                    key,
                    {
                        "rule_text": key,
                        "assert_kind": v["assert_kind"],
                        "subjective": v["subjective"],
                        "off_pass": 0,
                        "off_counted": 0,
                        "on_pass": 0,
                        "on_counted": 0,
                        "off_vacuous": 0,
                        "on_vacuous": 0,
                    },
                )
                if v["outcome"] == checkers.VACUOUS:
                    e[f"{arm}_vacuous"] += 1
                    continue
                e[f"{arm}_counted"] += 1
                if v["outcome"] == checkers.PASS:
                    e[f"{arm}_pass"] += 1

    out = []
    for e in acc.values():
        off_rate = e["off_pass"] / e["off_counted"] if e["off_counted"] else None
        on_rate = e["on_pass"] / e["on_counted"] if e["on_counted"] else None
        e["off_rate"] = None if off_rate is None else round(off_rate, 3)
        e["on_rate"] = None if on_rate is None else round(on_rate, 3)
        e["delta"] = (
            None
            if off_rate is None or on_rate is None
            else round(on_rate - off_rate, 3)
        )
        out.append(e)
    return out


def _arm_totals(rows: list[dict]) -> dict[str, Any]:
    counted = sum(r["counted"] for r in rows)
    passed = sum(r["passed"] for r in rows)
    return {
        "tasks": len(rows),
        "compliance_rate": round(passed / counted, 4) if counted else 0.0,
        "passed": passed,
        "counted": counted,
        "leaks": sum(r["n_leaked"] for r in rows),
        "avg_user_visible_s": round(
            sum(r["timing"]["user_visible"] for r in rows) / max(1, len(rows)), 2
        ),
        "avg_retrieval_s": round(
            sum(r["timing"]["retrieval"] for r in rows) / max(1, len(rows)), 3
        ),
        "total_cost_usd": round(sum(r["cost"]["total_usd"] for r in rows), 6),
        "memory_cost_usd": round(sum(r["cost"]["memory_usd"] for r in rows), 8),
        "cache_read": sum(r["usage"]["cache_read"] for r in rows),
        "cache_write": sum(r["usage"]["cache_write"] for r in rows),
        "avg_findings": round(
            sum(r["n_findings"] for r in rows) / max(1, len(rows)), 2
        ),
    }


def run(
    task_ids: list[str] | None = None,
    *,
    db: str = ":memory:",
    repeats: int = 1,
) -> dict[str, Any]:
    """Run both arms. `repeats` re-runs the whole sequence to average out noise.

    Repeats matter more than they look. Measured on a single pass, a rule of the
    form "report category X at severity >= S" was scored against just one or two
    relevant findings per task, and the model's finding set varies run to run --
    one pass emitted a single high error-handling finding (scoring a lucky pass),
    the next split the same problem into two findings and rated one medium
    (scoring a fail). With denominators that small, a per-rule rate is noise.
    Default 1 keeps the demo fast; use 3+ for a number worth quoting.
    """
    base = task_ids or cases.AB_TASKS
    sequence = list(base) * max(1, repeats)
    conn = memory.connect(db)
    llm = shared()
    task_ids = base

    # Seed the fixed rule set so both arms face identical conventions.
    for rule in cases.SEED_RULES:
        memory.add_rule(conn, rule)

    judge_usage = Usage()
    t0 = time.perf_counter()

    off_rows = run_arm(
        conn, sequence, memory_on=False, llm=llm, judge_usage=judge_usage
    )
    on_rows = run_arm(conn, sequence, memory_on=True, llm=llm, judge_usage=judge_usage)

    return {
        "task_ids": task_ids,
        "repeats": max(1, repeats),
        "seeded_rules": [
            {"rule_text": r["rule_text"], "scope": r["scope"], "lang": r["lang"],
             "assert_kind": r["assert_kind"], "is_meta": r["is_meta"]}
            for r in _rule_rows(conn)
        ],
        "off": {"rows": off_rows, "totals": _arm_totals(off_rows)},
        "on": {"rows": on_rows, "totals": _arm_totals(on_rows)},
        "per_rule": _per_rule_table(off_rows, on_rows),
        "judge_usage": judge_usage.as_dict(),
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }


# --- reporting ------------------------------------------------------------


def _fmt_rate(v: float | None) -> str:
    return "  n/a" if v is None else f"{v * 100:5.1f}%"


def print_report(res: dict[str, Any]) -> None:
    off, on = res["off"]["totals"], res["on"]["totals"]

    print("\n" + "=" * 78)
    print("A/B: memory OFF vs ON")
    print("=" * 78)
    reps = res.get("repeats", 1)
    print(
        f"tasks: {', '.join(res['task_ids'])}    repeats: {reps}    "
        f"wall clock: {res['wall_clock_s']}s"
    )
    if reps < 3:
        print(
            "note: per-rule denominators are small at repeats<3; the model's "
            "finding set varies run to run, so treat single-pass per-rule rates\n"
            "      as indicative and use --repeats 3 for a number worth quoting."
        )

    print("\n-- criterion 3: memory effectiveness (rule compliance) " + "-" * 23)
    print(f"{'':22} {'OFF':>10} {'ON':>10} {'delta':>10}")
    d = on["compliance_rate"] - off["compliance_rate"]
    print(
        f"{'compliance rate':22} {_fmt_rate(off['compliance_rate']):>10} "
        f"{_fmt_rate(on['compliance_rate']):>10} {d * 100:+9.1f}pp"
    )
    off_frac = f"{off['passed']}/{off['counted']}"
    on_frac = f"{on['passed']}/{on['counted']}"
    print(f"{'  (passed/counted)':22} {off_frac:>10} {on_frac:>10}")
    print(
        f"{'cross-language leaks':22} {off['leaks']:>10} {on['leaks']:>10}"
        "   <- must be 0"
    )

    print("\n-- per-rule compliance " + "-" * 55)
    print(f"{'rule':44} {'OFF':>7} {'ON':>7} {'delta':>7}  kind")
    for e in res["per_rule"]:
        mark = " (subj)" if e["subjective"] else ""
        delta = "    n/a" if e["delta"] is None else f"{e['delta'] * 100:+6.0f}pp"
        print(
            f"{e['rule_text'][:44]:44} {_fmt_rate(e['off_rate']):>7} "
            f"{_fmt_rate(e['on_rate']):>7} {delta:>7}  {e['assert_kind']}{mark}"
        )
        if e["off_vacuous"] or e["on_vacuous"]:
            print(
                f"{'':44} vacuous: OFF={e['off_vacuous']} ON={e['on_vacuous']}"
                "  (excluded from rate)"
            )

    print("\n-- criterion 2: conversation speed " + "-" * 43)
    print(f"{'':22} {'OFF':>10} {'ON':>10}")
    print(
        f"{'user-visible (s)':22} {off['avg_user_visible_s']:>10} "
        f"{on['avg_user_visible_s']:>10}"
    )
    print(
        f"{'  of which retrieval':22} {off['avg_retrieval_s']:>10} "
        f"{on['avg_retrieval_s']:>10}"
    )

    print("\n-- criterion 1: memory cost " + "-" * 50)
    print(f"{'':22} {'OFF':>12} {'ON':>12}")
    print(
        f"{'total (USD)':22} {off['total_cost_usd']:>12.6f} "
        f"{on['total_cost_usd']:>12.6f}"
    )
    print(
        f"{'memory only (USD)':22} {off['memory_cost_usd']:>12.8f} "
        f"{on['memory_cost_usd']:>12.8f}"
    )
    print(
        f"{'cache read tokens':22} {off['cache_read']:>12} {on['cache_read']:>12}"
    )
    ju = res["judge_usage"]
    if ju["calls"]:
        print(
            f"\njudge calls (subjective rules only): {ju['calls']}, "
            f"${ju['cost_usd']:.6f}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the memory A/B evaluation.")
    p.add_argument(
        "--tasks", nargs="*", default=None, help="task ids (default: billing auth worker)"
    )
    p.add_argument("--json", default=None, help="write full results to this path")
    p.add_argument(
        "--db", default=":memory:", help="sqlite path (default: fresh in-memory)"
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "re-run the whole sequence N times to average out run-to-run "
            "variation in the model's finding set (default 1; use 3+ to quote)"
        ),
    )
    args = p.parse_args(argv)

    res = run(args.tasks, db=args.db, repeats=args.repeats)
    print_report(res)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")

    # Non-zero exit if a Python rule leaked into the Go task -- that is a real
    # regression in the retrieval gate, not a soft metric.
    return 1 if res["on"]["totals"]["leaks"] else 0


if __name__ == "__main__":
    sys.exit(main())
