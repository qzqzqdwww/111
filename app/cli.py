"""Interactive CLI. Same four operations the web UI exposes.

    python -m app.cli preflight              check both models are reachable
    python -m app.cli review billing         run a review (memory ON by default)
    python -m app.cli review pay --no-memory baseline, nothing injected
    python -m app.cli feedback               act on the last review's findings
    python -m app.cli rules                  browse the memory store
    python -m app.cli metrics                cost and latency summary
    python -m app.cli demo                   full learn-then-apply loop

All business logic lives in app.pipeline; this module only renders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import catalog, config, memory, metrics, pipeline
from .llm import shared

console = Console()

# Where `review` stashes its findings so `feedback` can act on them.
LAST_REVIEW = config.DATA_DIR / "last_review.json"

SEV_STYLE = {"high": "bold red", "medium": "yellow", "low": "dim"}
CAT_STYLE = {"style": "dim magenta"}


def _findings_table(findings: list[dict], title: str = "Findings") -> Table:
    t = Table(title=title, show_lines=False, header_style="bold")
    t.add_column("#", width=3, justify="right")
    t.add_column("location", style="cyan", no_wrap=True)
    t.add_column("category", width=15)
    t.add_column("sev", width=6)
    t.add_column("message")
    for i, f in enumerate(findings):
        t.add_row(
            str(i),
            f"{f['file']}:{f['line']}",
            f"[{CAT_STYLE.get(f['category'], '')}]{f['category']}[/]"
            if CAT_STYLE.get(f["category"])
            else f["category"],
            f"[{SEV_STYLE.get(f['severity'], '')}]{f['severity']}[/]",
            f["message"],
        )
    return t


def _rules_panel(rules: list[dict]) -> Panel:
    if not rules:
        return Panel("[dim]none — memory is empty or nothing matched[/]",
                     title="Injected rules", border_style="dim")
    lines = []
    for r in rules:
        scope = r["scope"] + (f"/{r['lang']}" if r.get("lang") else "")
        meta = "[cyan]META[/]" if r.get("is_meta") else f"sim={r.get('_similarity')}"
        lines.append(
            f"[bold]{r['rule_text']}[/]\n"
            f"  [dim]{scope} · conf={r['confidence']} · {meta} · "
            f"{r['assert_kind']}[/]"
        )
    return Panel("\n".join(lines), title=f"Injected rules ({len(rules)})",
                 border_style="green")


def _cost_line(payload: dict[str, Any]) -> str:
    c, t, u = payload["cost"], payload["timing"], payload["usage"]
    return (
        f"[bold]latency[/] user-visible {t['user_visible']}s "
        f"(retrieval {t['retrieval']}s + generation {t['generation']}s)   "
        f"[bold]cost[/] total ${c['total_usd']:.5f} "
        f"(memory ${c['memory_usd']:.8f})   "
        f"[bold]cache[/] read {u['cache_read']} write {u['cache_write']} "
        f"hit={u['cache_hit_ratio']}"
    )


# --- commands -------------------------------------------------------------


def cmd_preflight(args: argparse.Namespace) -> int:
    res = pipeline.preflight()
    t = Table(title="Preflight", header_style="bold")
    t.add_column("role")
    t.add_column("model")
    t.add_column("status")
    for role, info in res["models"].items():
        t.add_row(
            role,
            info["model"],
            "[green]ok[/]" if info["ok"] else f"[red]FAIL[/] {info['detail'][:60]}",
        )
    console.print(t)
    console.print(f"base_url: [dim]{config.BASE_URL}[/]")
    return 0 if res["ok"] else 1


def cmd_review(args: argparse.Namespace) -> int:
    conn = memory.connect()
    memory_on = not args.no_memory
    console.print(
        f"reviewing [bold cyan]{args.task_id}[/] "
        f"(memory [{'green' if memory_on else 'yellow'}]"
        f"{'ON' if memory_on else 'OFF'}[/])…"
    )
    with console.status("running agent…"):
        res = pipeline.review_task(conn, args.task_id, memory_on=memory_on)

    console.print(_rules_panel(res["injected_rules"]))
    console.print(_findings_table(res["findings"], f"Findings — {res['task']['path']}"))
    console.print(_cost_line(res))
    if res.get("hit_round_cap"):
        console.print("[yellow]note:[/] phase 1 hit the tool-round cap")

    LAST_REVIEW.write_text(
        json.dumps(
            {"task_id": args.task_id, "findings": res["findings"]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    console.print(
        f"[dim]saved to {LAST_REVIEW.name}; run "
        f"`python -m app.cli feedback` to teach the agent[/]"
    )
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    if not LAST_REVIEW.exists():
        console.print("[red]no saved review.[/] Run `python -m app.cli review <task>` first.")
        return 1
    saved = json.loads(LAST_REVIEW.read_text(encoding="utf-8"))
    findings = saved["findings"]
    task_id = saved["task_id"]

    console.print(_findings_table(findings, f"Findings from {task_id}"))

    payload: list[dict] = []
    if args.delete or args.raise_sev:
        for i in args.delete or []:
            if 0 <= i < len(findings):
                payload.append({"index": i, "action": "delete", "note": args.note})
        for i in args.raise_sev or []:
            if 0 <= i < len(findings):
                payload.append(
                    {
                        "index": i,
                        "action": "edit",
                        "replacement": {**findings[i], "severity": "high"},
                        "note": args.note,
                    }
                )
    else:
        console.print(
            "\n[bold]Act on each finding.[/] "
            "[dim]d = delete (stop reporting this class), "
            "h = raise to high, Enter = keep[/]"
        )
        for i, f in enumerate(findings):
            ans = console.input(
                f"  #{i} [{f['category']}/{f['severity']}] "
                f"{f['message'][:56]} → "
            ).strip().lower()
            if ans.startswith("d"):
                note = console.input("     why? [dim](optional)[/] ").strip() or None
                payload.append({"index": i, "action": "delete", "note": note})
            elif ans.startswith("h"):
                payload.append(
                    {
                        "index": i,
                        "action": "edit",
                        "replacement": {**findings[i], "severity": "high"},
                    }
                )

    if not payload:
        console.print("[yellow]no changes — nothing to learn.[/]")
        return 0

    conn = memory.connect()
    with console.status("distilling conventions…"):
        res = pipeline.apply_feedback(
            conn, task_id, findings=findings, payload=payload
        )

    t = Table(title="Distilled conventions", header_style="bold")
    t.add_column("scope", width=16)
    t.add_column("meta", width=5)
    t.add_column("conf", width=5, justify="right")
    t.add_column("assertion", width=34)
    t.add_column("rule")
    for r in res["rules"]:
        scope = r["scope"] + (f"/{r['lang']}" if r.get("lang") else "")
        assertion = r["assert_kind"]
        operand = r.get("assert_cat") or r.get("assert_field") or ""
        if operand:
            assertion += f"({operand}"
            assertion += f",{r['assert_sev']})" if r.get("assert_sev") else ")"
        t.add_row(
            scope,
            "yes" if r["is_meta"] else "",
            f"{r['confidence']:.2f}",
            assertion,
            f"{r['rule_text']}  [dim]({r['action']})[/]",
        )
    console.print(t)
    console.print(
        f"actions={res['actions']}   distill {res['t_distill']}s   "
        f"cost ${res['cost']['memory_usd']:.6f}"
    )
    return 0


def cmd_rules(args: argparse.Namespace) -> int:
    conn = memory.connect()
    rules = memory.list_rules(conn, status=None if args.all else "active")
    if not rules:
        console.print("[dim]memory store is empty.[/]")
        return 0
    t = Table(title=f"Memory store ({len(rules)} rules)", header_style="bold")
    t.add_column("id", width=3, justify="right")
    t.add_column("scope", width=16)
    t.add_column("meta", width=4)
    t.add_column("ev", width=3, justify="right")
    t.add_column("conf", width=5, justify="right")
    t.add_column("hits", width=4, justify="right")
    t.add_column("ok/n", width=6, justify="right")
    t.add_column("status", width=11)
    t.add_column("rule")
    for r in rules:
        scope = r["scope"] + (f"/{r['lang']}" if r.get("lang") else "")
        flag = " [yellow]⚠[/]" if r["needs_review"] else ""
        t.add_row(
            str(r["id"]),
            scope,
            "yes" if r["is_meta"] else "",
            str(r["evidence_count"]),
            f"{r['confidence']:.2f}",
            str(r["hits"]),
            f"{r['applied_ok']}/{r['applied_total']}",
            r["status"],
            r["rule_text"][:64] + flag,
        )
    console.print(t)
    s = memory.stats(conn)
    console.print(
        f"active={s['active']} meta={s['meta']} superseded={s['superseded']} "
        f"needs_review={s['needs_review']} avg_conf={s['avg_conf']}"
    )
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    conn = memory.connect()
    s = metrics.summary(conn)
    on, off = s["review"]["memory_on"], s["review"]["memory_off"]

    t = Table(title="Review turns", header_style="bold")
    t.add_column("metric", width=26)
    t.add_column("memory OFF", justify="right", width=14)
    t.add_column("memory ON", justify="right", width=14)
    rows = [
        ("turns", "turns", "{:d}"),
        ("avg user-visible (s)", "avg_user_visible", "{:.2f}"),
        ("avg retrieval (s)", "avg_retrieval", "{:.3f}"),
        ("avg generation (s)", "avg_generation", "{:.2f}"),
        ("avg cost (USD)", "avg_cost", "{:.5f}"),
        ("total cost (USD)", "sum_cost", "{:.5f}"),
        ("memory cost (USD)", "sum_memory_cost", "{:.8f}"),
        ("cache read tokens", "cache_read", "{:d}"),
        ("avg rules injected", "avg_rules", "{:.1f}"),
        ("avg findings", "avg_findings", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        t.add_row(
            label,
            fmt.format(off.get(key, 0) or 0),
            fmt.format(on.get(key, 0) or 0),
        )
    console.print(t)

    fb = s["feedback"]
    if fb.get("turns"):
        console.print(
            f"\n[bold]feedback turns[/] {fb['turns']}   "
            f"avg distill {fb['avg_distill']:.2f}s   "
            f"cost ${fb['sum_memory_cost']:.6f}  "
            f"[dim](off the user's critical path)[/]"
        )
    tot = s["totals"]
    console.print(
        f"[bold]totals[/] cost ${tot['total_cost_usd']:.5f}   "
        f"memory ${tot['memory_cost_usd']:.6f} "
        f"({tot['memory_cost_share'] * 100:.1f}% of spend)   "
        f"cache hit {tot['cache_hit_ratio'] * 100:.1f}%"
    )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Full loop: review, teach, then show the lesson applied elsewhere."""
    from eval import cases

    conn = memory.connect()
    console.rule("[bold]1. review pay.py — nothing learned yet")
    first = pipeline.review_task(conn, "pay", memory_on=True)
    console.print(_rules_panel(first["injected_rules"]))
    console.print(_findings_table(first["findings"]))

    payload = cases.feedback_payload_for(first["findings"])
    if not payload:
        console.print(
            "[yellow]this run produced no style/observability findings to act on, "
            "so there is no feedback signal. Re-run to try again.[/]"
        )
        return 0

    console.rule("[bold]2. reviewer acts: delete nits, raise logging severity")
    fb = pipeline.apply_feedback(conn, "pay", findings=first["findings"], payload=payload)
    for r in fb["rules"]:
        console.print(
            f"  learned: [bold]{r['rule_text']}[/]\n"
            f"    [dim]{r['scope']} · conf={r['confidence']} · "
            f"{r['assert_kind']}[/]"
        )

    console.rule("[bold]3. review billing.py — learned rules should apply")
    second = pipeline.review_task(conn, "billing", memory_on=True)
    console.print(_rules_panel(second["injected_rules"]))
    console.print(_findings_table(second["findings"]))
    console.print(_cost_line(second))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Feedback-memory code review agent.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preflight", help="check model reachability").set_defaults(
        fn=cmd_preflight
    )

    r = sub.add_parser("review", help="review a task")
    r.add_argument("task_id", choices=catalog.ids())
    r.add_argument("--no-memory", action="store_true", help="baseline: inject nothing")
    r.set_defaults(fn=cmd_review)

    f = sub.add_parser("feedback", help="act on the last review's findings")
    f.add_argument("--delete", type=int, nargs="*", help="finding indices to delete")
    f.add_argument("--raise-sev", type=int, nargs="*", help="indices to raise to high")
    f.add_argument("--note", default=None, help="reason, passed to the distiller")
    f.set_defaults(fn=cmd_feedback)

    ru = sub.add_parser("rules", help="browse the memory store")
    ru.add_argument("--all", action="store_true", help="include superseded rules")
    ru.set_defaults(fn=cmd_rules)

    sub.add_parser("metrics", help="cost and latency summary").set_defaults(
        fn=cmd_metrics
    )
    sub.add_parser("demo", help="full learn-then-apply loop").set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        return 130
    except Exception as e:  # noqa: BLE001 - surface the failure, don't traceback-dump
        console.print(f"[red]error:[/] {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
