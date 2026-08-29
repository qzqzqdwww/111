"""Launcher for the packaged Windows .exe.

Presents a simple numbered menu so the user can choose between the CLI
commands and the web UI without needing to remember command-line flags.
"""

from __future__ import annotations

import sys
import subprocess
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def _run_web():
    """Start the uvicorn dev server and open the browser."""
    url = "http://127.0.0.1:8077"
    print(f"Starting web server at {url} ...")
    print("Press Ctrl+C in this window to stop the server.\n")

    # Open browser after a short delay so the server is ready
    import threading
    import time

    def _open():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

    sys.exit(
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.web:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8077",
            ],
            cwd=str(APP_DIR),
        ).returncode
    )


def _run_cli(argv: list[str]) -> int:
    """Run a CLI command and return its exit code."""
    return subprocess.run(
        [sys.executable, "-m", "app.cli", *argv],
        cwd=str(APP_DIR),
    ).returncode


MENU = """
=====================================
  Feedback-Memory Review Agent
=====================================

  1. Preflight check (test API connectivity)
  2. Review a task (with memory)
  3. Review a task (without memory)
  4. Demo (learn-then-apply loop)
  5. Browse rules
  6. View metrics
  7. Start Web UI
  8. Run A/B evaluation
  0. Exit

Choose: """


TASKS = ["pay", "billing", "auth", "worker"]


def _choose_task() -> str | None:
    print("\nAvailable tasks:")
    for i, t in enumerate(TASKS, 1):
        print(f"  {i}. {t}")
    print("  0. Cancel")
    try:
        choice = input("Task number: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if choice == "0":
        return None
    idx = int(choice) - 1
    if 0 <= idx < len(TASKS):
        return TASKS[idx]
    print("Invalid choice.")
    return None


def main() -> int:
    while True:
        try:
            choice = input(MENU).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice == "0":
            return 0

        if choice == "1":
            rc = _run_cli(["preflight"])

        elif choice == "2":
            task = _choose_task()
            if task:
                rc = _run_cli(["review", task])

        elif choice == "3":
            task = _choose_task()
            if task:
                rc = _run_cli(["review", task, "--no-memory"])

        elif choice == "4":
            rc = _run_cli(["demo"])

        elif choice == "5":
            rc = _run_cli(["rules"])

        elif choice == "6":
            rc = _run_cli(["metrics"])

        elif choice == "7":
            _run_web()
            continue  # _run_web never returns normally

        elif choice == "8":
            rc = _run_cli(["-m", "eval.run_ab"])

        else:
            print("Invalid choice.")
            continue

        if rc:
            print(f"\n[exit code: {rc}]")
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    raise SystemExit(main())
