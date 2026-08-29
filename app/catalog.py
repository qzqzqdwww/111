"""The review tasks available in the fixture repo.

Shared by the CLI, the web API, and the eval harness so a task id means the same
thing everywhere. `lang` and `area` are the retrieval tags -- they are known from
the task definition, so the common path needs no classification call at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    path: str
    lang: str
    area: str
    change_kind: str
    label: str

    @property
    def query_text(self) -> str:
        """The text embedded for retrieval. Describes the task, not the rules."""
        return (
            f"Review {self.lang} {self.change_kind} in {self.path}: {self.label}"
        )


TASKS: dict[str, Task] = {
    "pay": Task(
        task_id="pay",
        path="svc/pay.py",
        lang="python",
        area="svc/pay.py",
        change_kind="new-feature",
        label="add charge() posting to the payment endpoint",
    ),
    "billing": Task(
        task_id="billing",
        path="svc/billing.py",
        lang="python",
        area="svc/billing.py",
        change_kind="new-feature",
        label="add refund() posting to the refund endpoint",
    ),
    "auth": Task(
        task_id="auth",
        path="svc/auth.py",
        lang="python",
        area="svc/auth.py",
        change_kind="new-feature",
        label="add validate_token() hashing and posting a token",
    ),
    "worker": Task(
        task_id="worker",
        path="internal/queue/worker.go",
        lang="go",
        area="internal/queue/worker.go",
        change_kind="new-feature",
        label="add Dequeue() with a retry loop",
    ),
}


def get(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(
            f"unknown task {task_id!r}. Available: {sorted(TASKS)}"
        )
    return TASKS[task_id]


def ids() -> list[str]:
    return list(TASKS)
