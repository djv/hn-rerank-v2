from __future__ import annotations

import asyncio

from textual.pilot import Pilot
from textual.widgets import Markdown
from textual.worker import WorkerState

# The impression worker sleeps for a second by design; waiting on it would
# reintroduce the fixed delay this helper replaces.
_IGNORED_GROUPS = {"impression"}


async def settle(pilot: Pilot, timeout: float = 5.0) -> None:
    """Wait until the reader has no pending work instead of sleeping blindly."""
    app = pilot.app
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        await pilot.pause()
        busy = any(
            worker.state in (WorkerState.PENDING, WorkerState.RUNNING)
            and worker.group not in _IGNORED_GROUPS
            for worker in app.workers
        )
        loading = any(
            "Loading summary" in widget._markdown for widget in app.query(Markdown)
        )
        busy = busy or bool(getattr(app, "summary_requests", None))
        if not busy and not loading:
            break
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("app did not settle")
        await pilot.pause(0.02)
    await pilot.pause()
