"""Offline editorial UI inspection with synthetic stories; never accesses a profile."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace

from textual.pilot import Pilot
from textual.widgets import Select

from hn_rerank.app import Reader

from .test_editorial import EditorialServer


async def inspect(pilot: Pilot) -> None:
    await pilot.pause(3)
    pilot.app.save_screenshot("hn-editorial-populated.svg", path="/tmp")
    await pilot.press("enter", "j", "j", "escape", "j", "1", "u", "?", "r")
    await pilot.pause(3)
    app = pilot.app
    if isinstance(app, Reader):
        app.query_one("#age", Select).value = "archive"
        app.query_one("#sort", Select).value = "popular"
        await pilot.pause()
        app.save_screenshot("hn-editorial-empty.svg", path="/tmp")
        app.status("Could not reach server. Press r to retry.", error=True)
        app.workers.cancel_group(app, "refresh")
        app.workers.cancel_group(app, "vote")
        app.workers.cancel_group(app, "summary")
        await pilot.pause(1.0)
        app.feed = None
        app.stories = []
        app.show_failure("Could not reach server.")
        await pilot.pause()
        app.save_screenshot("hn-editorial-error.svg", path="/tmp")
        app.setup()
        await pilot.pause()
        app.save_screenshot("hn-editorial-setup.svg", path="/tmp")
    if not pilot.app.is_headless:
        await asyncio.sleep(25)
    pilot.app.exit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    fake = EditorialServer()
    fake.feed.stories[0] = replace(
        fake.feed.stories[0],
        title="Building a calmer terminal reader with careful typography and reliable keyboard navigation",
        time=1789250000,
    )
    app = Reader(api=fake.api())
    app.run(
        headless=args.headless,
        size=(140, 40) if args.headless else None,
        auto_pilot=inspect,
    )


if __name__ == "__main__":
    main()
