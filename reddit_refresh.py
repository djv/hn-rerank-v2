"""Coalescing Reddit refresh worker, independent of core regeneration."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from database import Database
from pipeline.config import Config
from pipeline.ranking import Embedder
from reddit_limiter import RedditCircuitSnapshot, limiter as reddit_limiter


class RedditRefreshWorker:
    """Run at most one Reddit refresh and retain one pending rerun."""

    def __init__(
        self,
        config: Config,
        db: Database,
        embedder: Embedder,
        on_changed: Callable[[], None],
    ) -> None:
        self._config = config
        self._db = db
        self._embedder = embedder
        self._on_changed = on_changed
        self._condition = threading.Condition()
        self._pending = False
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="reddit-refresh", daemon=True
        )
        persisted_circuit = db.get_reddit_circuit_state()
        if persisted_circuit is not None:
            reddit_limiter.restore(RedditCircuitSnapshot(*persisted_circuit))
        self._thread.start()

    def submit(self) -> None:
        with self._condition:
            self._pending = True
            self._condition.notify()

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                self._pending = False
            try:
                from pipeline import refresh_reddit_candidates

                result = refresh_reddit_candidates(
                    self._config, self._db, self._embedder
                )
                if result.changed:
                    self._on_changed()
            except Exception:
                logging.exception("reddit_refresh_failed")
            finally:
                snapshot = reddit_limiter.snapshot()
                self._db.save_reddit_circuit_state(
                    snapshot.consecutive_429, snapshot.retry_at, time.time()
                )
