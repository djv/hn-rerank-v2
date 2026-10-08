"""Users' latest classifier fits on disk, as warm starts across restarts.

The joined classifier and the linear blend warm-start each fit from the
user's previous one (``joined_classifier.latest``, ``linear_blend.latest``).
Those live in memory, so a restart's first rerank fit every model cold:
on 2026-10-08 that was ~60 s of a 92 s rerank, against ~3 s warm. Saved
fits only seed the next fit's solver; they never score anything.

Writes are debounced (a later fit replaces a pending one) and done by a
background thread, atomically. Disabled until :func:`configure` (the
server's startup), so tests and scripts write nothing.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Arrays = dict[str, NDArray[np.generic]]

WRITE_DELAY_SECONDS = 30.0

_cond = threading.Condition()
_directory: Path | None = None
_pending: dict[str, Arrays] = {}
_writer: threading.Thread | None = None


def default_directory() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "hn-rewrite" / "warm_start"


def configure(directory: Path | None) -> None:
    """Store fits under *directory*; None disables the store."""
    global _directory, _writer
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    with _cond:
        _directory = directory
        _pending.clear()
        if directory is not None and _writer is None:
            _writer = threading.Thread(target=_run, name="warm-store", daemon=True)
            _writer.start()


def enabled() -> bool:
    with _cond:
        return _directory is not None


def save(name: str, arrays: Arrays) -> None:
    """Queue *arrays* as fit *name*; a no-op while the store is disabled."""
    with _cond:
        if _directory is None:
            return
        _pending[name] = arrays
        _cond.notify()


def load(name: str) -> Arrays | None:
    """The latest saved (or still queued) arrays of fit *name*, or None."""
    with _cond:
        directory = _directory
        pending = _pending.get(name)
    if directory is None:
        return None
    if pending is not None:
        return pending
    path = directory / f"{name}.npz"
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except FileNotFoundError:
        return None
    except Exception:
        logging.warning("warm_store: unreadable %s", path, exc_info=True)
        return None


def flush() -> None:
    """Write every queued fit now."""
    with _cond:
        directory = _directory
        batch = dict(_pending)
        _pending.clear()
    if directory is None:
        return
    for name, arrays in batch.items():
        tmp = directory / f".{name}.tmp.npz"
        try:
            np.savez_compressed(tmp, allow_pickle=False, **arrays)
            os.replace(tmp, directory / f"{name}.npz")
        except Exception:
            logging.warning("warm_store: could not write %s", name, exc_info=True)


def _run() -> None:
    while True:
        with _cond:
            while not _pending:
                _cond.wait()
        # A vote burst refits several times; write only the last fit.
        time.sleep(WRITE_DELAY_SECONDS)
        flush()
