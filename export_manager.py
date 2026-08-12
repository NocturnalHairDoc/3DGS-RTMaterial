"""Thread-safe export orchestration; workers never call GUI APIs."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable

import numpy as np


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB in [0,1] to IEC 61966-2-1 sRGB."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb,
                    1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)


def estimate_frame_bytes(width: int, height: int, channels: int = 4,
                         working_buffers: int = 8) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return width * height * channels * 4 * working_buffers


@dataclass(frozen=True)
class ExportEvent:
    kind: str
    current: int = 0
    total: int = 0
    eta_seconds: float | None = None
    message: str = ""


class ExportCancelled(RuntimeError):
    pass


class ExportManager:
    """Runs one export at a time and reports progress through a queue."""

    def __init__(self) -> None:
        self.events: Queue[ExportEvent] = Queue()
        self._cancel = Event()
        self._lock = Lock()
        self._thread: Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, total: int, work: Callable[[Callable[[int], None], Event], None]) -> None:
        if total < 1:
            raise ValueError("total must be at least one")
        with self._lock:
            if self.running:
                raise RuntimeError("an export is already running")
            self._cancel.clear()

            def runner() -> None:
                started = monotonic()
                self.events.put(ExportEvent("started", total=total, message="Export started"))

                def progress(current: int) -> None:
                    if self._cancel.is_set():
                        raise ExportCancelled("Export cancelled")
                    elapsed = monotonic() - started
                    eta = elapsed / current * (total - current) if current else None
                    self.events.put(ExportEvent("progress", current, total, eta))

                try:
                    work(progress, self._cancel)
                    if self._cancel.is_set():
                        raise ExportCancelled("Export cancelled")
                except ExportCancelled as exc:
                    self.events.put(ExportEvent("cancelled", message=str(exc)))
                except Exception as exc:
                    self.events.put(ExportEvent("failed", message=f"{type(exc).__name__}: {exc}"))
                else:
                    self.events.put(ExportEvent("completed", total, total, 0.0, "Export completed"))

            self._thread = Thread(target=runner, name="3dgs-export", daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def drain_events(self) -> list[ExportEvent]:
        result = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except Empty:
                return result
