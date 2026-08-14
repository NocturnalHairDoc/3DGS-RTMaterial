"""Bounded command-state history with optional coalescing."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any


@dataclass
class _Entry:
    state: Any
    key: str | None
    timestamp: float


class UndoManager:
    """Store compact application snapshots, never GPU/Gaussian tensors."""

    def __init__(self, limit: int = 32, coalesce_seconds: float = 0.35):
        if limit < 2:
            raise ValueError("history limit must be at least two")
        self.limit = limit
        self.coalesce_seconds = coalesce_seconds
        self._undo: deque[_Entry] = deque(maxlen=limit)
        self._redo: deque[_Entry] = deque(maxlen=limit)

    def record(self, state: Any, key: str | None = None, now: float | None = None) -> None:
        stamp = monotonic() if now is None else now
        entry = _Entry(deepcopy(state), key, stamp)
        if (key is not None and self._undo and self._undo[-1].key == key
                and stamp - self._undo[-1].timestamp <= self.coalesce_seconds):
            self._undo[-1] = entry
        else:
            self._undo.append(entry)
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 1

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> Any | None:
        if not self.can_undo:
            return None
        self._redo.append(self._undo.pop())
        return deepcopy(self._undo[-1].state)

    def redo(self) -> Any | None:
        if not self._redo:
            return None
        entry = self._redo.pop()
        self._undo.append(entry)
        return deepcopy(entry.state)

