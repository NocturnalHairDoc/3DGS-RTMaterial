"""Thread-safe export orchestration; workers never call GUI APIs."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable
import copy

import numpy as np
import torch


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


class FrozenGaussianSnapshot:
    """Tensor clone exposing the subset of GaussianModel used by exporters."""

    def __init__(self, model) -> None:
        for name in ("_xyz", "_rotation", "_scaling", "_opacity",
                     "_features_dc", "_features_rest"):
            setattr(self, name, getattr(model, name).detach().clone())
        self.active_sh_degree = int(getattr(model, "active_sh_degree", 3))

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        from utils.general_utils import build_scaling_rotation, strip_symmetric
        matrix = build_scaling_rotation(
            scaling_modifier * self.get_scaling, self._rotation)
        return strip_symmetric(matrix @ matrix.transpose(1, 2))


@dataclass(frozen=True)
class ExportSnapshot:
    """Immutable export inputs captured before the background worker starts."""

    scene: FrozenGaussianSnapshot
    scene_mask: torch.Tensor
    visible_mask: torch.Tensor | None
    material_assignments: dict
    camera: object
    render_state: dict | None = None


def capture_export_snapshot(model, camera, material_assignments, hidden_segments,
                            render_state=None) -> ExportSnapshot:
    scene_mask = model._mask.detach().clone()
    hidden = torch.zeros_like(scene_mask, dtype=torch.bool)
    for segment_id in hidden_segments:
        hidden |= scene_mask == int(segment_id) + 1
    visible = (~hidden).contiguous() if bool(hidden.any().item()) else None
    return ExportSnapshot(
        scene=FrozenGaussianSnapshot(model),
        scene_mask=scene_mask,
        visible_mask=visible,
        material_assignments=copy.deepcopy(material_assignments),
        camera=copy.deepcopy(camera),
        render_state=copy.deepcopy(render_state),
    )


def compose_depth_ordered_ids(layers, height: int, width: int,
                              opacity_threshold: float = 1e-4) -> np.ndarray:
    """Compose integer IDs by the nearest valid depth, never by opacity size."""
    ids = np.zeros((height, width), dtype=np.uint16)
    best_depth = np.full((height, width), np.inf, dtype=np.float32)
    for value, output in layers:
        if value <= 0 or value > np.iinfo(np.uint16).max:
            raise ValueError(f"export ID {value} is outside uint16 range")
        depth = np.asarray(output["depth"], dtype=np.float32)
        opacity = np.asarray(output["opacity"], dtype=np.float32)[..., 0]
        hit = (np.isfinite(depth) & (depth > 0.0)
               & np.isfinite(opacity) & (opacity > opacity_threshold))
        replace = hit & (depth < best_depth)
        best_depth[replace] = depth[replace]
        ids[replace] = np.uint16(value)
    return ids


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
