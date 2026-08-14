"""Runtime policy for responsive, camera-correct interactive rendering."""

from __future__ import annotations

import os


def env_flag(name: str, default: bool = True) -> bool:
    fallback = "1" if default else "0"
    return os.environ.get(name, fallback).strip().lower() not in {
        "0", "false", "off", "no"
    }


class InteractiveRenderPolicy:
    """Select the responsive backend used during camera interaction."""

    def __init__(self, adaptive_preview: bool | None = None):
        self.enabled = (env_flag("RTM_ADAPTIVE_RT_PREVIEW") if adaptive_preview is None
                        else bool(adaptive_preview))

    def use_raster(self, moving: bool, moving_middle: bool) -> bool:
        """Use rasterization only while orbiting or panning."""
        return self.enabled and (bool(moving) or bool(moving_middle))
