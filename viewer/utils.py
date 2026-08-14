"""Dependency-free helpers used by the GUI and CPU-only tests."""

from __future__ import annotations

import os


def scoped_output_path(path: str, directory: str, source_file: str) -> str:
    root = os.path.realpath(os.path.join(os.path.dirname(source_file), directory))
    resolved = os.path.realpath(os.path.abspath(path))
    if os.path.commonpath([root, resolved]) != root:
        raise ValueError(f"path must be inside {root}")
    return resolved


def visibility_cache_key(hidden_segments, scene_mask, segment_times: int):
    return (tuple(sorted(hidden_segments)), scene_mask.data_ptr(),
            getattr(scene_mask, "_version", 0), int(segment_times))
