"""Versioned, pickle-free persistence for segmentation and material editing state."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


STATE_VERSION = 2
APPLICATION_VERSION = "2.0.0-dev"


def model_fingerprint(model_path, scene_iteration=30000, feature_iteration=10000) -> str:
    """Stable fingerprint of the model artifacts without loading large PLY files."""
    root = Path(model_path).expanduser().resolve()
    paths = [
        root / "point_cloud" / f"iteration_{scene_iteration}" / "scene_point_cloud.ply",
        root / "point_cloud" / f"iteration_{feature_iteration}" / "contrastive_feature_point_cloud.ply",
        root / "point_cloud" / f"iteration_{feature_iteration}" / "scale_gate.pt",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        if not path.is_file():
            digest.update(b"missing")
            continue
        stat = path.stat()
        digest.update(f"{stat.st_size}".encode())
        with path.open("rb") as stream:
            digest.update(stream.read(65536))
    return digest.hexdigest()


def migrate_metadata(metadata: dict) -> dict:
    payload = dict(metadata)
    version = int(payload.get("version", 1))
    if version == 1:
        payload.setdefault("application_version", "1.x")
        payload.setdefault("saved_at", None)
        payload.setdefault("model_fingerprint", None)
        for assignment in payload.get("material_assignments", {}).values():
            assignment.setdefault("params", None)
        payload["version"] = 2
        version = 2
    if version != STATE_VERSION:
        raise ValueError(f"Unsupported project state version: {version}")
    return payload


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported project-state value: {type(value).__name__}")


def save_project_state(path, mask, metadata: dict) -> Path:
    """Save an integer Gaussian mask and JSON metadata in one compressed NPZ."""
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["version"] = STATE_VERSION
    payload["application_version"] = APPLICATION_VERSION
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()
    encoded = json.dumps(payload, ensure_ascii=False, default=_json_default)
    np.savez_compressed(
        destination,
        mask=np.asarray(mask, dtype=np.int32),
        metadata=np.asarray(encoded),
    )
    return destination


def load_project_state(path) -> tuple[np.ndarray, dict]:
    """Load and validate a state file without enabling NumPy pickle support."""
    source = Path(path).expanduser()
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"mask", "metadata"}:
            raise ValueError("Invalid project state: expected mask and metadata entries")
        mask = np.asarray(archive["mask"], dtype=np.int32)
        metadata = json.loads(str(archive["metadata"].item()))
    if mask.ndim != 1:
        raise ValueError(f"Invalid project state mask shape: {mask.shape}")
    return mask, migrate_metadata(metadata)
