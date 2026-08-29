"""Safe loading of binary labels and confidence-aware Gaussian membership."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class GaussianMembership:
    """Per-Gaussian discrete labels plus optional soft object membership."""

    labels: np.ndarray
    values: np.ndarray
    selected: np.ndarray
    mode: str
    confidence_path: Path | None = None


def paired_confidence_path(mask_path) -> Path:
    """Return the conventional confidence artifact paired with ``*_mask.pt``."""
    source = Path(mask_path).expanduser()
    suffix = "_mask.pt"
    if source.name.endswith(suffix):
        return source.with_name(source.name[:-len(suffix)] + "_confidence.npz")
    return source.with_name(source.stem + "_confidence.npz")


def _load_labels(mask_path) -> np.ndarray:
    try:
        loaded = torch.load(mask_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        loaded = torch.load(mask_path, map_location="cpu")
    labels = np.asarray(loaded.detach().cpu() if torch.is_tensor(loaded) else loaded)
    if labels.ndim != 1:
        raise ValueError(f"segmentation labels must be one-dimensional, got {labels.shape}")
    if labels.dtype == np.bool_:
        return np.where(labels, 2, 1).astype(np.int64)
    if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
        raise ValueError("segmentation labels must contain finite numeric values")
    if not np.equal(labels, np.round(labels)).all():
        raise ValueError("segmentation labels must be integers")
    return labels.astype(np.int64, copy=False)


def load_gaussian_membership(mask_path, mode="binary", confidence_path=None,
                             expected_count=None) -> GaussianMembership:
    """Load a saved segmentation without enabling NumPy pickle support.

    ``binary`` preserves the historical convention that labels greater than one
    are selected. ``soft`` loads the paired research artifact's ``confidence``
    vector while retaining the labels for colouring and discrete extraction.
    """
    mode = str(mode).lower()
    if mode not in {"binary", "soft"}:
        raise ValueError("membership mode must be 'binary' or 'soft'")
    labels = _load_labels(mask_path)
    if expected_count is not None and labels.size != int(expected_count):
        raise ValueError(
            f"segmentation mask has {labels.size} values; scene has {int(expected_count)}")
    selected = labels > 1
    if mode == "binary":
        return GaussianMembership(
            labels=labels, values=selected.astype(np.float32), selected=selected,
            mode=mode)

    source = (Path(confidence_path).expanduser() if confidence_path is not None
              else paired_confidence_path(mask_path))
    if not source.is_file():
        raise FileNotFoundError(f"soft membership artifact not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if "confidence" not in archive.files:
            raise ValueError(f"soft membership artifact has no 'confidence' array: {source}")
        confidence = np.asarray(archive["confidence"], dtype=np.float32)
        artifact_selected = (
            np.asarray(archive["selected"], dtype=bool)
            if "selected" in archive.files else None)
    if confidence.ndim != 1 or confidence.shape != labels.shape:
        raise ValueError(
            f"confidence shape {confidence.shape} does not match labels {labels.shape}")
    if not np.isfinite(confidence).all():
        raise ValueError("confidence must contain only finite values")
    if bool(((confidence < 0.0) | (confidence > 1.0)).any()):
        raise ValueError("confidence must lie in [0, 1]")
    if artifact_selected is not None:
        if artifact_selected.shape != selected.shape:
            raise ValueError("confidence artifact selected array has the wrong shape")
        if not np.array_equal(artifact_selected, selected):
            raise ValueError("confidence artifact and segmentation labels select different Gaussians")
    # Confidence outside the discrete extraction must never re-introduce a point.
    values = confidence.copy()
    values[~selected] = 0.0
    return GaussianMembership(
        labels=labels, values=values, selected=selected, mode=mode,
        confidence_path=source)
