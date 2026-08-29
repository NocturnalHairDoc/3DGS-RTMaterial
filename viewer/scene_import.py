"""Resolve trained 3DGS/SAGA scene assets without fixed iteration numbers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch
import torch.nn.functional as F


_ITERATION_RE = re.compile(r"^iteration_(\d+)$")
_SCENE_NAMES = ("scene_point_cloud.ply", "point_cloud.ply")
_FEATURE_NAMES = ("contrastive_feature_point_cloud.ply", "feature_point_cloud.ply")


@dataclass(frozen=True)
class SceneAssets:
    model_path: Path
    scene_ply: Path
    scene_iteration: int | None
    feature_ply: Path | None = None
    feature_iteration: int | None = None
    scale_gate: Path | None = None

    @property
    def has_semantic_features(self) -> bool:
        return self.feature_ply is not None and self.scale_gate is not None


def _iteration(path: Path) -> int | None:
    match = _ITERATION_RE.match(path.parent.name)
    return int(match.group(1)) if match else None


def _ranked_files(point_cloud_root: Path, names: tuple[str, ...]):
    ranked = []
    if point_cloud_root.is_dir():
        for name_priority, name in enumerate(names):
            direct = point_cloud_root / name
            if direct.is_file():
                ranked.append((-1, -name_priority, direct))
            for candidate in point_cloud_root.glob(f"iteration_*/{name}"):
                iteration = _iteration(candidate)
                if iteration is not None:
                    ranked.append((iteration, -name_priority, candidate))
    return sorted(ranked, reverse=True, key=lambda item: (item[0], item[1]))


def _select(candidates, requested_iteration, label):
    if requested_iteration is None or int(requested_iteration) < 0:
        return candidates[0][2] if candidates else None
    requested_iteration = int(requested_iteration)
    for iteration, _, path in candidates:
        if iteration == requested_iteration:
            return path
    raise FileNotFoundError(f"{label} iteration {requested_iteration} was not found")


def resolve_scene_assets(path, scene_iteration=None, feature_iteration=None) -> SceneAssets:
    """Resolve a model directory, ``point_cloud`` directory, or scene PLY.

    Standard 3DGS ``point_cloud.ply`` scenes are accepted without SAGA assets.
    In that case the viewer creates deterministic geometry/appearance features
    so automatic segmentation and PBR editing remain available.
    """
    source = Path(path).expanduser().resolve()
    explicit_scene = source if source.is_file() else None
    if explicit_scene is not None and explicit_scene.suffix.lower() != ".ply":
        raise ValueError(f"scene file must be a PLY: {explicit_scene}")

    if explicit_scene is not None:
        point_cloud_root = (explicit_scene.parent.parent
                            if _ITERATION_RE.match(explicit_scene.parent.name)
                            else explicit_scene.parent)
        model_path = point_cloud_root.parent if point_cloud_root.name == "point_cloud" else point_cloud_root
        scene_ply = explicit_scene
    else:
        if not source.is_dir():
            raise FileNotFoundError(f"scene path does not exist: {source}")
        point_cloud_root = source if source.name == "point_cloud" else source / "point_cloud"
        model_path = source.parent if source.name == "point_cloud" else source
        scene_ply = _select(
            _ranked_files(point_cloud_root, _SCENE_NAMES), scene_iteration, "scene")
        if scene_ply is None:
            raise FileNotFoundError(
                f"no trained scene PLY found under {point_cloud_root}; expected "
                "iteration_*/scene_point_cloud.ply or iteration_*/point_cloud.ply"
            )

    # Select only complete feature/gate pairs. An interrupted newer feature
    # export must not hide an older usable SAGA iteration.
    feature_candidates = [
        item for item in _ranked_files(point_cloud_root, _FEATURE_NAMES)
        if (item[2].parent / "scale_gate.pt").is_file()
    ]
    feature_ply = _select(feature_candidates, feature_iteration, "feature")
    feature_iter = _iteration(feature_ply) if feature_ply else None
    gate = feature_ply.parent / "scale_gate.pt" if feature_ply else None

    return SceneAssets(
        model_path=model_path,
        scene_ply=scene_ply,
        scene_iteration=_iteration(scene_ply),
        feature_ply=feature_ply,
        feature_iteration=feature_iter,
        scale_gate=gate,
    )


def discover_scenes(root) -> list[SceneAssets]:
    """Return every valid trained scene immediately below ``root``."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return []
    result = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        try:
            result.append(resolve_scene_assets(candidate))
        except (FileNotFoundError, ValueError):
            continue
    return result


def build_proxy_features(scene, feature_dim: int = 32) -> torch.Tensor:
    """Build deterministic geometry/appearance features for plain 3DGS scenes."""
    xyz = scene.get_xyz.detach().float()
    centered = xyz - xyz.mean(0, keepdim=True)
    xyz_feature = (
        centered / centered.std(0, keepdim=True, unbiased=False).clamp_min(1e-5)
    ).clamp(-4, 4) / 4
    dc = scene._features_dc.detach().float()
    if dc.ndim == 3:
        dc = dc[:, 0, :]
    rgb = (0.5 + 0.28209479177387814 * dc).clamp(0, 1)
    opacity = scene.get_opacity.detach().float().clamp(0, 1)
    components = [xyz_feature, rgb, opacity]
    for frequency in (1.0, 2.0, 4.0):
        phase = xyz_feature * (torch.pi * frequency)
        components.extend((torch.sin(phase), torch.cos(phase)))
    components.extend((xyz_feature.square(), torch.ones_like(opacity)))
    features = torch.cat(components, dim=1)
    if features.shape[1] < feature_dim:
        features = F.pad(features, (0, feature_dim - features.shape[1]))
    elif features.shape[1] > feature_dim:
        features = features[:, :feature_dim]
    return F.normalize(features, dim=1, eps=1e-6)
