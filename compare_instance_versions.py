"""Evaluate V2.1 and V2.2 instance masks on the same real COLMAP/SAM scene."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from plyfile import PlyData
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment

from instance_graph import build_anchor_graph
from sam_driven_segment import _load_cfg_args, _mask_assignments_for_view, _project_points_to_image
from scene.dataset_readers import sceneLoadTypeCallbacks
from segmentation_utils import nearest_visible_points
from utils.camera_utils import cameraList_from_camInfos


def _read_ply(path, features=False):
    vertices = PlyData.read(path)["vertex"]
    xyz = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1).astype(np.float32)
    if not features:
        return xyz
    names = sorted((item.name for item in vertices.properties if item.name.startswith("f_")),
                   key=lambda value: int(value.split("_")[-1]))
    return xyz, np.stack([vertices[name] for name in names], axis=1).astype(np.float32)


def _load_cameras(model_path, sample_rate):
    cfg = _load_cfg_args(model_path)
    args = Namespace(**vars(cfg))
    args.need_masks = True
    args.need_features = False
    args.resolution = getattr(args, "resolution", -1)
    args.data_device = getattr(args, "data_device", "cuda")
    args.images = getattr(args, "images", "images")
    args.allow_principle_point_shift = getattr(args, "allow_principle_point_shift", False)
    args.eval = getattr(args, "eval", False)
    info = sceneLoadTypeCallbacks["Colmap"](
        args.source_path, args.images, args.eval, need_features=False, need_masks=True,
        sample_rate=sample_rate,
        allow_principle_point_shift=args.allow_principle_point_shift,
        replica="replica" in model_path,
    )
    return cameraList_from_camInfos(info.train_cameras, 1.0, args)


def _labels(path):
    values = torch.load(path, map_location="cpu").numpy().astype(np.int64)
    return np.where(values > 1, values - 2, -1)


def _match_labels(reference, candidate):
    ref_ids = np.unique(reference[reference >= 0])
    cand_ids = np.unique(candidate[candidate >= 0])
    overlap = np.zeros((len(ref_ids), len(cand_ids)), dtype=np.int64)
    for i, ref in enumerate(ref_ids):
        for j, cand in enumerate(cand_ids):
            overlap[i, j] = np.count_nonzero((reference == ref) & (candidate == cand))
    rows, cols = linear_sum_assignment(-overlap)
    return {int(cand_ids[col]): int(ref_ids[row]) for row, col in zip(rows, cols)}


def _graph_metrics(labels, features, graph):
    covered = labels >= 0
    unique = np.unique(labels[covered])
    coherence_sum = 0.0
    for label in unique:
        selected = features[labels == label]
        selected = selected / np.maximum(np.linalg.norm(selected, axis=1, keepdims=True), 1e-8)
        prototype = selected.mean(0)
        prototype /= max(np.linalg.norm(prototype), 1e-8)
        coherence_sum += float((selected @ prototype).sum())

    anchor_labels = np.full(len(graph.centroids), -1, dtype=np.int64)
    anchor_pure = 0
    for anchor in range(len(anchor_labels)):
        point_labels = labels[graph.point_to_anchor == anchor]
        point_labels = point_labels[point_labels >= 0]
        if point_labels.size:
            counts = np.bincount(point_labels)
            anchor_labels[anchor] = counts.argmax()
            anchor_pure += int(counts.max())
    adjacency = [[] for _ in anchor_labels]
    crossing = 0
    eligible_edges = 0
    for left, right in graph.edges:
        adjacency[left].append(int(right))
        adjacency[right].append(int(left))
        if anchor_labels[left] >= 0 and anchor_labels[right] >= 0:
            eligible_edges += 1
            crossing += int(anchor_labels[left] != anchor_labels[right])
    components = 0
    for label in unique:
        remaining = set(np.flatnonzero(anchor_labels == label).tolist())
        while remaining:
            components += 1
            queue = deque([remaining.pop()])
            while queue:
                current = queue.popleft()
                for other in adjacency[current]:
                    if other in remaining and anchor_labels[other] == label:
                        remaining.remove(other)
                        queue.append(other)
    return {
        "instances": int(len(unique)),
        "coverage": float(covered.mean()),
        "saga_coherence": coherence_sum / max(int(covered.sum()), 1),
        "anchor_purity": anchor_pure / max(int(covered.sum()), 1),
        "spatial_components_per_instance": components / max(len(unique), 1),
        "cross_instance_anchor_edges": crossing / max(eligible_edges, 1),
    }


def _sam_agreement(xyz, cameras, label_sets):
    matched = defaultdict(int)
    observed = defaultdict(int)
    for cam in cameras:
        masks = cam.original_masks
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.shape[-2:] != (cam.image_height, cam.image_width):
            masks = torch.nn.functional.interpolate(
                masks[:, None].float(), (cam.image_height, cam.image_width), mode="nearest"
            )[:, 0] > 0.5
        assignments, _ = _mask_assignments_for_view(xyz, cam, masks)
        for name, labels in label_sets.items():
            for local_mask in np.unique(assignments[assignments >= 0]):
                point_labels = labels[assignments == local_mask]
                point_labels = point_labels[point_labels >= 0]
                if point_labels.size:
                    counts = np.bincount(point_labels)
                    matched[name] += int(counts.max())
                    observed[name] += int(point_labels.size)
    return {name: matched[name] / max(observed[name], 1) for name in label_sets}


def _overlay(image, xyz, cam, labels, palette):
    rgb = image[:3].detach().cpu().permute(1, 2, 0).numpy()
    height, width = rgb.shape[:2]
    u, v, valid, depth = _project_points_to_image(
        xyz, cam.full_proj_transform.to(xyz.device), width, height, return_depth=True,
    )
    ids = torch.nonzero(valid, as_tuple=False).flatten()
    pixels = v[ids] * width + u[ids]
    front = nearest_visible_points(pixels, depth[ids], width * height)
    ids = ids[front].detach().cpu().numpy()
    uu, vv = u.detach().cpu().numpy(), v.detach().cpu().numpy()
    output = rgb.copy()
    for label in np.unique(labels[ids]):
        if label < 0:
            continue
        region = np.zeros((height, width), dtype=bool)
        selected = ids[labels[ids] == label]
        region[vv[selected], uu[selected]] = True
        region = binary_dilation(region, iterations=1)
        colour = palette[int(label) % len(palette)]
        output[region] = 0.38 * output[region] + 0.62 * colour
    return np.clip(output, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--scene_iteration", type=int, default=610)
    parser.add_argument("--feature_iteration", type=int, default=4)
    parser.add_argument("--v21_mask", required=True)
    parser.add_argument("--v22_mask", required=True)
    parser.add_argument("--sample_rate", type=float, default=0.2)
    parser.add_argument("--output_dir", default="comparisons/bicycle")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scene_ply = os.path.join(args.model_path, "point_cloud", f"iteration_{args.scene_iteration}", "scene_point_cloud.ply")
    feature_ply = os.path.join(args.model_path, "point_cloud", f"iteration_{args.feature_iteration}", "contrastive_feature_point_cloud.ply")
    xyz_np = _read_ply(scene_ply)
    _, features = _read_ply(feature_ply, features=True)
    xyz = torch.from_numpy(xyz_np).cuda()
    labels = {"V2.1": _labels(args.v21_mask), "V2.2": _labels(args.v22_mask)}
    graph = build_anchor_graph(xyz_np, features, target_anchors=1600, max_anchor_points=128)
    cameras = _load_cameras(args.model_path, args.sample_rate)
    metrics = {name: _graph_metrics(value, features, graph) for name, value in labels.items()}
    agreement = _sam_agreement(xyz, cameras, labels)
    for name in metrics:
        metrics[name]["sam_mask_agreement"] = agreement[name]

    mapping = _match_labels(labels["V2.1"], labels["V2.2"])
    display_v22 = np.array([mapping.get(int(value), int(value)) if value >= 0 else -1
                            for value in labels["V2.2"]], dtype=np.int64)
    palette = np.asarray(plt.get_cmap("tab10").colors, dtype=np.float32)
    selected_views = sorted(set((0, len(cameras) // 2, len(cameras) - 1)))
    fig, axes = plt.subplots(len(selected_views), 3, figsize=(13, 3.4 * len(selected_views)))
    if len(selected_views) == 1:
        axes = axes[None]
    for row, view_id in enumerate(selected_views):
        cam = cameras[view_id]
        rgb = cam.original_image[:3].detach().cpu().permute(1, 2, 0).numpy()
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(_overlay(cam.original_image, xyz, cam, labels["V2.1"], palette))
        axes[row, 2].imshow(_overlay(cam.original_image, xyz, cam, display_v22, palette))
        axes[row, 0].set_ylabel(cam.image_name)
        for col, title in enumerate(("Real RGB", "V2.1 point voting", "V2.2 anchor graph")):
            axes[row, col].set_title(title)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.tight_layout()
    figure_path = os.path.join(args.output_dir, "real_bicycle_comparison.png")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report = {
        "dataset": os.path.basename(os.path.normpath(args.model_path)),
        "sample_rate": args.sample_rate,
        "views": len(cameras),
        "gaussians": len(xyz_np),
        "metrics": metrics,
        "label_mapping_v22_to_v21": mapping,
        "figure": os.path.relpath(figure_path),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
