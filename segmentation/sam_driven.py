"""
SAM-driven 3D segmentation: project 2D SAM masks to 3D Gaussian points via multi-view voting.
Requires: extract_segment_everything_masks.py + get_scale.py to be run first.
"""

import os
import json
import torch
import numpy as np
from argparse import Namespace
from tqdm import tqdm

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from .utils import nearest_visible_points
from .instance_graph import (
    associate_mask_nodes,
    build_anchor_graph,
    make_mask_nodes,
    mask_context_features,
    refine_boundary_gaussians,
    solve_anchor_instances,
)


def _expand_selected_instances(
    xyz, features, point_instances, voxel_size, minimum_similarity=0.78,
    spatial_margin_voxels=2.0, max_growth_factor=25.0,
    max_instance_fraction=0.05,
):
    """Complete selected object surfaces with local SAGA feature evidence.

    Multi-view projection supplies high-precision visible seeds.  Expansion is
    restricted to each seed component's padded 3D bounds and is capped by both
    seed growth and scene fraction, preventing a feature match from flooding the
    countertop or surrounding cabinetry.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    labels = np.asarray(point_instances, dtype=np.int64).copy()
    count = len(labels)
    maximum_size = max(10, int(float(max_instance_fraction) * count))
    valid_instances = []
    for instance in sorted(np.unique(labels[labels >= 0]).tolist()):
        size = int(np.count_nonzero(labels == instance))
        if size > maximum_size:
            labels[labels == instance] = -1
        elif size >= 10:
            valid_instances.append(int(instance))
    if not valid_instances:
        return labels, 0

    unlabelled = labels < 0
    best_score = np.full(count, -np.inf, dtype=np.float32)
    best_label = np.full(count, -1, dtype=np.int64)
    added = 0
    margin = max(float(voxel_size) * float(spatial_margin_voxels), 1e-6)
    for instance in valid_instances:
        seeds = np.flatnonzero(labels == instance)
        prototype = features[seeds].mean(0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
        seed_similarity = features[seeds] @ prototype
        threshold = max(
            float(minimum_similarity),
            float(np.quantile(seed_similarity, 0.10)) - 0.05,
        )
        lower = xyz[seeds].min(0) - margin
        upper = xyz[seeds].max(0) + margin
        spatial = np.all((xyz >= lower) & (xyz <= upper), axis=1)
        candidates = np.flatnonzero(unlabelled & spatial)
        if candidates.size == 0:
            continue
        scores = features[candidates] @ prototype
        accepted = candidates[scores >= threshold]
        accepted_scores = scores[scores >= threshold]
        growth_limit = min(
            maximum_size - len(seeds),
            max(0, int(len(seeds) * float(max_growth_factor)) - len(seeds)),
        )
        if growth_limit <= 0:
            continue
        if accepted.size > growth_limit:
            keep = np.argpartition(accepted_scores, -growth_limit)[-growth_limit:]
            accepted, accepted_scores = accepted[keep], accepted_scores[keep]
        improve = accepted_scores > best_score[accepted]
        best_score[accepted[improve]] = accepted_scores[improve]
        best_label[accepted[improve]] = instance
    chosen = best_label >= 0
    labels[chosen] = best_label[chosen]
    added = int(np.count_nonzero(chosen))
    return labels, added


def _direct_selected_point_instances(
    nodes, roots, point_count, min_views=1, visible_views=None,
    minimum_visible_support_fraction=0.0,
):
    """Assign selected points supported by ``min_views`` calibrated images.

    Requiring votes per Gaussian, rather than only per instance root, rejects
    background centres that happen to fall inside one silhouette. Those
    one-view false positives otherwise create very large splat footprints on
    nearby surfaces when the selected object is rendered from a new angle.
    """
    root_nodes = {}
    for node, root in zip(nodes, roots):
        root_nodes.setdefault(int(root), []).append(node)
    labels = np.full(int(point_count), -1, dtype=np.int64)
    best_votes = np.zeros(int(point_count), dtype=np.int16)
    next_instance = 0
    for root, members in sorted(root_nodes.items()):
        if len({node.view for node in members}) < int(min_views):
            continue
        all_points = np.concatenate([node.points for node in members])
        votes = np.bincount(all_points, minlength=point_count)
        required = np.full(int(point_count), int(min_views), dtype=np.int64)
        if visible_views is not None and float(minimum_visible_support_fraction) > 0:
            visible = np.asarray(visible_views, dtype=np.int64)
            if visible.shape != (int(point_count),):
                raise ValueError("visible_views must match point_count")
            fraction_required = np.ceil(
                visible * float(minimum_visible_support_fraction)).astype(np.int64)
            required = np.maximum(required, fraction_required)
        improve = (votes >= required) & (votes > best_votes)
        labels[improve] = next_instance
        best_votes[improve] = votes[improve].astype(np.int16)
        next_instance += 1
    return labels


def _apply_selection_correspondence(nodes, roots, selection_mode):
    """Apply correspondence explicitly declared by a selection manifest.

    In ``single_object`` mode every selected proposal is a calibrated
    observation of the same physical object. Requiring the generic graph to
    rediscover that fact can reject a correct selection when different views
    expose disjoint reconstructed surfaces.
    """
    mode = str(selection_mode or "instance_graph").strip().lower()
    if mode == "instance_graph":
        return roots
    if mode == "single_object":
        if len({int(node.view) for node in nodes}) < 2:
            raise ValueError("single_object selection needs at least two usable views")
        return np.zeros(len(nodes), dtype=np.int64)
    raise ValueError(f"unsupported selection mode: {selection_mode}")


def _load_cfg_args(model_path):
    """Load config from cfg_args saved during training."""
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = eval(f.read())
    return cfg


def _project_points_to_image(xyz, full_proj_transform, width, height, return_depth=False):
    """
    Project 3D points (N, 3) to 2D pixel coordinates.
    full_proj_transform: 4x4 world-to-clip matrix.
    Returns: (u, v) in [0, width-1] x [0, height-1], and valid mask (in front of camera).
    """
    device = xyz.device
    if isinstance(full_proj_transform, np.ndarray):
        proj = torch.from_numpy(full_proj_transform).float().to(device)
    else:
        proj = full_proj_transform.float().to(device)

    N = xyz.shape[0]
    ones = torch.ones(N, 1, device=device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)  # (N, 4)
    # Camera.full_proj_transform follows the CUDA rasterizer's transposed,
    # row-vector convention (world_view @ projection). Multiplying it as a
    # column-vector matrix silently projects into the wrong photographs for
    # non-identity cameras.
    clip = xyz_h @ proj  # (N, 4)
    raw_w = clip[:, 3]
    w = raw_w.unsqueeze(1).clamp(min=1e-6)
    x_ndc = clip[:, 0:1] / w
    y_ndc = clip[:, 1:2] / w
    z_ndc = clip[:, 2:3] / w

    # NDC to pixel: x_ndc in [-1,1] -> u in [0, width-1]
    u = (x_ndc.squeeze(-1) + 1.0) * 0.5 * (width - 1)
    # The CUDA rasterizer's ndc2Pix applies the same +1 mapping on both axes;
    # its projection matrix already carries the camera/image orientation.
    v = (y_ndc.squeeze(-1) + 1.0) * 0.5 * (height - 1)

    # Valid: in front of camera (z_ndc in [0,1] for typical projection) and within image
    valid = ((raw_w > 1e-6) & (z_ndc.squeeze(-1) > 0)
             & (u >= 0) & (u < width) & (v >= 0) & (v < height))
    result = (u.long().clamp(0, width - 1),
              v.long().clamp(0, height - 1), valid)
    return (*result, raw_w) if return_depth else result


def _mask_assignments_for_view(xyz, cam, masks, mask_chunk=32,
                               visibility_tolerance=0.01):
    """Return the smallest SAM mask per visible Gaussian and visible point ids.

    SAM's automatic masks overlap at multiple scales.  A projected point is
    assigned to the smallest containing mask, which prevents one view from
    casting several votes for the same point.
    """
    height, width = int(cam.image_height), int(cam.image_width)
    u, v, valid, depth = _project_points_to_image(
        xyz, cam.full_proj_transform.to(xyz.device), width, height,
        return_depth=True,
    )
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_ids.numel() == 0 or masks.shape[0] == 0:
        return np.full(xyz.shape[0], -1, dtype=np.int64), np.empty(0, dtype=np.int64)

    pixels = v[valid_ids] * width + u[valid_ids]
    front = nearest_visible_points(
        pixels, depth[valid_ids], width * height,
        relative_tolerance=float(visibility_tolerance))
    visible_ids = valid_ids[front]
    if visible_ids.numel() == 0:
        return np.full(xyz.shape[0], -1, dtype=np.int64), np.empty(0, dtype=np.int64)

    masks = masks.to(device=xyz.device, dtype=torch.bool)
    areas = masks.flatten(1).sum(1)
    best_area = torch.full((visible_ids.numel(),), torch.iinfo(torch.int64).max,
                           device=xyz.device, dtype=torch.int64)
    best_mask = torch.full((visible_ids.numel(),), -1, device=xyz.device, dtype=torch.long)
    query_v, query_u = v[visible_ids], u[visible_ids]
    for start in range(0, masks.shape[0], mask_chunk):
        stop = min(start + mask_chunk, masks.shape[0])
        membership = masks[start:stop, query_v, query_u]
        candidate_area = torch.where(
            membership,
            areas[start:stop, None],
            torch.full_like(membership, torch.iinfo(torch.int64).max, dtype=torch.int64),
        )
        chunk_area, chunk_index = candidate_area.min(dim=0)
        replace = chunk_area < best_area
        best_area[replace] = chunk_area[replace]
        best_mask[replace] = chunk_index[replace] + start

    assignments = np.full(xyz.shape[0], -1, dtype=np.int64)
    visible_numpy = visible_ids.detach().cpu().numpy()
    assignments[visible_numpy] = best_mask.detach().cpu().numpy()
    return assignments, visible_numpy


@torch.no_grad()
def run_sam_driven_segment(
    model_path, scene_model, feat_model, min_votes=2, sample_rate=1.0,
    target_anchors=1600, max_anchor_points=128, graph_threshold=0.50,
    diagnostics_path=None, source_path=None, masks_dir=None, selected_masks=None,
    min_mask_area_fraction=0.0, max_mask_area_fraction=1.0,
    max_instance_fraction=1.0,
    feature_expand=False, feature_similarity=0.78,
    spatial_margin_voxels=2.0, max_growth_factor=25.0,
    visibility_tolerance=0.01,
    minimum_visible_support_fraction=0.0,
    selection_mode="instance_graph",
):
    """
    Project SAM masks to a cross-view anchor graph and assign 3D instances.

    Args:
        model_path: Path to trained model (contains cfg_args, source_path from training).
        scene_model: GaussianModel (will be segmented in place).
        feat_model: FeatureGaussianModel (will be segmented in place).
        min_votes: Minimum number of views that must agree for a point to be assigned to a mask.
        sample_rate: Camera sampling (1.0 = use all, 0.5 = use half).
        target_anchors: Desired spatial graph scale before dense-anchor splitting.
        max_anchor_points: Hard limit on the number of Gaussians in one anchor.
        graph_threshold: Minimum five-signal affinity for cross-view node association.
        diagnostics_path: Optional JSON path for graph/run statistics.

    Returns:
        Number of segments created, or -1 on error.
    """
    cfg = _load_cfg_args(model_path)
    if cfg is None:
        print("SAM-driven: cfg_args not found in", model_path)
        return -1

    source_path = source_path or getattr(cfg, "source_path", None)
    if not source_path or not os.path.isdir(source_path):
        print("SAM-driven: source_path not found:", source_path)
        return -1

    sam_masks_dir = masks_dir or os.path.join(source_path, "sam_masks")
    mask_scales_dir = os.path.join(source_path, "mask_scales")
    if not os.path.isdir(sam_masks_dir):
        print("SAM-driven: sam_masks not found. Run extract_segment_everything_masks.py first.")
        return -1
    if masks_dir is None and not os.path.isdir(mask_scales_dir):
        print("SAM-driven: mask_scales not found. Run get_scale.py first.")
        return -1

    # Build args for camera loading (from cfg_args)
    args = Namespace(**vars(cfg))
    args.source_path = source_path
    args.model_path = model_path
    # Explicit multi-view selections load their mask tensors below. This avoids
    # loading masks for every training camera and lets a compact selection
    # directory or manifest drive the run.
    args.need_masks = masks_dir is None and selected_masks is None
    args.need_features = False
    args.resolution = getattr(args, "resolution", -1)
    args.data_device = getattr(args, "data_device", "cuda")
    args.images = getattr(args, "images", "images")
    args.allow_principle_point_shift = getattr(args, "allow_principle_point_shift", False)
    args.eval = getattr(args, "eval", False)

    requested_views = {
        os.path.splitext(str(name))[0] for name in (selected_masks or {})
    }
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            source_path, args.images, args.eval,
            need_features=False, need_masks=args.need_masks,
            sample_rate=sample_rate,
            allow_principle_point_shift=args.allow_principle_point_shift,
            replica="replica" in model_path,
            view_names=requested_views or None,
        )
    else:
        print("SAM-driven: Colmap sparse/ not found. Only Colmap format is supported.")
        return -1

    selected_masks = {
        os.path.splitext(str(name))[0]: indices
        for name, indices in (selected_masks or {}).items()
    }
    if selected_masks:
        wanted = set(selected_masks)
        selected_camera_infos = [
            camera for camera in scene_info.train_cameras
            if os.path.splitext(str(camera.image_name))[0] in wanted
        ]
        missing = wanted - {
            os.path.splitext(str(camera.image_name))[0]
            for camera in selected_camera_infos
        }
        if missing:
            print("SAM-driven: selected cameras were not found:", sorted(missing))
            return -1
        scene_info = scene_info._replace(
            train_cameras=selected_camera_infos, test_cameras=[])

    resolution_scale = 1.0
    try:
        cameras = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
    except Exception as e:
        print("SAM-driven: Failed to load cameras:", e)
        return -1

    xyz = scene_model.get_xyz  # (N, 3)
    N = xyz.shape[0]
    point_features = feat_model.get_point_features.squeeze()
    if point_features.ndim == 1:
        point_features = point_features[:, None]
    if point_features.shape[0] != N:
        print(f"SAM-driven: scene/feature point counts differ ({N} vs {point_features.shape[0]}).")
        return -1
    graph = build_anchor_graph(
        xyz.detach().cpu().numpy(), point_features.detach().cpu().numpy(),
        target_anchors=target_anchors, max_anchor_points=max_anchor_points,
    )
    nodes = []
    usable_views = 0
    selected_visible_views = (
        np.zeros(N, dtype=np.int16) if selected_masks else None)

    for cam in tqdm(cameras, desc="SAM-driven: projecting views"):
        stem = os.path.splitext(str(cam.image_name))[0]
        masks = cam.original_masks
        if masks is None:
            mask_path = os.path.join(sam_masks_dir, stem + ".pt")
            if not os.path.isfile(mask_path):
                continue
            masks = torch.load(mask_path, map_location="cpu", weights_only=False)
        # A selection manifest stores the proposal indices clicked/accepted in
        # each photo. An empty list deliberately excludes that view.
        explicit_selection_union = False
        if selected_masks:
            indices = selected_masks.get(stem, [])
            if not indices:
                continue
            valid_indices = [int(index) for index in indices
                             if 0 <= int(index) < int(masks.shape[0])]
            if not valid_indices:
                continue
            chosen = masks[valid_indices]
            if chosen.dim() == 2:
                chosen = chosen.unsqueeze(0)
            # A painted selection may contain several SAM parts (body, lid,
            # spout, handle). Filter each proposal by area, then union the
            # parts so one photograph can never manufacture multiple votes.
            chosen_areas = chosen.flatten(1).sum(1).float()
            chosen_pixels = float(chosen.shape[-2] * chosen.shape[-1])
            chosen_keep = (
                (chosen_areas >= chosen_pixels * float(min_mask_area_fraction))
                & (chosen_areas <= chosen_pixels * float(max_mask_area_fraction)))
            chosen = chosen[chosen_keep]
            if chosen.shape[0] == 0:
                continue
            masks = torch.any(chosen.bool(), dim=0, keepdim=True)
            explicit_selection_union = True
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        if not explicit_selection_union:
            areas = masks.flatten(1).sum(1).float()
            total_pixels = float(masks.shape[-2] * masks.shape[-1])
            keep = ((areas >= total_pixels * float(min_mask_area_fraction))
                    & (areas <= total_pixels * float(max_mask_area_fraction)))
            masks = masks[keep]
        if masks.shape[0] == 0:
            continue
        H, W = masks.shape[1], masks.shape[2]
        cam_h, cam_w = cam.image_height, cam.image_width

        # Resize masks to camera resolution if needed
        if H != cam_h or W != cam_w:
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(),
                size=(cam_h, cam_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            masks = (masks >= 0.5)

        assignments, visible = _mask_assignments_for_view(
            xyz, cam, masks, visibility_tolerance=visibility_tolerance)
        if visible.size == 0:
            continue
        if selected_visible_views is not None:
            selected_visible_views[visible] += 1
        if not np.any(assignments >= 0):
            continue
        contexts = mask_context_features(cam.original_image, masks)
        view_nodes = make_mask_nodes(
            graph, assignments, visible, contexts, usable_views,
        )
        if view_nodes:
            nodes.extend(view_nodes)
            usable_views += 1

    if not nodes:
        print("SAM-driven: No usable mask-to-anchor observations.")
        return 0
    roots, evidence = associate_mask_nodes(nodes, graph, score_threshold=graph_threshold)
    if selected_masks:
        roots = _apply_selection_correspondence(nodes, roots, selection_mode)
    anchor_instances, unary, root_to_instance = solve_anchor_instances(
        nodes, roots, graph, min_views=min_votes,
    )
    point_instances, boundary = refine_boundary_gaussians(
        graph, anchor_instances, point_features.detach().cpu().numpy(),
        nodes, roots, root_to_instance,
    )
    if selected_masks:
        # The graph is still used to associate proposals across photographs,
        # but an explicit user click must not disappear because its spatial
        # anchor also contains many unselected/background Gaussians.
        point_instances = _direct_selected_point_instances(
            nodes, roots, N, min_views=min_votes,
            visible_views=selected_visible_views,
            minimum_visible_support_fraction=minimum_visible_support_fraction)
    expanded_gaussians = 0
    if feature_expand:
        point_instances, expanded_gaussians = _expand_selected_instances(
            xyz.detach().cpu().numpy(), point_features.detach().cpu().numpy(),
            point_instances, graph.voxel_size,
            minimum_similarity=feature_similarity,
            spatial_margin_voxels=spatial_margin_voxels,
            max_growth_factor=max_growth_factor,
            max_instance_fraction=max_instance_fraction,
        )
    segments_to_apply = []
    kept_instances = []
    for instance in sorted(np.unique(point_instances[point_instances >= 0]).tolist()):
        point_ids = np.flatnonzero(point_instances == instance)
        if point_ids.size < 10:
            continue
        if point_ids.size > int(float(max_instance_fraction) * N):
            continue
        mask = torch.zeros(N, dtype=torch.bool, device=xyz.device)
        mask[torch.as_tensor(point_ids, device=xyz.device)] = True
        segments_to_apply.append(mask)
        kept_instances.append(int(instance))

    if not segments_to_apply:
        print("SAM-driven: No valid segments (all masks too small or no votes).")
        return 0

    # Clear existing and apply new segments
    try:
        scene_model.clear_segment()
        feat_model.clear_segment()
    except Exception:
        pass

    for seg_mask in segments_to_apply:
        scene_model.segment(seg_mask)
        feat_model.segment(seg_mask)

    diagnostics = {
        "version": "2.2",
        "gaussians": int(N),
        "anchors": int(graph.centroids.shape[0]),
        "anchor_edges": int(graph.edges.shape[0]),
        "voxel_size": float(graph.voxel_size),
        "target_anchors": int(target_anchors),
        "max_anchor_points": int(max_anchor_points),
        "graph_threshold": float(graph_threshold),
        "source_path": os.path.abspath(source_path),
        "masks_dir": os.path.abspath(sam_masks_dir),
        "selected_views": sorted(selected_masks) if selected_masks else [],
        "min_mask_area_fraction": float(min_mask_area_fraction),
        "max_mask_area_fraction": float(max_mask_area_fraction),
        "max_instance_fraction": float(max_instance_fraction),
        "visibility_tolerance": float(visibility_tolerance),
        "minimum_visible_support_fraction": float(
            minimum_visible_support_fraction),
        "selection_mode": str(selection_mode),
        "instance_gaussian_counts": [int(mask.sum().item()) for mask in segments_to_apply],
        "feature_expand": bool(feature_expand),
        "feature_similarity": float(feature_similarity),
        "expanded_gaussians": int(expanded_gaussians),
        "usable_views": int(usable_views),
        "mask_nodes": int(len(nodes)),
        "association_edges": int(len(evidence)),
        "accepted_associations": int(sum(item["score"] >= graph_threshold for item in evidence)),
        "instances": int(len(segments_to_apply)),
        "boundary_anchors": int(boundary.sum()),
        "boundary_gaussians": int(boundary[graph.point_to_anchor].sum()),
        "covered_gaussians": int(np.count_nonzero(point_instances >= 0)),
    }
    run_sam_driven_segment.last_diagnostics = diagnostics
    run_sam_driven_segment.last_point_instances = point_instances
    run_sam_driven_segment.last_anchor_graph = graph
    if diagnostics_path:
        os.makedirs(os.path.dirname(os.path.abspath(diagnostics_path)), exist_ok=True)
        with open(diagnostics_path, "w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2)
    print(f"SAM-driven V2.2: Created {len(segments_to_apply)} graph instances "
          f"from {usable_views} views and {graph.centroids.shape[0]} anchors; "
          f"refined {diagnostics['boundary_gaussians']} boundary Gaussians.")
    return len(segments_to_apply)


def run_multiview_selection(manifest_path, model_path, scene_model, feat_model,
                            diagnostics_path=None):
    """Fuse the SAM proposals selected in several calibrated photographs."""
    from .multiview_selection import load_selection_manifest
    payload, source_path, masks_dir, selections = load_selection_manifest(manifest_path)
    return run_sam_driven_segment(
        model_path, scene_model, feat_model,
        min_votes=int(payload.get("min_votes", 2)),
        sample_rate=1.0,
        target_anchors=int(payload.get("target_anchors", 1600)),
        max_anchor_points=int(payload.get("max_anchor_points", 128)),
        graph_threshold=float(payload.get("graph_threshold", 0.50)),
        diagnostics_path=diagnostics_path,
        source_path=str(source_path), masks_dir=str(masks_dir),
        selected_masks=selections,
        min_mask_area_fraction=float(payload.get("min_mask_area_fraction", 0.0)),
        max_mask_area_fraction=float(payload.get("max_mask_area_fraction", 1.0)),
        max_instance_fraction=float(payload.get("max_instance_fraction", 0.50)),
        feature_expand=bool(payload.get("feature_expand", False)),
        feature_similarity=float(payload.get("feature_similarity", 0.78)),
        spatial_margin_voxels=float(payload.get("spatial_margin_voxels", 2.0)),
        max_growth_factor=float(payload.get("max_growth_factor", 25.0)),
        visibility_tolerance=float(payload.get("visibility_tolerance", 0.01)),
        minimum_visible_support_fraction=float(
            payload.get("minimum_visible_support_fraction", 0.50)),
        selection_mode=str(payload.get("selection_mode", "instance_graph")),
    )


def main():
    """Standalone: load models, run SAM-driven segmentation, save mask to file."""
    import argparse
    from scene import GaussianModel, FeatureGaussianModel

    parser = argparse.ArgumentParser(description="SAM-driven segmentation (standalone)")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-f", "--feature_iteration", type=int, default=10000)
    parser.add_argument("-s", "--scene_iteration", type=int, default=30000)
    parser.add_argument("-o", "--output", type=str, default="./segmentation_res/sam_driven_mask.pt")
    parser.add_argument("--min_votes", type=int, default=2)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--target_anchors", type=int, default=1600)
    parser.add_argument("--max_anchor_points", type=int, default=128)
    parser.add_argument("--graph_threshold", type=float, default=0.50)
    parser.add_argument("--diagnostics", type=str, default=None)
    parser.add_argument("--selection", type=str, default=None,
                        help="multi-view photo/SAM selection manifest JSON")
    args = parser.parse_args()

    scene_ply = os.path.join(args.model_path, f"point_cloud/iteration_{args.scene_iteration}/scene_point_cloud.ply")
    feature_ply = os.path.join(args.model_path, f"point_cloud/iteration_{args.feature_iteration}/contrastive_feature_point_cloud.ply")
    if not os.path.isfile(scene_ply):
        print(f"Error: Scene PLY not found: {scene_ply}")
        return
    if not os.path.isfile(feature_ply):
        print(f"Error: Feature PLY not found: {feature_ply}")
        return

    print("Loading models...")
    gs_model = GaussianModel(sh_degree=3)
    feat_model = FeatureGaussianModel(feature_dim=32)
    gs_model.load_ply(scene_ply)
    feat_model.load_ply(feature_ply)
    print("Models loaded.")

    if args.selection:
        n = run_multiview_selection(
            args.selection, args.model_path, gs_model, feat_model,
            diagnostics_path=args.diagnostics)
    else:
        n = run_sam_driven_segment(
            args.model_path,
            gs_model,
            feat_model,
            min_votes=args.min_votes,
            sample_rate=args.sample_rate,
            target_anchors=args.target_anchors,
            max_anchor_points=args.max_anchor_points,
            graph_threshold=args.graph_threshold,
            diagnostics_path=args.diagnostics,
        )
    if n < 0:
        return

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mask = gs_model._mask.cpu()
    torch.save(mask, args.output)
    print(f"Saved segmentation mask to {args.output}")
    print("Run rt_gs_gui.py and click 'Load segmentation' to load this mask.")


if __name__ == "__main__":
    main()
