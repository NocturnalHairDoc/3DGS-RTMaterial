"""
SAM-driven 3D segmentation: project 2D SAM masks to 3D Gaussian points via multi-view voting.
Requires: extract_segment_everything_masks.py + get_scale.py to be run first.
"""

import os
import torch
import numpy as np
from collections import defaultdict
from argparse import Namespace
from tqdm import tqdm

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


def _load_cfg_args(model_path):
    """Load config from cfg_args saved during training."""
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = eval(f.read())
    return cfg


def _project_points_to_image(xyz, full_proj_transform, width, height):
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
    # clip = proj @ xyz_h.T -> (4,4) @ (4,N) = (4,N), then .T -> (N,4)
    clip = (proj @ xyz_h.T).T  # (N, 4)
    w = clip[:, 3:4].clamp(min=1e-6)
    x_ndc = clip[:, 0:1] / w
    y_ndc = clip[:, 1:2] / w
    z_ndc = clip[:, 2:3] / w

    # NDC to pixel: x_ndc in [-1,1] -> u in [0, width-1]
    u = (x_ndc.squeeze(-1) + 1.0) * 0.5 * (width - 1)
    v = (1.0 - y_ndc.squeeze(-1)) * 0.5 * (height - 1)  # y flipped for image coords

    # Valid: in front of camera (z_ndc in [0,1] for typical projection) and within image
    valid = (z_ndc.squeeze(-1) > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u.long().clamp(0, width - 1), v.long().clamp(0, height - 1), valid


@torch.no_grad()
def run_sam_driven_segment(model_path, scene_model, feat_model, min_votes=1, sample_rate=1.0):
    """
    Run SAM-driven segmentation: project 2D SAM masks to 3D, vote across views, assign segments.

    Args:
        model_path: Path to trained model (contains cfg_args, source_path from training).
        scene_model: GaussianModel (will be segmented in place).
        feat_model: FeatureGaussianModel (will be segmented in place).
        min_votes: Minimum number of views that must agree for a point to be assigned to a mask.
        sample_rate: Camera sampling (1.0 = use all, 0.5 = use half).

    Returns:
        Number of segments created, or -1 on error.
    """
    cfg = _load_cfg_args(model_path)
    if cfg is None:
        print("SAM-driven: cfg_args not found in", model_path)
        return -1

    source_path = getattr(cfg, "source_path", None)
    if not source_path or not os.path.isdir(source_path):
        print("SAM-driven: source_path not found:", source_path)
        return -1

    sam_masks_dir = os.path.join(source_path, "sam_masks")
    mask_scales_dir = os.path.join(source_path, "mask_scales")
    if not os.path.isdir(sam_masks_dir):
        print("SAM-driven: sam_masks not found. Run extract_segment_everything_masks.py first.")
        return -1
    if not os.path.isdir(mask_scales_dir):
        print("SAM-driven: mask_scales not found. Run get_scale.py first.")
        return -1

    # Build args for camera loading (from cfg_args)
    args = Namespace(**vars(cfg))
    args.source_path = source_path
    args.model_path = model_path
    args.need_masks = True
    args.need_features = False
    args.resolution = getattr(args, "resolution", -1)
    args.data_device = getattr(args, "data_device", "cuda")
    args.images = getattr(args, "images", "images")
    args.allow_principle_point_shift = getattr(args, "allow_principle_point_shift", False)
    args.eval = getattr(args, "eval", False)

    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            source_path, args.images, args.eval,
            need_features=False, need_masks=True,
            sample_rate=sample_rate,
            allow_principle_point_shift=args.allow_principle_point_shift,
            replica="replica" in model_path,
        )
    else:
        print("SAM-driven: Colmap sparse/ not found. Only Colmap format is supported.")
        return -1

    resolution_scale = 1.0
    try:
        cameras = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
    except Exception as e:
        print("SAM-driven: Failed to load cameras:", e)
        return -1

    xyz = scene_model.get_xyz  # (N, 3)
    N = xyz.shape[0]
    point_votes = defaultdict(lambda: defaultdict(int))  # point_id -> mask_id -> count

    for cam in tqdm(cameras, desc="SAM-driven: projecting views"):
        if cam.original_masks is None:
            continue
        masks = cam.original_masks  # (num_masks, H, W)
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
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

        u, v, valid = _project_points_to_image(
            xyz,
            cam.full_proj_transform.to(xyz.device),
            cam_w, cam_h,
        )
        valid = valid.cpu()
        valid_pids = torch.where(valid)[0].cpu().numpy()
        u_valid = u[valid].cpu().numpy()
        v_valid = v[valid].cpu().numpy()

        masks_dev = masks.to(xyz.device) if masks.device != xyz.device else masks
        for mid in range(masks.shape[0]):
            in_mask = masks_dev[mid, v_valid, u_valid].cpu().numpy()
            for idx in np.where(in_mask)[0]:
                point_votes[int(valid_pids[idx])][mid] += 1

    # Assign each point to mask with most votes (if >= min_votes)
    mask_id_to_point_mask = defaultdict(list)
    for pid in range(N):
        votes = point_votes.get(pid, {})
        if not votes:
            continue
        best_mask = max(votes.items(), key=lambda x: x[1])
        if best_mask[1] >= min_votes:
            mask_id_to_point_mask[best_mask[0]].append(pid)

    # Sort by mask_id for deterministic ordering; filter empty
    sorted_mask_ids = sorted(mask_id_to_point_mask.keys())
    segments_to_apply = []
    for mid in sorted_mask_ids:
        pids = mask_id_to_point_mask[mid]
        if len(pids) < 10:  # skip tiny segments
            continue
        mask = torch.zeros(N, dtype=torch.bool, device=xyz.device)
        mask[pids] = True
        segments_to_apply.append(mask)

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

    print(f"SAM-driven: Created {len(segments_to_apply)} segments from {len(cameras)} views.")
    return len(segments_to_apply)


def main():
    """Standalone: load models, run SAM-driven segmentation, save mask to file."""
    import argparse
    from scene import GaussianModel, FeatureGaussianModel

    parser = argparse.ArgumentParser(description="SAM-driven segmentation (standalone)")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-f", "--feature_iteration", type=int, default=10000)
    parser.add_argument("-s", "--scene_iteration", type=int, default=30000)
    parser.add_argument("-o", "--output", type=str, default="./segmentation_res/sam_driven_mask.pt")
    parser.add_argument("--min_votes", type=int, default=1)
    parser.add_argument("--sample_rate", type=float, default=1.0)
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

    n = run_sam_driven_segment(
        args.model_path,
        gs_model,
        feat_model,
        min_votes=args.min_votes,
        sample_rate=args.sample_rate,
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
