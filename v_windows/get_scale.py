import math
import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel

import gaussian_renderer
import importlib
importlib.reload(gaussian_renderer)

ALLOW_PRINCIPLE_POINT_SHIFT = False


def get_combined_args(parser : ArgumentParser):
    # cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()

    target_cfg_file = "cfg_args"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except FileNotFoundError:
        print("Config file not found: {}".format(cfgfilepath))
        print("Hint: If you trained without -m/--model_path, the output is under ./output/<uuid_prefix>/ (e.g. ./output/33a6e73f-5/). Use that path as --model_path.")
        raise
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    # for k in args_cfgfile.__dict__.keys():
        # print(k, args_cfgfile.__dict__[k], "?")

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    # for k in merged_dict.keys():
        # print(k, merged_dict[k])
    return Namespace(**merged_dict)

def generate_grid_index(depth):
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack((x, y), dim=-1)


if __name__ == '__main__':

    parser = ArgumentParser(description="Get scales for SAM masks")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=None, type=int,
                        help="Process one training-view index (for diagnostics or repair).")
    parser.add_argument('--precomputed_mask', default=None, type=str)

    parser.add_argument("--image_root", default=None, type=str,
                        help="Dataset root. Defaults to source_path stored in the trained model config.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute scale files that already exist.")
    parser.add_argument("--limit", default=None, type=int,
                        help="Process only the first N training views (for diagnostics).")

    args = get_combined_args(parser)

    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False

    # ALLOW_PRINCIPLE_POINT_SHIFT = 'lerf' in args.model_path
    dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT

    scene_gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, scene_gaussians, None, load_iteration=args.iteration, feature_load_iteration=-1, shuffle=False, mode='eval', target='scene')

    mask_dir = os.path.join(dataset.source_path, 'sam_masks')
    assert os.path.isdir(mask_dir), "Please run the SAM2 mask extraction first."

    def load_masks_for_view(image_name):
        # image_name may be with or without extension (e.g. IMG_4026.JPG or IMG_4026)
        base = os.path.splitext(image_name)[0]
        pt_name = base + '.pt'
        path = os.path.join(mask_dir, pt_name)
        try:
            return torch.load(path, map_location='cpu', weights_only=True).float()
        except TypeError:
            return torch.load(path, map_location='cpu').float()

    output_root = getattr(args, "image_root", None) or dataset.source_path
    OUTPUT_DIR = os.path.join(output_root, 'mask_scales')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cameras = scene.getTrainCameras()
    view_index = getattr(args, "idx", None)
    limit = getattr(args, "limit", None)
    if view_index is not None:
        if not 0 <= view_index < len(cameras):
            raise IndexError(f"--idx must be in [0, {len(cameras) - 1}]")
        cameras = [cameras[view_index]]
    elif limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        cameras = cameras[:limit]

    missing_masks = [
        view.image_name for view in cameras
        if not os.path.isfile(os.path.join(mask_dir, os.path.splitext(view.image_name)[0] + '.pt'))
    ]
    if missing_masks:
        preview = ', '.join(missing_masks[:5])
        raise FileNotFoundError(f"Missing {len(missing_masks)} SAM2 mask files (first: {preview})")

    background = torch.zeros(3, dtype=scene_gaussians.get_xyz.dtype, device='cuda')
    pipe = pipeline.extract(args)
    completed = 0
    skipped = 0

    for view in tqdm(cameras, desc="Computing mask scales"):
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(view.image_name)[0] + '.pt')
        if os.path.isfile(output_path) and not args.overwrite:
            skipped += 1
            continue

        rendered_pkg = gaussian_renderer.render_with_depth(view, scene_gaussians, pipe, background)
        depth = rendered_pkg['depth'].detach().cpu().squeeze(0)

        corresponding_masks = load_masks_for_view(view.image_name)
        if corresponding_masks.ndim == 2:
            corresponding_masks = corresponding_masks.unsqueeze(0)

        grid_index = generate_grid_index(depth)

        points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3)
        points_in_3D[:,:,-1] = depth

        # Reconstruct camera-space points using this view's intrinsics.
        cx = depth.shape[1] / 2
        cy = depth.shape[0] / 2
        fx = cx / math.tan(view.FoVx / 2)
        fy = cy / math.tan(view.FoVy / 2)

        points_in_3D[:,:,0] = (grid_index[:,:,0] - cx) * depth / fx
        points_in_3D[:,:,1] = (grid_index[:,:,1] - cy) * depth / fy

        upsampled_mask = F.interpolate(corresponding_masks.unsqueeze(1), mode='bilinear', size=depth.shape, align_corners=False)

        eroded_masks = F.conv2d(
            upsampled_mask.float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

        scale = torch.zeros(len(corresponding_masks))
        valid_depth = torch.isfinite(depth) & (depth > 0)
        for mask_id in range(len(corresponding_masks)):
            valid_mask = eroded_masks[mask_id] & valid_depth
            point_in_3D_in_mask = points_in_3D[valid_mask]
            # Tiny boundary masks can disappear during erosion. Preserve their
            # scale by falling back to the un-eroded, thresholded SAM2 mask.
            if point_in_3D_in_mask.shape[0] < 2:
                valid_mask = (upsampled_mask[mask_id, 0] >= 0.5) & valid_depth
                point_in_3D_in_mask = points_in_3D[valid_mask]
            if point_in_3D_in_mask.shape[0] >= 2:
                scale[mask_id] = (point_in_3D_in_mask.std(dim=0, correction=0) * 2).norm()

        # A SAM2 region that lies entirely outside the reconstructed Gaussian
        # surface has no measurable 3D extent. Use this view's median valid
        # scale so downstream scale-gate training never receives NaN or zero.
        invalid_scale = ~torch.isfinite(scale) | (scale <= 0)
        if invalid_scale.any():
            valid_scale = scale[~invalid_scale]
            fallback_scale = valid_scale.median() if valid_scale.numel() else torch.tensor(1.0)
            scale[invalid_scale] = fallback_scale

        torch.save(scale, output_path)
        completed += 1
        del corresponding_masks, depth, grid_index, points_in_3D, upsampled_mask, eroded_masks, scale, rendered_pkg
    print(f"Mask scales ready: {completed} computed, {skipped} skipped, output={OUTPUT_DIR}")
