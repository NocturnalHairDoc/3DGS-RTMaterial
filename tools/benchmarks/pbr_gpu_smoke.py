"""Headless real-scene validation of the V3 G-buffer and secondary-ray path."""

from __future__ import annotations

import argparse
import json
import os
import time
from argparse import Namespace

import imageio.v3 as iio
import numpy as np
import torch

from optix_integration import OptiXRenderer
from materials import MaterialSHEditor
from materials.pbr_lite import (
    HDREnvironment,
    PBRLiteCompositor,
    PBRMaterial,
    reconstruct_world_positions,
    reflection_directions,
    refraction_directions,
    srgb_to_linear,
    stabilize_gbuffer_normals,
)
from segmentation.sam_driven import _load_cfg_args
from scene import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


MATERIAL_ROUNDS = {
    "dielectric": PBRMaterial((0.62, 0.28, 0.12), 0.55, 0.0, 1.0, 1.5),
    "metal": PBRMaterial((0.83, 0.57, 0.22), 0.18, 0.92, 1.0, 1.5),
    "glass": PBRMaterial((0.68, 0.88, 1.0), 0.08, 0.0, 0.22, 1.5),
}


def time_cuda(operation):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def load_camera(model_path, camera_index=0, resolution=8):
    cfg = _load_cfg_args(model_path)
    if cfg is None:
        raise FileNotFoundError(f"cfg_args missing in {model_path}")
    args = Namespace(**vars(cfg))
    args.source_path = cfg.source_path
    args.model_path = model_path
    args.images = getattr(args, "images", "images")
    args.eval = False
    args.resolution = resolution
    args.data_device = "cuda"
    args.allow_principle_point_shift = getattr(args, "allow_principle_point_shift", False)
    info = sceneLoadTypeCallbacks["Colmap"](
        args.source_path, args.images, False, need_features=False, need_masks=False,
        sample_rate=1.0, allow_principle_point_shift=args.allow_principle_point_shift,
        replica=False,
    )
    index = int(camera_index) % len(info.train_cameras)
    return cameraList_from_camInfos([info.train_cameras[index]], 1.0, args)[0]


def world_rays(renderer, camera):
    batch = renderer._ray_batch(camera)
    transform = batch.T_to_world[0]
    rotation, translation = transform[:3, :3], transform[:3, 3]
    directions = torch.nn.functional.normalize(batch.rays_dir[0] @ rotation.T, dim=-1)
    origins = batch.rays_ori[0] @ rotation.T + translation
    return origins, directions


@torch.no_grad()
def render_rounds(model_path, scene_iteration=30000, camera_index=0, resolution=8,
                  environment_path=None):
    ply = os.path.join(model_path, "point_cloud", f"iteration_{scene_iteration}",
                       "scene_point_cloud.ply")
    model = GaussianModel(3)
    model.load_ply(ply)
    camera = load_camera(model_path, camera_index, resolution)
    renderer = OptiXRenderer(model, width=camera.image_width, height=camera.image_height)
    if not renderer.available:
        raise RuntimeError("3DGRT OptiX tracer is unavailable")
    renderer.build_bvh()

    primary, primary_seconds = time_cuda(lambda: renderer.render(camera))
    # Strict previous-version control: V2.2 Metal uses temporary SH edits on
    # the same model/camera and is restored before all V3 rounds.
    editor = MaterialSHEditor(model)
    original_mask = model._mask.clone()
    model._mask.fill_(2)
    editor.apply_all({1: {"type": "Metal"}}, model._mask)
    try:
        stylized = renderer.render(camera)
    finally:
        editor.restore()
        model._mask.copy_(original_mask)
    origins, directions = world_rays(renderer, camera)
    depth, normals = primary["depth"], primary["normals"]
    view = -directions
    positions = reconstruct_world_positions(origins, directions, depth)
    normals = stabilize_gbuffer_normals(normals, positions, depth, view)
    epsilon = depth[depth > 0].median() * 1e-2 if torch.any(depth > 0) else depth.new_tensor(1e-2)
    light = torch.tensor([0.45, 0.78, 0.43], device="cuda")
    light = torch.nn.functional.normalize(light, dim=0)

    shadow, shadow_seconds = time_cuda(lambda: renderer.trace_world_rays(
        positions + normals * epsilon, light.view(1, 1, 3).expand_as(positions)))
    visibility = (1 - shadow["opacity"]).clamp(0.03, 1)

    environment = (HDREnvironment.load(environment_path, device="cuda")
                   if environment_path else HDREnvironment.procedural(device="cuda"))
    compositor = PBRLiteCompositor(environment)
    images, metrics = {"original": primary["rgb"], "v22_stylized": stylized["rgb"]}, {}
    hit_primary = primary["depth"] > 0
    stylized_luminance = stylized["rgb"].mean(-1)
    metrics["v22_stylized"] = {
        "finite_fraction": float(torch.isfinite(stylized["rgb"]).all(-1).float().mean()),
        "mean_luminance": float(stylized_luminance[hit_primary].mean()) if hit_primary.any() else 0.0,
        "contrast": float(stylized_luminance[hit_primary].std()) if hit_primary.any() else 0.0,
    }
    for name, material in MATERIAL_ROUNDS.items():
        shape = (*depth.shape, 1)
        albedo = torch.tensor(material.albedo, device="cuda").view(1, 1, 3).expand(*depth.shape, 3)
        roughness = torch.full(shape, material.roughness, device="cuda")
        metallic = torch.full(shape, material.metallic, device="cuda")
        opacity = torch.full(shape, material.opacity, device="cuda")
        reflected_dirs = reflection_directions(directions, normals, roughness)
        refracted_dirs = refraction_directions(
            directions, normals, torch.full(shape, material.ior, device="cuda"))
        (reflected, refracted), secondary_seconds = time_cuda(lambda: (
            renderer.trace_world_rays(positions + normals * epsilon, reflected_dirs),
            renderer.trace_world_rays(positions - normals * epsilon, refracted_dirs),
        ))
        reflected_linear = torch.where(
            reflected["opacity"] > 1e-3, srgb_to_linear(reflected["rgb"]),
            environment.sample(reflected_dirs, roughness))
        refracted_linear = torch.where(
            refracted["opacity"] > 1e-3, srgb_to_linear(refracted["rgb"]),
            environment.sample(refracted_dirs, roughness))
        image = compositor.shade(
            srgb_to_linear(albedo), roughness, metallic, opacity, normals, depth,
            view, light, torch.tensor([1.5, 1.5, 1.5], device="cuda"), visibility,
            primary_rgb=primary["rgb"], reflected_linear=reflected_linear,
            refracted_linear=refracted_linear, gbuffer_opacity=primary["opacity"],
        )
        images[name] = image
        hit = depth > 0
        luminance = image.mean(-1)
        metrics[name] = {
            "finite_fraction": float(torch.isfinite(image).all(-1).float().mean()),
            "mean_luminance": float(luminance[hit].mean()) if hit.any() else 0.0,
            "contrast": float(luminance[hit].std()) if hit.any() else 0.0,
            "highlight_fraction": float((luminance[hit] > 0.9).float().mean()) if hit.any() else 0.0,
            "secondary_seconds": secondary_seconds,
        }
    metrics["runtime"] = {
        "primary_seconds": primary_seconds,
        "shadow_seconds": shadow_seconds,
        "mean_visibility": float(visibility[depth > 0].mean()) if torch.any(depth > 0) else 1.0,
        "gaussians": int(model.get_xyz.shape[0]),
        "width": int(camera.image_width), "height": int(camera.image_height),
    }
    return images, metrics, camera.image_name


def save_panel(images, path):
    ordered = [images[key] for key in ("original", "v22_stylized", "dielectric", "metal", "glass")]
    arrays = [(image.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
              for image in ordered]
    panel = np.concatenate(arrays, axis=1)
    iio.imwrite(path, panel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-s", "--scene_iteration", type=int, default=30000)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--environment", default=None)
    parser.add_argument("--output_dir", default="comparisons/pbr_smoke")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    images, metrics, camera_name = render_rounds(
        args.model_path, args.scene_iteration, args.camera_index, args.resolution,
        args.environment,
    )
    scene_name = os.path.basename(os.path.normpath(args.model_path))
    panel = os.path.join(args.output_dir, f"{scene_name}_{camera_name}_pbr_rounds.png")
    save_panel(images, panel)
    payload = {"scene": scene_name, "camera": camera_name,
               "panel": os.path.relpath(panel),
               "metrics": metrics}
    with open(os.path.join(args.output_dir, f"{scene_name}_metrics.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
