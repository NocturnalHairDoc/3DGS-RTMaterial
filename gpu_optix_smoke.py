#!/usr/bin/env python3
"""Real-GPU OptiX single-frame and tiled-equivalence smoke test."""

import os
from pathlib import Path
from time import perf_counter

import imageio.v2 as imageio
import numpy as np
import torch

from export_manager import linear_to_srgb
from optix_integration import OptiXRenderer
from rt_gs_gui import OrbitCamera
from scene import GaussianModel


ROOT = Path(__file__).resolve().parent
MODEL_SETTING = os.environ.get("RTM_TEST_MODEL")
MODEL = Path(MODEL_SETTING) if MODEL_SETTING else None


def main() -> None:
    if MODEL is None or not MODEL.is_dir():
        raise ValueError("Set RTM_TEST_MODEL to a compatible trained model directory")
    width, height = 96, 64
    model = GaussianModel(3)
    model.load_ply(str(MODEL / "point_cloud/iteration_30000/scene_point_cloud.ply"))
    camera = OrbitCamera(width, height)
    renderer = OptiXRenderer(model, width, height, 3, torch.zeros(3, device="cuda"))
    if not renderer.available:
        raise RuntimeError("OptiX backend unavailable")
    renderer.build_bvh()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    full = renderer.render(camera)
    torch.cuda.synchronize()
    full_seconds = perf_counter() - started
    started = perf_counter()
    tiled = renderer.render_tiled(camera, tile_size=32)
    torch.cuda.synchronize()
    tiled_seconds = perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    for key in ("rgb", "normals", "depth", "opacity"):
        if not torch.isfinite(full[key]).all() or not torch.isfinite(tiled[key]).all():
            raise RuntimeError(f"NaN/Inf in {key}")
    visible = torch.ones(model.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    visible[::2] = False
    hidden = renderer.render(camera, segment_mask=visible)
    hidden_difference = float((full["rgb"] - hidden["rgb"]).abs().max())
    if hidden_difference <= 1e-5:
        raise RuntimeError("visibility mask did not change the rendered frame")
    difference = (full["rgb"].cpu() - tiled["rgb"]).abs()
    if float(difference.max()) > 2e-5:
        raise RuntimeError(f"tiled seam/mismatch: max={float(difference.max())}")
    path = ROOT / "exports" / "optix_tiled_smoke_96x64.png"
    srgb = linear_to_srgb(tiled["rgb"].numpy())
    imageio.imwrite(path, (srgb * 255 + 0.5).astype(np.uint8))
    print(f"OptiX full={full_seconds:.4f}s tiled={tiled_seconds:.4f}s "
          f"peak={peak_mib:.1f}MiB max_diff={float(difference.max()):.8g} "
          f"hidden_diff={hidden_difference:.6g} output={path}")


if __name__ == "__main__":
    main()
