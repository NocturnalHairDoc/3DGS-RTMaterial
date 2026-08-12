#!/usr/bin/env python3
"""Headless real-OptiX PNG exporter used for 1080p/4K validation."""

import argparse
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
DEFAULT_MODEL = os.environ.get("RTM_TEST_MODEL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Compatible trained model directory (or set RTM_TEST_MODEL)")
    args = parser.parse_args()
    if not args.model or not Path(args.model).is_dir():
        raise ValueError("Provide --model or set RTM_TEST_MODEL")
    output_path = Path(args.output).resolve()
    export_root = (ROOT / "exports").resolve()
    if export_root not in output_path.parents:
        raise ValueError(f"output must be inside {export_root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = GaussianModel(3)
    model.load_ply(str(Path(args.model) / "point_cloud/iteration_30000/scene_point_cloud.ply"))
    renderer = OptiXRenderer(model, args.width, args.height, 3, torch.zeros(3, device="cuda"))
    if not renderer.available:
        raise RuntimeError("OptiX backend unavailable")
    renderer.build_bvh()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    result = renderer.render_tiled(OrbitCamera(args.width, args.height), tile_size=args.tile)
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    rgb = result["rgb"]
    if rgb.shape != (args.height, args.width, 3) or not torch.isfinite(rgb).all():
        raise RuntimeError(f"invalid RGB output {tuple(rgb.shape)}")
    imageio.imwrite(output_path, (linear_to_srgb(rgb.numpy()) * 255 + 0.5).astype(np.uint8))
    decoded = imageio.imread(output_path)
    if decoded.shape != (args.height, args.width, 3):
        raise RuntimeError(f"decoded PNG shape mismatch: {decoded.shape}")
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"export={args.width}x{args.height} tile={args.tile} elapsed={elapsed:.3f}s "
          f"peak={peak:.1f}MiB output={output_path}")


if __name__ == "__main__":
    main()
