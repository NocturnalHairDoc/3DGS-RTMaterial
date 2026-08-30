#!/usr/bin/env python3
"""Reproducible 61k-Gaussian OptiX smoke and performance benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import imageio.v2 as imageio
import numpy as np
import torch

from viewer.export_manager import linear_to_srgb
from optix_integration import OptiXRenderer
from viewer.gui.base import OrbitCamera
from scene import GaussianModel


ROOT = Path(__file__).resolve().parents[2]


def _scene_ply(model: Path, iteration: int | None) -> Path:
    if iteration is not None:
        candidate = model / "point_cloud" / f"iteration_{iteration}" / "scene_point_cloud.ply"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate
    candidates = list(model.glob("point_cloud/iteration_*/scene_point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"no scene point cloud under {model}")
    return max(candidates, key=lambda path: int(path.parent.name.split("_")[-1]))


def _time_cuda(callable_):
    torch.cuda.synchronize()
    started = perf_counter()
    result = callable_()
    torch.cuda.synchronize()
    return result, perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path(os.environ.get("RTM_TEST_MODEL", ROOT / "output-v2/smoke_scene")))
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--stable-frames", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/gpu_smoke_61k.json")
    args = parser.parse_args()
    if args.stable_frames < 5:
        raise ValueError("--stable-frames must be at least 5")

    ply = _scene_ply(args.model.resolve(), args.iteration)
    model = GaussianModel(3)
    model.load_ply(str(ply))
    gaussian_count = int(model.get_xyz.shape[0])
    if not 60_000 <= gaussian_count <= 62_000:
        raise RuntimeError(f"expected a 61k scene, loaded {gaussian_count:,} Gaussians")

    camera = OrbitCamera(args.width, args.height)
    renderer = OptiXRenderer(model, args.width, args.height, 3,
                             torch.zeros(3, device="cuda"))
    if not renderer.available:
        raise RuntimeError("OptiX backend unavailable")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _, bvh_seconds = _time_cuda(lambda: renderer.build_bvh())
    first, first_seconds = _time_cuda(lambda: renderer.render(camera))
    frame_seconds = []
    last = first
    for _ in range(args.stable_frames):
        last, elapsed = _time_cuda(lambda: renderer.render(camera))
        frame_seconds.append(elapsed)
        print(f"stable frame {len(frame_seconds)}/{args.stable_frames}: "
              f"{elapsed * 1000:.2f} ms", flush=True)

    for key in ("rgb", "normals", "depth", "opacity"):
        if not torch.isfinite(last[key]).all():
            raise RuntimeError(f"NaN/Inf in {key}")
    if float(last["opacity"].max()) <= 0:
        raise RuntimeError("smoke camera produced an empty frame")

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    sorted_times = sorted(frame_seconds)
    p95_index = min(len(sorted_times) - 1, int(np.ceil(0.95 * len(sorted_times))) - 1)
    stable_mean = mean(frame_seconds)
    device = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "scene": str(ply.relative_to(args.model.resolve().parent)),
        "gaussian_count": gaussian_count,
        "resolution": [args.width, args.height],
        "stable_frame_count": args.stable_frames,
        "timings_ms": {
            "bvh_build": bvh_seconds * 1000,
            "first_frame": first_seconds * 1000,
            "stable_mean": stable_mean * 1000,
            "stable_median": median(frame_seconds) * 1000,
            "stable_p95": sorted_times[p95_index] * 1000,
            "stable_min": min(frame_seconds) * 1000,
            "stable_max": max(frame_seconds) * 1000,
        },
        "stable_fps": 1.0 / stable_mean,
        "vram_mib": {"peak_allocated": peak_allocated, "peak_reserved": peak_reserved},
        "system": {
            "gpu": device.name,
            "gpu_total_mib": device.total_memory / 1024**2,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
        },
        "validation": {"finite_channels": True, "nonempty_opacity": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    preview = args.output.with_suffix(".png")
    imageio.imwrite(preview, (linear_to_srgb(last["rgb"].cpu().numpy()) * 255 + 0.5).astype(np.uint8))
    print(json.dumps(result, indent=2))
    print(f"result={args.output} preview={preview}")


if __name__ == "__main__":
    main()
