#!/usr/bin/env python3
"""Fast, non-GUI diagnostics for the 3DGS-RTMaterial runtime."""

from __future__ import annotations

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
THREEDGRUT_ROOT = os.path.join(REPO_ROOT, "3dgrut")
if THREEDGRUT_ROOT not in sys.path:
    sys.path.insert(0, THREEDGRUT_ROOT)


REQUIRED_MODULES = (
    "torch",
    "dearpygui",
    "imageio",
    "imageio_ffmpeg",
    "diff_gaussian_rasterization",
    "diff_gaussian_rasterization_contrastive_f",
    "diff_gaussian_rasterization_depth",
    "simple_knn",
    "open_clip",
    "pytorch3d.ops",
    "segment_anything",
    "slangtorch",
    "threedgrut",
    "threedgrt_tracer",
)



def main() -> int:
    failures: list[str] = []
    print(f"Python: {sys.version.split()[0]}")
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            print(f"[OK] {name} {version}".rstrip())
        except Exception as exc:  # diagnostic must report every missing component
            failures.append(name)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
        if not torch.cuda.is_available():
            failures.append("cuda")
            print("[FAIL] CUDA is not available to PyTorch")
        else:
            capability = torch.cuda.get_device_capability()
            arch = f"sm_{capability[0]}{capability[1]}"
            supported = set(torch.cuda.get_arch_list())
            print(f"GPU: {torch.cuda.get_device_name()} ({arch})")
            if arch not in supported:
                failures.append("gpu_architecture")
                print(f"[FAIL] This PyTorch build does not include {arch}; supported: {sorted(supported)}")
            else:
                print(f"[OK] PyTorch includes {arch}")
    except Exception as exc:
        failures.append("torch_cuda_check")
        print(f"[FAIL] CUDA check: {type(exc).__name__}: {exc}")

    repo = REPO_ROOT
    for path in ("3dgrut/setup.py", "3dgrut/threedgrt_tracer", "submodules"):
        absolute = os.path.join(repo, path)
        if not os.path.exists(absolute):
            failures.append(path)
            print(f"[FAIL] Missing source: {absolute}")

    if failures:
        print("Runtime check failed: " + ", ".join(failures))
        return 1
    print("Runtime check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
