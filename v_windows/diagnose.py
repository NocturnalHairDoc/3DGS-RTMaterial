"""Read-only prerequisite and Python dependency diagnostics."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

from windows_bootstrap import configure_windows_runtime


def status(ok: bool, label: str, detail: str = "") -> bool:
    mark = "OK" if ok else "MISSING"
    print(f"[{mark:7}] {label}" + (f": {detail}" if detail else ""))
    return ok


def main() -> int:
    configure_windows_runtime()
    print(f"Platform: {platform.platform()}")
    print(f"Python:   {sys.version.split()[0]} ({sys.executable})")

    essential_ok = True
    essential_ok &= status(sys.platform == "win32", "Native Windows")
    # Compilers are required for installation/JIT rebuilds, not for an already
    # built viewer runtime, so report them without failing runtime readiness.
    status(shutil.which("nvcc") is not None, "CUDA nvcc (build only)", shutil.which("nvcc") or "")
    status(shutil.which("cl") is not None, "MSVC cl.exe (build only)", shutil.which("cl") or "")

    modules = [
        ("torch", True), ("torchvision", True), ("dearpygui.dearpygui", True),
        ("diff_gaussian_rasterization", True), ("diff_gaussian_rasterization_contrastive_f", True),
        ("simple_knn._C", True), ("open_clip", False), ("hdbscan", False),
        ("sam2", False),
        ("threedgrut", False), ("threedgrt_tracer", False),
        ("threedgrt_tracer.lib3dgrt_cc", False),
        ("diff_gaussian_rasterization_depth", False),
    ]
    module_status = {}
    for module, essential in modules:
        try:
            imported = importlib.import_module(module)
            detail = getattr(imported, "__version__", "imported")
            ok = status(True, module, str(detail))
        except Exception as exc:
            ok = status(False, module, f"{type(exc).__name__}: {exc}")
        module_status[module] = ok
        if essential:
            essential_ok &= ok

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if cuda_ok else "not available"
        essential_ok &= status(cuda_ok, "PyTorch CUDA", detail)
        if cuda_ok:
            status(True, "CUDA capability", ".".join(map(str, torch.cuda.get_device_capability(0))))
    except Exception:
        pass

    project = Path(__file__).resolve().parent
    status((project / "vendor" / "3dgrut").is_dir(), "Local 3DGRUT source (optional)")
    if essential_ok and module_status.get("threedgrt_tracer.lib3dgrt_cc"):
        result = "ready for the viewer (OptiX plugin available)"
    elif essential_ok:
        result = "ready for the base viewer"
    else:
        result = "setup is incomplete"
    print("\nResult:", result)
    return 0 if essential_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
