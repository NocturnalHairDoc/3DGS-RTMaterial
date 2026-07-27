"""Windows runtime setup shared by launchers and diagnostic tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_HANDLES = []


def configure_windows_runtime() -> None:
    if sys.platform != "win32":
        return

    candidates = []
    for key in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value) / "bin")
    for prefix in filter(None, (os.environ.get("CONDA_PREFIX"), sys.prefix)):
        root = Path(prefix)
        candidates.extend((root / "Library" / "bin", root / "DLLs", root / "Scripts"))

    try:
        import site
        for package_root in site.getsitepackages():
            candidates.append(Path(package_root) / "torch" / "lib")
    except Exception:
        pass

    seen = set()
    for directory in candidates:
        resolved = str(directory.resolve())
        if resolved in seen or not directory.is_dir():
            continue
        seen.add(resolved)
        os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(resolved))

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONUTF8", "1")


def windows_font_candidates() -> list[str]:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    return [
        str(windir / "Fonts" / "msyh.ttc"),
        str(windir / "Fonts" / "segoeui.ttf"),
        str(windir / "Fonts" / "arial.ttf"),
    ]
