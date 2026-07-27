"""Command dispatcher for the native Windows tools."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from windows_bootstrap import configure_windows_runtime


TARGETS = {
    "viewer": "rt_gs_gui.py",
    "saga": "saga_gui.py",
    "train-scene": "train_scene.py",
    "train-feature": "train_contrastive_feature.py",
    "render": "render.py",
    "convert": "convert.py",
    "extract-sam2": "extract_sam2_masks.py",
    "get-scale": "get_scale.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="3DGS-RTMaterial Windows launcher")
    parser.add_argument("target", choices=TARGETS)
    args, forwarded = parser.parse_known_args()
    configure_windows_runtime()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    sys.argv = [str(root / TARGETS[args.target]), *forwarded]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
