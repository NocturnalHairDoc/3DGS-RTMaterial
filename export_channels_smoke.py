#!/usr/bin/env python3
"""Exercise every offline still-image channel on real OptiX."""

import os
from pathlib import Path

import dearpygui.dearpygui as dpg
import imageio.v2 as imageio
import numpy as np
import torch

from rt_gs_gui import RTGSConfig
from rt_gs_gui_sh_clip import SHMaterialViewer
from scene import FeatureGaussianModel, GaussianModel
from export_manager import capture_export_snapshot
from material_sh_edit import MaterialSHEditor


def main():
    root = Path(__file__).resolve().parent
    opt = RTGSConfig()
    model_path = os.environ.get("RTM_TEST_MODEL")
    if not model_path or not Path(model_path).is_dir():
        raise ValueError("Set RTM_TEST_MODEL to a compatible trained model directory")
    opt.MODEL_PATH = model_path
    opt.width = opt.window_width = 64
    opt.height = opt.window_height = 48
    opt.control_width, opt.control_height = 350, 48
    scale_gate = torch.nn.Sequential(torch.nn.Linear(1, opt.FEATURE_DIM), torch.nn.Sigmoid()).cuda()
    gui = SHMaterialViewer(opt, GaussianModel(opt.sh_degree),
                           FeatureGaussianModel(opt.FEATURE_DIM), scale_gate)
    snapshot = capture_export_snapshot(gui.engine["scene"], gui.camera,
                                       gui.material_assignments, gui.hidden_segments)
    renderer = gui._new_export_renderer(64, 48, snapshot.scene)
    renderer.build_bvh()
    editor = MaterialSHEditor(snapshot.scene)
    cases = {
        "RGB": "channel_rgb.png", "RGBA": "channel_rgba.png",
        "Depth": "channel_depth.png", "Normals": "channel_normals.png",
        "Segmentation ID": "channel_segmentation_id.png",
        "Material ID": "channel_material_id.png",
        "Original / Edited": "channel_comparison.png",
    }
    for channel, filename in cases.items():
        path = gui._write_export_image(str(root / "exports" / filename), renderer,
                                       snapshot, editor, channel)
        decoded = imageio.imread(path)
        if decoded.shape[:2] != (48, 64):
            raise RuntimeError(f"{channel}: invalid decoded shape {decoded.shape}")
        print(channel, Path(path).name, decoded.shape, decoded.dtype)
    dpg.destroy_context()


if __name__ == "__main__":
    main()
