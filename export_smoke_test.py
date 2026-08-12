#!/usr/bin/env python3
"""Render the required tiny PNG/MP4 smoke-test artifacts."""

import os
from pathlib import Path

import dearpygui.dearpygui as dpg
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from rt_gs_gui import RTGSConfig
from rt_gs_gui_sh_clip import SHMaterialViewer
from scene import FeatureGaussianModel, GaussianModel


MODEL_SETTING = os.environ.get("RTM_TEST_MODEL")
MODEL_PATH = Path(MODEL_SETTING) if MODEL_SETTING else None
EXPORT_DIR = Path(__file__).resolve().parent / "exports"


def main() -> None:
    if MODEL_PATH is None or not MODEL_PATH.is_dir():
        raise ValueError("Set RTM_TEST_MODEL to a compatible trained model directory")
    opt = RTGSConfig()
    opt.MODEL_PATH = str(MODEL_PATH)
    opt.width = opt.window_width = 64
    opt.height = opt.window_height = 48
    opt.control_width = 350
    opt.control_height = 48

    gs_model = GaussianModel(opt.sh_degree)
    feat_model = FeatureGaussianModel(opt.FEATURE_DIM)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, opt.FEATURE_DIM, bias=True), torch.nn.Sigmoid()
    ).cuda()
    gui = SHMaterialViewer(opt, gs_model, feat_model, scale_gate)
    dpg.set_value("_view_mode", gui.VIEW_RAYTRACING)

    EXPORT_DIR.mkdir(exist_ok=True)
    base_rotation = gui.camera.rot
    frames = []
    for index in range(3):
        gui.camera.rot = Rotation.from_rotvec(
            np.array([0.0, 2.0 * np.pi * index / 3.0, 0.0])
        ) * base_rotation
        gui.fetch_data(gui._construct_camera())
        frame = np.clip(gui.render_buffer * 255.0, 0, 255).astype(np.uint8)
        if frame.shape != (48, 64, 3):
            raise RuntimeError(f"unexpected frame shape: {frame.shape}")
        frames.append(frame)

    imageio.imwrite(EXPORT_DIR / "bicycle_smoke_64x48.png", frames[0])
    with imageio.get_writer(
        EXPORT_DIR / "bicycle_smoke_3frames_2fps.mp4",
        fps=2,
        codec="libx264",
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    dpg.destroy_context()


if __name__ == "__main__":
    main()
