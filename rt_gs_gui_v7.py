"""
rt_gs_gui_v7.py  —  Material-SH viewer
=======================================
Strategy 1: edit Gaussian SH coefficients in-place to simulate material
appearance, then restore after rendering.

Changes from rt_gs_gui.py
--------------------------
- Adds MaterialSHEditor (material_sh_edit/) after model load.
- Replaces the Blinn-Phong post-processing pipeline in _render_raytracing
  with a simple: apply-SH-edits → render → restore cycle.
- Keeps all segmentation, CLIP, camera, and UI logic unchanged (inherited).

Why this works
--------------
SH DC term  = base diffuse colour (albedo).
Higher-order SH terms = view-dependent appearance (specular-like shimmer).

Metal  : desaturate DC → achromatic grey  +  amplify higher-order SH
         → grey base that shimmers differently from every camera angle.
Glass  : blue-tint DC  +  reduce opacity to ~28 %
         → translucent blue object; objects behind it show through.
Plastic: saturate DC   +  zero 85 % of higher-order SH
         → vivid flat colour with almost no view-dependence.
Matte  : zero ALL higher-order SH
         → purely diffuse; identical from every angle, "dead flat".
Default: no change.

Usage
-----
    python rt_gs_gui_v7.py -m ./output/bicycle --scale 2.0
"""

import os
import torch
import dearpygui.dearpygui as dpg
from argparse import ArgumentParser

from rt_gs_gui import RTGSViewerGUI, RTGSConfig, KEYWORD_MATERIAL_MAP
from scene import GaussianModel, FeatureGaussianModel
from gaussian_renderer import render

from material_sh_edit import MaterialSHEditor
from clip_utils.material_clip import (
    params_from_text_prompt,
    MATERIAL_PARAM_PRESETS,
)


class SHMaterialViewer(RTGSViewerGUI):
    """RTGSViewerGUI subclass that uses SH coefficient editing for materials."""

    # ------------------------------------------------------------------ init --

    def __init__(self, opt, gs_model, feat_model, scale_gate):
        # Parent __init__ loads the model, builds OptiX BVH (if available),
        # and creates the DearPyGui context.
        super().__init__(opt, gs_model, feat_model, scale_gate)

        # Attach the SH editor now that the model is loaded into GPU.
        self._sh_editor = MaterialSHEditor(self.engine["scene"])
        print("[V7] MaterialSHEditor ready — SH-based material rendering active.")

    # -------------------------------------------------------- rendering core --

    @torch.no_grad()
    def _render_raytracing(self, cam, pipe):
        """
        Material rendering via SH coefficient editing.

        Sub-modes
        ---------
        Material (SH) : apply SH edits, render, restore  — primary path
        Original      : plain SH render, no edits
        Compare       : left half = SH-edited, right half = original
        """
        hide_mask = self._get_hide_mask()
        rt_submode = (dpg.get_value("_rt_submode")
                      if dpg.does_item_exist("_rt_submode") else "Material (SH)")

        # ── plain original render ───────────────────────────────────────────
        def _render_original():
            return render(cam, self.engine["scene"], pipe,
                          self.bg_color, filtered_mask=hide_mask)["render"]

        # ── SH-edited render ────────────────────────────────────────────────
        def _render_sh():
            if self._sh_editor is None or not self.material_assignments:
                return _render_original()
            self._sh_editor.apply_all(
                self.material_assignments,
                self.engine["scene"]._mask,
            )
            out = render(cam, self.engine["scene"], pipe,
                         self.bg_color, filtered_mask=hide_mask)["render"]
            self._sh_editor.restore()
            return out

        # ── dispatch ────────────────────────────────────────────────────────
        if rt_submode == "Original":
            return _render_original()

        if rt_submode == "Compare":
            orig = _render_original()
            edited = _render_sh()
            W = orig.shape[2]
            # Left half = original, right half = SH-edited
            result = orig.clone()
            result[:, :, W // 2:] = edited[:, :, W // 2:]
            # Draw a thin white divider
            result[:, :, W // 2 - 1: W // 2 + 1] = 1.0
            return result

        # Default: "Material (SH)"
        return _render_sh()

    # ──────────────────────────── text-prompt material application ──────────

    def _apply_text_prompt_material(self):
        """Read the text prompt input and apply params to the selected segment."""
        if not dpg.does_item_exist("_mat_prompt_input"):
            return
        text = dpg.get_value("_mat_prompt_input").strip()
        if not text:
            return

        # Parse material parameters from free-text description
        params = params_from_text_prompt(text)

        # Find selected segment
        val = dpg.get_value("_material_segment_select") if dpg.does_item_exist("_material_segment_select") else None
        seg_id = self._parse_segment_select_value(val)
        if seg_id < 1:
            print("[V8] No segment selected — select a segment first.")
            return

        # Store as Custom type with parsed params
        self.material_assignments[seg_id] = {
            "type": "Custom",
            "name": text,
            "params": params,
        }
        self.material_labels[seg_id] = text
        self._mat_map_dirty = True
        self._dirty = True
        print(f"[V8] Segment {seg_id} ← '{text}' → params: {params}")

    # ─────────────────────────────────────────── override UI sub-mode items --

    def _register_dpg(self):
        """Override to swap sub-mode combo and add text-prompt panel."""
        super()._register_dpg()

        # Replace the combo choices with V8-specific ones
        if dpg.does_item_exist("_rt_submode"):
            dpg.configure_item(
                "_rt_submode",
                items=["Material (SH)", "Original", "Compare"],
                default_value="Material (SH)",
            )

        # Hide light controls (no longer needed for SH editing)
        for tag in ("_LightAz", "_LightEl", "_FillIntensity"):
            if dpg.does_item_exist(tag):
                dpg.hide_item(tag)

        # Add text-prompt material panel inside the RT group
        if dpg.does_item_exist("_rt_group"):
            with dpg.group(parent="_rt_group"):
                dpg.add_separator()
                dpg.add_text("Material Prompt (team CLIP module)")
                dpg.add_input_text(
                    hint="e.g. 'frosted glass', 'shiny gold metal'",
                    tag="_mat_prompt_input",
                    width=-1,
                )
                dpg.add_button(
                    label="Apply Prompt to Segment",
                    callback=lambda: self._apply_text_prompt_material(),
                    width=-1,
                )
                dpg.add_text("Preset params:", color=[180, 180, 180])
                for mat_name, preset in MATERIAL_PARAM_PRESETS.items():
                    label = (f"{mat_name}: "
                             f"spec={preset['specular_gain']:.1f}  "
                             f"sat={preset['saturation']:.1f}  "
                             f"opa={preset['opacity_scale']:.2f}")
                    dpg.add_text(label, color=[140, 200, 140])


# ──────────────────────────────────────────────────────────── entry point ──

def main():
    parser = ArgumentParser(description="SAGA V7 — SH Material Viewer")
    parser.add_argument("-m", "--model_path", type=str, default="./output/figurines")
    parser.add_argument("-f", "--feature_iteration", type=int, default=10000)
    parser.add_argument("-s", "--scene_iteration", type=int, default=30000)
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Window scale divisor (2 = half resolution)")
    args = parser.parse_args()

    opt = RTGSConfig()
    opt.r = args.scale
    opt.window_width  = int(2160 / opt.r)
    opt.window_height = int(1200 / opt.r)
    opt.width         = opt.window_width
    opt.height        = opt.window_height
    opt.control_width  = int(350 * (2 / opt.r))
    opt.control_height = int(700 * (2 / opt.r))
    opt.font_size = min(28, max(16, int(18 * (2 / opt.r))))
    opt.MODEL_PATH = args.model_path
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION   = args.scene_iteration

    for name, path in [
        ("Scene PLY",   opt.SCENE_PCD_PATH),
        ("Feature PLY", opt.FEATURE_PCD_PATH),
        ("Scale gate",  opt.SCALE_GATE_PATH),
    ]:
        if not os.path.isfile(path):
            print(f"Error: {name} not found: {path}")
            return

    gs_model   = GaussianModel(opt.sh_degree)
    feat_model = FeatureGaussianModel(opt.FEATURE_DIM)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, opt.FEATURE_DIM, bias=True),
        torch.nn.Sigmoid(),
    ).cuda()

    gui = SHMaterialViewer(opt, gs_model, feat_model, scale_gate)
    gui.render()


if __name__ == "__main__":
    main()
