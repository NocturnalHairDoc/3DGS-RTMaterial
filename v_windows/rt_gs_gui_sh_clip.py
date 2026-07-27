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
- Adds a dedicated "CLIP Material" panel below Material with:
    · Auto Detect  — uses per-Gaussian CLIP features (already on GPU) to rank
                     materials without loading any extra model at first.
    · Text prompt  — "blue glass", "brushed metal", etc. → continuous params.

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
    MATERIAL_CLIP_PROMPTS,
)


class SHMaterialViewer(RTGSViewerGUI):
    """RTGSViewerGUI subclass that uses SH coefficient editing for materials."""

    # ------------------------------------------------------------------ init --

    def __init__(self, opt, gs_model, feat_model, scale_gate):
        # CLIP auto-detection state — must exist before _register_dpg is called
        self._clip_network = None
        self._clip_detected_material = None

        # Parent __init__ loads model, builds BVH, creates DearPyGui context,
        # and calls self._register_dpg() via dynamic dispatch.
        super().__init__(opt, gs_model, feat_model, scale_gate)

        # Attach the SH editor now that the model is loaded into GPU.
        self._sh_editor = MaterialSHEditor(self.engine["scene"])
        print("[V8] MaterialSHEditor ready — SH-based material rendering active.")

    # -------------------------------------------------------- rendering core --

    @torch.no_grad()
    def _render_raytracing(self, cam, pipe):
        """
        Combined OptiX ray tracing + SH material editing.

        Pipeline
        --------
        1. Apply SH edits to Gaussian parameters (DC tint, higher-order SH scale, opacity).
        2. Run OptiX ray tracer — it reads the (modified) SH/opacity via GaussianAdapter.
           BVH does NOT need rebuilding: geometry (positions/scales) is unchanged.
        3. Restore original Gaussian parameters.
        4. Return ray-traced RGB with material appearance baked in.

        Fallback: if OptiX unavailable, use standard rasterizer with SH edits.

        Sub-modes
        ---------
        Material (SH+RT) : SH edits + OptiX ray trace  — primary path
        Original         : OptiX ray trace only, no edits
        Compare          : left = original RT, right = SH-edited RT
        """
        hide_mask = self._get_hide_mask()
        rt_submode = (dpg.get_value("_rt_submode")
                      if dpg.does_item_exist("_rt_submode") else "Material (SH+RT)")

        optix_ok = (self._optix_renderer is not None
                    and self._optix_renderer.available)

        def _rt_render():
            if optix_ok:
                try:
                    out = self._optix_renderer.render(self.camera)
                    if out is not None:
                        return out["rgb"].permute(2, 0, 1)
                except Exception as e:
                    print(f"[V8] OptiX render failed: {e}")
            return render(cam, self.engine["scene"], pipe,
                          self.bg_color, filtered_mask=hide_mask)["render"]

        def _render_original():
            return _rt_render()

        def _render_edited():
            if self._sh_editor and self.material_assignments:
                self._sh_editor.apply_all(
                    self.material_assignments,
                    self.engine["scene"]._mask,
                )
            result = _rt_render()
            if self._sh_editor and self.material_assignments:
                self._sh_editor.restore()
            return result

        if rt_submode == "Original":
            return _render_original()

        if rt_submode == "Compare":
            orig   = _render_original()
            edited = _render_edited()
            W = orig.shape[2]
            result = orig.clone()
            result[:, :, W // 2:] = edited[:, :, W // 2:]
            result[:, :, W // 2 - 1: W // 2 + 1] = 1.0
            return result

        return _render_edited()

    # ────────────────────────── fast RT fetch_data override ─────────────────

    @torch.no_grad()
    def fetch_data(self, cam):
        """
        Skip the expensive contrastive-feature render in RT mode when there is
        no pending segmentation interaction.  Falls back to the full parent
        implementation for all other view modes and when the user is actively
        clicking / running Segment 3D.
        """
        view = (dpg.get_value("_view_mode")
                if dpg.does_item_exist("_view_mode") else self.VIEW_RGB)

        needs_feature = (
            view != self.VIEW_RAYTRACING
            or len(self.new_click_xy) > 0
            or self.segment3d_flag
            or self.auto_segment_flag
            or self.sam_driven_flag
        )

        if needs_feature:
            super().fetch_data(cam)
            return

        # ── Fast RT path: skip feature render ────────────────────────────────
        import numpy as np
        pipe = type('Pipe', (), {
            'convert_SHs_python': self.opt.convert_SHs_python,
            'compute_cov3D_python': self.opt.compute_cov3D_python,
            'debug': self.opt.debug,
        })()

        hide_mask = self._get_hide_mask()

        # State mutations that don't require feature features
        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            self.prompt_num = 0
            self.hidden_segments = set()
            self.material_labels = {}
            self.material_assignments = {}
            self.segment_colors = None
            self._mat_map_cache = None
            self._mat_map_dirty = True
            try:
                self.engine["scene"].clear_segment()
                self.engine["feature"].clear_segment()
            except Exception:
                pass
            self._last_segment_times = -1
            self._dirty = True

        if self.roll_back:
            self.new_click_xy = []
            self.roll_back = False
            self.prompt_num = 0
            try:
                self.engine["scene"].roll_back()
                self.engine["feature"].roll_back()
                st = self.engine["scene"].segment_times
                self.hidden_segments = {sid for sid in self.hidden_segments if sid <= st}
            except Exception:
                pass
            self._dirty = True

        if self.confirm_hide_flag:
            self.confirm_hide_flag = False
            st = self.engine["scene"].segment_times
            if st >= 1:
                self.hidden_segments.add(st)
                self.new_click_xy = []
                self.chosen_feature = None
                self.prompt_num = 0
                self._last_segment_times = -1
            self._dirty = True

        # RT render
        img = self._render_raytracing(cam, pipe).permute(1, 2, 0)
        self.render_buffer = img.cpu().numpy().astype(np.float32)
        if dpg.does_item_exist("_texture"):
            dpg.set_value("_texture", self.render_buffer.reshape(-1))
        self._dirty = False
        self._update_material_labels()

    # ──────────────────────────── CLIP auto-detection ───────────────────────

    def _load_clip_network(self):
        """Lazy-load the CLIP text encoder (ViT-B-16, one-time ~600 MB load)."""
        if self._clip_network is None:
            from clip_utils.clip_utils import OpenCLIPNetwork, OpenCLIPNetworkConfig
            print("[V8] Loading CLIP text encoder for material detection…")
            self._clip_network = OpenCLIPNetwork(OpenCLIPNetworkConfig())
            self._clip_network.eval()
        return self._clip_network

    @torch.no_grad()
    def _auto_detect_material(self):
        """
        Cosine similarity between selected segment's mean CLIP feature vector
        and each material's text prompts.  Uses pre-computed per-Gaussian CLIP
        features (already in GPU VRAM from training) — no image re-encoding.
        """
        val = (dpg.get_value("_material_segment_select")
               if dpg.does_item_exist("_material_segment_select") else None)
        seg_id = self._parse_segment_select_value(val)
        if seg_id < 1:
            print("[V8] Select a segment first.")
            return

        scene_mask = self.engine["scene"]._mask
        if scene_mask is None:
            print("[V8] No segmentation mask available.")
            return

        feat_mask = (scene_mask == seg_id + 1)
        if not feat_mask.any():
            print(f"[V8] Segment {seg_id} has no Gaussians.")
            return

        # Per-Gaussian CLIP features — (N, C) or (N, 1, C)
        all_feats = self.engine["feature"].get_point_features
        if all_feats.dim() == 3:
            all_feats = all_feats.squeeze(1)
        seg_feats = all_feats[feat_mask].float()

        mean_feat = seg_feats.mean(0)
        mean_feat = mean_feat / (mean_feat.norm() + 1e-6)

        clip_net = self._load_clip_network()

        scores = {}
        for mat_name, prompts in MATERIAL_CLIP_PROMPTS.items():
            tokens = torch.cat(
                [clip_net.tokenizer(p) for p in prompts]
            ).to(mean_feat.device)
            text_feats = clip_net.model.encode_text(tokens).float()
            text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (mean_feat.unsqueeze(0) @ text_feats.T).mean()
            scores[mat_name] = float(sim.item())

        # Update score labels in UI
        for mat_name, score in scores.items():
            tag = f"_clip_score_{mat_name}"
            if dpg.does_item_exist(tag):
                pct = max(0, min(100, int((score + 0.5) * 100)))
                dpg.set_value(tag, f"{mat_name}: {pct}%")

        top_mat = max(scores, key=scores.get)
        self._clip_detected_material = top_mat

        # Show detected result in the apply button label
        if dpg.does_item_exist("_clip_apply_btn"):
            dpg.configure_item("_clip_apply_btn",
                               label=f"Apply  [{top_mat}]")

        self._dirty = True
        print(f"[V8] CLIP → {top_mat}  "
              f"({', '.join(f'{k}={v:.3f}' for k, v in scores.items())})")

    def _apply_clip_material(self):
        """
        Apply CLIP-determined material to the selected segment.

        Priority:
          1. Text prompt (if filled)  → keyword-parsed continuous params
          2. Auto Detect result       → discrete preset (Metal / Glass / …)
        """
        val = (dpg.get_value("_material_segment_select")
               if dpg.does_item_exist("_material_segment_select") else None)
        seg_id = self._parse_segment_select_value(val)
        if seg_id < 1:
            print("[V8] Select a segment first.")
            return

        prompt = (dpg.get_value("_clip_prompt_input").strip()
                  if dpg.does_item_exist("_clip_prompt_input") else "")

        if prompt:
            params = params_from_text_prompt(prompt)
            self.material_assignments[seg_id] = {
                "type": "Custom",
                "name": f"[CLIP] {prompt}",
                "params": params,
            }
            self.material_labels[seg_id] = f"[CLIP] {prompt}"
            self._mat_map_dirty = True
            self._dirty = True
            print(f"[V8] Seg {seg_id} ← prompt '{prompt}' → {params}")

        elif self._clip_detected_material:
            mat = self._clip_detected_material
            self.material_assignments[seg_id] = {"type": mat}
            self.material_labels[seg_id] = f"[CLIP→{mat}]"
            self._mat_map_dirty = True
            self._dirty = True
            print(f"[V8] Seg {seg_id} ← CLIP auto '{mat}'")

        else:
            print("[V8] Run Auto Detect or enter a prompt first.")

    # ──────────────────────────────────── UI sub-mode + CLIP panel override --

    def _register_dpg(self):
        """Override: update sub-mode combo and inject CLIP Material panel."""
        super()._register_dpg()

        # ── sub-mode combo ───────────────────────────────────────────────────
        if dpg.does_item_exist("_rt_submode"):
            dpg.configure_item(
                "_rt_submode",
                items=["Material (SH+RT)", "Original", "Compare"],
                default_value="Material (SH+RT)",
            )

        # ── hide unused light controls ───────────────────────────────────────
        for tag in ("_LightAz", "_LightEl", "_FillIntensity"):
            if dpg.does_item_exist(tag):
                dpg.hide_item(tag)

        # ── CLIP Material panel (inserted before _rt_group) ──────────────────
        if not dpg.does_item_exist("_rt_group"):
            return

        with dpg.collapsing_header(
            label="CLIP Material",
            default_open=False,
            before="_rt_group",
            tag="_clip_material_header",
        ):
            dpg.add_text("Select a segment, then Auto Detect or enter a prompt.",
                         color=(180, 180, 180), wrap=340)

            dpg.add_separator()
            dpg.add_text("Auto Detect", color=(180, 200, 255))
            dpg.add_button(
                label="Auto Detect",
                tag="_clip_auto_detect_btn",
                width=-1,
                callback=lambda: self._auto_detect_material(),
            )

            # One score label per material
            with dpg.group(tag="_clip_scores_group"):
                for mat_name in MATERIAL_CLIP_PROMPTS:
                    dpg.add_text(f"{mat_name}: —",
                                 tag=f"_clip_score_{mat_name}",
                                 color=(200, 200, 200))

            dpg.add_separator()
            dpg.add_text("Prompt  (e.g. 'blue glass', 'brushed metal')",
                         color=(180, 200, 255))
            dpg.add_input_text(
                hint="blue glass / brushed metal / shiny gold…",
                tag="_clip_prompt_input",
                width=-1,
            )

            dpg.add_separator()
            dpg.add_button(
                label="Apply",
                tag="_clip_apply_btn",
                width=-1,
                callback=lambda: self._apply_clip_material(),
            )


# ──────────────────────────────────────────────────────────── entry point ──

def main():
    parser = ArgumentParser(description="SAGA V8 — SH Material Viewer")
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
