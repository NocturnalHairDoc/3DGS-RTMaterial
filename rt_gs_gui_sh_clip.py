"""
rt_gs_gui_sh_clip.py — Material-SH + CLIP viewer
=======================================
Strategy 1: edit Gaussian SH coefficients in-place to simulate material
appearance, then restore after rendering.

Changes from rt_gs_gui.py
--------------------------
- Adds MaterialSHEditor (materials/) after model load.
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
    python rt_gs_gui_sh_clip.py -m ./output/bicycle --scale 1.5
"""

# Keep --help available without importing CUDA/GUI dependencies.
if __name__ == "__main__":
    import sys as _early_sys
    if any(arg in {"-h", "--help"} for arg in _early_sys.argv[1:]):
        from argparse import ArgumentParser as _EarlyArgumentParser
        _early_parser = _EarlyArgumentParser(description="3DGS SH material viewer")
        _early_parser.add_argument("-m", "--model_path", help="trained model directory")
        _early_parser.print_help()
        raise SystemExit(0)

import os
import copy
import io
import json
import numpy as np
import torch
import dearpygui.dearpygui as dpg
import imageio.v2 as imageio
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation

from rt_gs_gui import RTGSViewerGUI, RTGSConfig
from scene import GaussianModel, FeatureGaussianModel
from gaussian_renderer import render

from materials import MaterialSHEditor
from optix_integration import OptiXRenderer
from viewer.project_state import load_project_state, save_project_state
from viewer.export_manager import (ExportCancelled, ExportManager, capture_export_snapshot,
                                   compose_depth_ordered_ids, estimate_frame_bytes,
                                   linear_to_srgb)
from viewer.undo_manager import UndoManager
from viewer.render_policy import InteractiveRenderPolicy
from clip_utils.material_clip import (
    aggregate_multiview_scores,
    masked_original_rgb_crop,
    params_from_text_prompt,
    score_crop_against_material_prompts,
    select_confident_material,
    topk_materials,
    MATERIAL_PARAM_PRESETS,
)
from viewer.utils import scoped_output_path, visibility_cache_key


class SHMaterialViewer(RTGSViewerGUI):
    """RTGSViewerGUI subclass that uses SH coefficient editing for materials."""

    # ------------------------------------------------------------------ init --

    def __init__(self, opt, gs_model, feat_model, scale_gate):
        # CLIP auto-detection state — must exist before _register_dpg is called
        self._clip_network = None
        self._clip_detected_material = None
        self._visibility_cache_key = None
        self._visibility_cache = None
        self._export_manager = ExportManager()
        self._export_renderer_cache = {}
        self._material_clipboard = None
        self._undo_manager = UndoManager(limit=24)
        self._restoring_history = False
        self._history_signature = None
        self._pending_state_load = None
        self._interactive_rt_preview = InteractiveRenderPolicy()

        # Parent __init__ loads model, builds BVH, creates DearPyGui context,
        # and calls self._register_dpg() via dynamic dispatch.
        super().__init__(opt, gs_model, feat_model, scale_gate)

        # Attach the SH editor now that the model is loaded into GPU.
        self._sh_editor = MaterialSHEditor(self.engine["scene"])
        self._observe_history(force=True)
        print("[V2] MaterialSHEditor ready — SH-based material rendering active.")

    def _visible_gaussian_mask(self):
        """Return a cached OptiX inclusion mask for the current hidden segments."""
        scene_mask = self.engine["scene"]._mask
        key = visibility_cache_key(
            self.hidden_segments, scene_mask, self.engine["scene"].segment_times)
        if key == self._visibility_cache_key:
            return self._visibility_cache
        hide = self._get_hide_mask()
        self._visibility_cache = None if hide is None else (~hide).contiguous()
        self._visibility_cache_key = key
        return self._visibility_cache

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
        interactive_preview = self._interactive_rt_preview.use_raster(
            self.moving, self.moving_middle)

        def _rt_render():
            if optix_ok and not interactive_preview:
                try:
                    out = self._optix_renderer.render(
                        # Use the exact SAGA Camera consumed by rasterization.
                        cam,
                        segment_mask=self._visible_gaussian_mask()
                    )
                    if out is not None:
                        return out["rgb"].permute(2, 0, 1)
                except Exception as e:
                    print(f"[V2] OptiX render failed: {e}")
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
            try:
                return _rt_render()
            finally:
                # A renderer exception must never leave the trainable scene
                # permanently modified by a temporary material preview.
                if self._sh_editor and self.material_assignments:
                    self._sh_editor.restore()

        # During camera motion a single material-aware raster pass keeps the
        # viewport responsive. Mouse release marks the view dirty, causing a
        # full-quality OptiX settle frame immediately afterwards.
        if interactive_preview:
            return _render_original() if rt_submode == "Original" else _render_edited()

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

    # ───────────────────────── project state persistence ────────────────────

    def _state_metadata(self):
        values = {}
        for tag in ("_view_mode", "_rt_submode", "_Scale", "_ScoreThres"):
            if dpg.does_item_exist(tag):
                values[tag] = dpg.get_value(tag)
        return {
            "model_path": os.path.abspath(self.opt.MODEL_PATH),
            "model_name": os.path.basename(os.path.normpath(self.opt.MODEL_PATH)),
            "scene_iteration": int(self.opt.SCENE_GAUSSIAN_ITERATION),
            "feature_iteration": int(self.opt.FEATURE_GAUSSIAN_ITERATION),
            "point_count": int(self.engine["scene"].get_xyz.shape[0]),
            "segment_times": int(self.engine["scene"].segment_times),
            "material_labels": {str(k): v for k, v in self.material_labels.items()},
            "material_assignments": {str(k): v for k, v in self.material_assignments.items()},
            "hidden_segments": sorted(int(v) for v in self.hidden_segments),
            "camera": {
                "rotation_xyzw": self.camera.rot.as_quat().tolist(),
                "center": self.camera.center.tolist(),
                "radius": float(self.camera.radius),
                "fovy": float(self.camera.fovy),
                "rot_mode": int(self.camera.rot_mode),
            },
            "ui": values,
        }

    @staticmethod
    def _scoped_path(path, directory):
        return scoped_output_path(path, directory, __file__)

    def _save_project(self):
        path = dpg.get_value("_project_state_path").strip()
        try:
            path = self._scoped_path(path, "segmentation_res")
            saved = save_project_state(
                path,
                self.engine["scene"]._mask.detach().cpu().numpy(),
                self._state_metadata(),
            )
            self._set_io_status(f"Saved project: {saved}")
        except Exception as exc:
            self._set_io_status(f"Save failed: {exc}", error=True)

    def _load_project(self):
        path = dpg.get_value("_project_state_path").strip()
        try:
            path = self._scoped_path(path, "segmentation_res")
            mask, metadata = load_project_state(path)
            absolute_path = os.path.abspath(path)
            if self._pending_state_load != absolute_path:
                self._pending_state_load = absolute_path
                if dpg.does_item_exist("_load_project_button"):
                    dpg.configure_item("_load_project_button", label="Confirm load")
                self._set_io_status(
                    f"State summary: model={metadata.get('model_name', '?')}, "
                    f"points={mask.shape[0]}, saved={metadata.get('saved_at', '?')}, "
                    f"materials={len(metadata.get('material_assignments', {}))}. "
                    "Click Confirm load to continue.")
                return
            self._pending_state_load = None
            if dpg.does_item_exist("_load_project_button"):
                dpg.configure_item("_load_project_button", label="Load project")
            scene = self.engine["scene"]
            expected = int(scene.get_xyz.shape[0])
            if int(metadata.get("point_count", -1)) != expected or mask.shape[0] != expected:
                raise ValueError(f"state has {mask.shape[0]} points, current model has {expected}")
            current_name = os.path.basename(os.path.normpath(self.opt.MODEL_PATH))
            if metadata.get("model_name", current_name) != current_name:
                raise ValueError(
                    f"state belongs to model {metadata.get('model_name')!r}, not {current_name!r}"
                )
            mask_gpu = torch.from_numpy(mask).to(device="cuda", dtype=scene._mask.dtype)
            scene._mask = mask_gpu.clone()
            scene.segment_times = int(metadata.get("segment_times", max(0, int(mask.max()) - 1)))
            scene.old_mask = []
            feature = self.engine["feature"]
            if feature.get_xyz.shape[0] == expected:
                feature._mask = mask_gpu.clone()
                feature.segment_times = scene.segment_times
                feature.old_mask = []
            self.material_labels = {int(k): str(v) for k, v in metadata.get("material_labels", {}).items()}
            self.material_assignments = {int(k): v for k, v in metadata.get("material_assignments", {}).items()}
            for segment_id, assignment in self.material_assignments.items():
                assignment.setdefault("type", "Default")
                assignment.setdefault("name", self.material_labels.get(segment_id, f"Material_{segment_id}"))
            self.hidden_segments = {int(v) for v in metadata.get("hidden_segments", [])}
            camera = metadata.get("camera", {})
            self.camera.rot = Rotation.from_quat(camera.get("rotation_xyzw", [0, 0, 0, 1]))
            self.camera.center = np.asarray(camera.get("center", [0, 0, 0]), dtype=np.float32)
            self.camera.radius = float(camera.get("radius", self.camera.radius))
            self.camera.fovy = float(camera.get("fovy", self.camera.fovy))
            self.camera.rot_mode = int(camera.get("rot_mode", self.camera.rot_mode))
            for tag, value in metadata.get("ui", {}).items():
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, value)
            if dpg.does_item_exist("_rt_group") and dpg.does_item_exist("_view_mode"):
                if dpg.get_value("_view_mode") == self.VIEW_RAYTRACING:
                    dpg.show_item("_rt_group")
                else:
                    dpg.hide_item("_rt_group")
            self._visibility_cache_key = None
            self._mat_map_dirty = True
            self._last_segment_times = -1
            self._dirty = True
            self._update_material_labels()
            self._set_io_status(f"Loaded project: {path}")
            self._observe_history(force=True)
        except Exception as exc:
            self._set_io_status(f"Load failed: {exc}", error=True)

    def _set_io_status(self, message, error=False):
        print(f"[V2] {message}")
        if dpg.does_item_exist("_project_io_status"):
            dpg.set_value("_project_io_status", message)
            dpg.configure_item("_project_io_status", color=(255, 110, 110) if error else (120, 220, 140))

    # ───────────────────────────── undo / redo ──────────────────────────────

    def _history_state(self):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, mask=self.engine["scene"]._mask.detach().cpu().numpy())
        return {
            "mask_npz": buffer.getvalue(),
            "segment_times": int(self.engine["scene"].segment_times),
            "materials": copy.deepcopy(self.material_assignments),
            "labels": copy.deepcopy(self.material_labels),
            "hidden": sorted(self.hidden_segments),
            "camera": copy.deepcopy(self.camera),
        }

    def _current_history_signature(self):
        scene = self.engine["scene"]
        pose = self.camera.pose_movecenter
        return (
            scene._mask.data_ptr(), getattr(scene._mask, "_version", 0),
            int(scene.segment_times), json.dumps(self.material_assignments, sort_keys=True),
            tuple(sorted(self.hidden_segments)), tuple(np.round(pose.ravel(), 5)),
            round(float(self.camera.radius), 5),
        )

    def _observe_history(self, force=False):
        if self._restoring_history:
            return
        signature = self._current_history_signature()
        if force or signature != self._history_signature:
            camera_only = (self._history_signature is not None
                           and signature[:5] == self._history_signature[:5])
            self._undo_manager.record(self._history_state(), key="camera" if camera_only else None)
            self._history_signature = signature
            self._update_history_buttons()

    def _restore_history_state(self, state):
        if state is None:
            return
        self._restoring_history = True
        try:
            with np.load(io.BytesIO(state["mask_npz"]), allow_pickle=False) as archive:
                mask = torch.from_numpy(archive["mask"]).to("cuda")
            scene = self.engine["scene"]
            scene._mask = mask.to(dtype=scene._mask.dtype)
            scene.segment_times = state["segment_times"]
            feature = self.engine["feature"]
            if feature.get_xyz.shape[0] == mask.shape[0]:
                feature._mask = scene._mask.clone()
                feature.segment_times = scene.segment_times
            self.material_assignments = copy.deepcopy(state["materials"])
            self.material_labels = copy.deepcopy(state["labels"])
            self.hidden_segments = set(state["hidden"])
            self.camera = copy.deepcopy(state["camera"])
            self._visibility_cache_key = None
            self._mat_map_dirty = True
            self._dirty = True
            self._update_material_labels()
        finally:
            self._restoring_history = False
        self._history_signature = self._current_history_signature()
        self._update_history_buttons()

    def _undo(self):
        self._restore_history_state(self._undo_manager.undo())

    def _redo(self):
        self._restore_history_state(self._undo_manager.redo())

    def _history_key(self, key):
        if dpg.is_key_down(dpg.mvKey_Control):
            if key == dpg.mvKey_Z: self._undo()
            elif key == dpg.mvKey_Y: self._redo()

    def _update_history_buttons(self):
        for tag, enabled in (("_undo_button", self._undo_manager.can_undo),
                             ("_redo_button", self._undo_manager.can_redo)):
            if dpg.does_item_exist(tag): dpg.configure_item(tag, enabled=enabled)

    # ─────────────────────────────── export ─────────────────────────────────

    def _new_export_renderer(self, width, height, scene=None):
        renderer = OptiXRenderer(
            self.engine["scene"] if scene is None else scene, width=width, height=height,
            sh_degree=self.opt.sh_degree, bg_color=self.bg_color,
        )
        if not renderer.available:
            raise RuntimeError("OptiX is required for high-resolution export")
        return renderer

    @torch.no_grad()
    def _render_export_output(self, renderer, snapshot, editor,
                              apply_materials=True, segment_mask=None):
        edited = apply_materials and bool(snapshot.material_assignments)
        if edited:
            editor.apply_all(snapshot.material_assignments, snapshot.scene_mask)
        try:
            mask = snapshot.visible_mask if segment_mask is None else segment_mask
            if renderer.width * renderer.height > 1920 * 1080:
                output = renderer.render_tiled(snapshot.camera, tile_size=1024, segment_mask=mask)
            else:
                output = renderer.render(snapshot.camera, segment_mask=mask)
            if output is None:
                raise RuntimeError("OptiX returned no frame")
            if any(not torch.isfinite(value).all() for value in output.values()):
                raise RuntimeError("export output contains invalid values")
            return {key: value.float().cpu().numpy() for key, value in output.items()}
        finally:
            if edited:
                editor.restore()

    def _render_export_frame(self, renderer, snapshot, editor, camera):
        frame_snapshot = copy.copy(snapshot)
        object.__setattr__(frame_snapshot, "camera", camera)
        return self._render_export_output(renderer, frame_snapshot, editor)["rgb"].clip(0, 1)

    def _render_export_ids(self, renderer, snapshot, editor, material=False):
        scene_mask = snapshot.scene_mask
        material_ids = {sid: index + 1 for index, sid in enumerate(sorted(snapshot.material_assignments))}
        layers = []
        for encoded in sorted(int(v) for v in torch.unique(scene_mask).tolist() if int(v) >= 2):
            segment_id = encoded - 1
            if snapshot.visible_mask is not None and not bool(snapshot.visible_mask[scene_mask == encoded].any()):
                continue
            value = material_ids.get(segment_id, 0) if material else segment_id
            if value == 0:
                continue
            output = self._render_export_output(
                renderer, snapshot, editor, apply_materials=False,
                segment_mask=(scene_mask == encoded))
            layers.append((value, output))
        return compose_depth_ordered_ids(layers, renderer.height, renderer.width)

    def _selected_export_channel(self):
        return dpg.get_value("_export_channel") if dpg.does_item_exist("_export_channel") else "RGB"

    def _write_export_image(self, path, renderer, snapshot, editor, channel):
        if channel == "Segmentation ID":
            data = self._render_export_ids(renderer, snapshot, editor, material=False)
        elif channel == "Material ID":
            data = self._render_export_ids(renderer, snapshot, editor, material=True)
        else:
            output = self._render_export_output(renderer, snapshot, editor)
        if channel == "RGB":
            data = (linear_to_srgb(output["rgb"].clip(0, 1)) * 255 + 0.5).astype(np.uint8)
        elif channel == "RGBA":
            rgb = (linear_to_srgb(output["rgb"].clip(0, 1)) * 255 + 0.5).astype(np.uint8)
            alpha = (output["opacity"].clip(0, 1) * 255 + 0.5).astype(np.uint8)
            data = np.concatenate([rgb, alpha], axis=-1)
        elif channel == "Normals":
            data = ((output["normals"].clip(-1, 1) * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)
        elif channel == "Depth":
            path = os.path.splitext(path)[0] + ".tiff"
            data = output["depth"].astype(np.float32)
        elif channel in {"Segmentation ID", "Material ID"}:
            pass
        elif channel == "Original / Edited":
            original = self._render_export_output(
                renderer, snapshot, editor, apply_materials=False)["rgb"]
            comparison = original.copy()
            comparison[:, renderer.width // 2:] = output["rgb"][:, renderer.width // 2:]
            data = (linear_to_srgb(comparison.clip(0, 1)) * 255 + 0.5).astype(np.uint8)
        else:
            raise ValueError(f"unsupported export channel: {channel}")
        imageio.imwrite(path, data)
        return path

    def _export_dimensions(self):
        width = int(dpg.get_value("_export_width"))
        height = int(dpg.get_value("_export_height"))
        if width < 16 or height < 16 or width > 8192 or height > 8192:
            raise ValueError("export dimensions must be between 16 and 8192")
        return width, height

    def _export_image(self):
        try:
            width, height = self._export_dimensions()
            path = dpg.get_value("_export_image_path").strip()
            if not path.lower().endswith(".png"):
                path += ".png"
            path = self._scoped_path(path, "exports")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            estimate_mib = estimate_frame_bytes(width, height) / 1024**2
            self._set_io_status(f"Queued PNG {width}×{height}; estimated working buffers {estimate_mib:.0f} MiB")
            snapshot = capture_export_snapshot(
                self.engine["scene"], self.camera, self.material_assignments,
                self.hidden_segments)
            channel = self._selected_export_channel()
            self._clip_network = None
            torch.cuda.empty_cache()
            def work(progress, cancelled):
                renderer = self._new_export_renderer(width, height, snapshot.scene)
                renderer.build_bvh()
                editor = MaterialSHEditor(snapshot.scene)
                if cancelled.is_set(): raise ExportCancelled("Export cancelled")
                self._write_export_image(path, renderer, snapshot, editor, channel)
                progress(1)
            self._export_manager.start(1, work)
        except Exception as exc:
            message = str(exc)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                message = (f"CUDA out of memory at {width}×{height}; try half the width/height "
                           "or enable tiled export")
            self._set_io_status(f"Image export failed: {message}", error=True)

    def _export_video(self):
        try:
            width, height = self._export_dimensions()
            frames = int(dpg.get_value("_export_frames"))
            fps = int(dpg.get_value("_export_fps"))
            if frames < 2 or frames > 1440 or fps < 1 or fps > 120:
                raise ValueError("frames must be 2–1440 and FPS 1–120")
            if width % 2 or height % 2:
                raise ValueError("MP4 width and height must be even")
            path = dpg.get_value("_export_video_path").strip()
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            path = self._scoped_path(path, "exports")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            estimate_mib = estimate_frame_bytes(width, height) / 1024**2
            self._set_io_status(f"Queued MP4 {width}×{height}; estimated working buffers {estimate_mib:.0f} MiB")
            snapshot = capture_export_snapshot(
                self.engine["scene"], self.camera, self.material_assignments,
                self.hidden_segments)
            base = snapshot.camera
            self._clip_network = None
            torch.cuda.empty_cache()
            write_sequence = bool(dpg.get_value("_export_png_sequence"))
            sequence_dir = os.path.splitext(path)[0] + "_frames"
            def work(progress, cancelled):
                renderer = self._new_export_renderer(width, height, snapshot.scene)
                renderer.build_bvh()
                editor = MaterialSHEditor(snapshot.scene)
                if write_sequence: os.makedirs(sequence_dir, exist_ok=True)
                with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                        macro_block_size=None) as writer:
                    for index in range(frames):
                        if cancelled.is_set(): raise ExportCancelled("Export cancelled")
                        camera = copy.deepcopy(base)
                        angle = 2.0 * np.pi * index / frames
                        camera.rot = Rotation.from_rotvec(np.array([0.0, angle, 0.0])) * base.rot
                        linear = self._render_export_frame(renderer, snapshot, editor, camera)
                        writer.append_data((linear_to_srgb(linear) * 255 + 0.5).astype(np.uint8))
                        if write_sequence:
                            imageio.imwrite(os.path.join(sequence_dir, f"frame_{index:06d}.png"),
                                          (linear_to_srgb(linear) * 255 + 0.5).astype(np.uint8))
                        progress(index + 1)
            self._export_manager.start(frames, work)
        except Exception as exc:
            self._set_io_status(f"Video export failed: {exc}", error=True)

    def _cached_export_renderer(self, width, height):
        key = (width, height)
        renderer = self._export_renderer_cache.get(key)
        if renderer is None:
            renderer = self._new_export_renderer(width, height)
            renderer.build_bvh()
            self._export_renderer_cache[key] = renderer
        return renderer

    def _cancel_export(self):
        self._export_manager.cancel()

    def _poll_export_events(self):
        for event in self._export_manager.drain_events():
            running = event.kind in {"started", "progress"}
            for tag in ("_export_png_button", "_export_video_button"):
                if dpg.does_item_exist(tag): dpg.configure_item(tag, enabled=not running)
            if dpg.does_item_exist("_export_cancel_button"):
                dpg.configure_item("_export_cancel_button", enabled=running)
            if dpg.does_item_exist("_export_progress") and event.total:
                dpg.set_value("_export_progress", event.current / event.total)
            if event.kind == "progress":
                eta = "" if event.eta_seconds is None else f", ETA {event.eta_seconds:.1f}s"
                self._set_io_status(f"Export {event.current}/{event.total}{eta}")
            elif event.message:
                message = event.message
                if event.kind == "failed" and ("OutOfMemory" in message or "out of memory" in message.lower()):
                    message += "; try half resolution, a smaller tile, or close CLIP and retry"
                self._set_io_status(message, error=event.kind == "failed")

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
            self._observe_history()
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
                self._prune_segment_state()
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
        self._observe_history()

    # ─────────────────────────── material parameter controls ───────────────

    def _material_params_from_ui(self):
        return {"tint": [float(v) for v in dpg.get_value("_mat_tint")[:3]],
                "strength": float(dpg.get_value("_mat_strength")),
                "saturation": float(dpg.get_value("_mat_saturation")),
                "specular_gain": float(dpg.get_value("_mat_sh_gain")),
                "opacity_scale": float(dpg.get_value("_mat_opacity"))}

    def _set_material_params_ui(self, params):
        values = {"tint": [1, 1, 1], "strength": 1, "saturation": 1,
                  "specular_gain": 1, "opacity_scale": 1, **(params or {})}
        mapping = {"_mat_tint": values["tint"], "_mat_strength": values["strength"],
                   "_mat_saturation": values["saturation"], "_mat_sh_gain": values["specular_gain"],
                   "_mat_opacity": values["opacity_scale"]}
        for tag, value in mapping.items():
            if dpg.does_item_exist(tag): dpg.set_value(tag, value)

    def _on_material_segment_select(self, value):
        super()._on_material_segment_select(value)
        info = self.material_assignments.get(self._parse_segment_select_value(value), {})
        preset = info.get("preset", info.get("type", "Default"))
        if preset not in MATERIAL_PARAM_PRESETS: preset = "Default"
        if dpg.does_item_exist("_material_type_select"):
            dpg.set_value("_material_type_select", preset)
        self._set_material_params_ui(info.get("params") or MATERIAL_PARAM_PRESETS.get(preset))

    def _replace_material_assignment(self, segment_id, assignment):
        """Replace SH settings without discarding renderer-specific fields."""
        previous = self.material_assignments.get(segment_id, {})
        merged = {key: copy.deepcopy(value) for key, value in previous.items()
                  if key not in {"type", "preset", "name", "params"}}
        merged.update(assignment)
        self.material_assignments[segment_id] = merged

    def _apply_material_assignment(self):
        seg_id = self._parse_segment_select_value(dpg.get_value("_material_segment_select"))
        if seg_id < 1: return
        name = str(dpg.get_value("_material_name_input") or f"Material_{seg_id}").strip()
        preset = dpg.get_value("_material_type_select")
        self._replace_material_assignment(
            seg_id,
            {"type": "Custom", "preset": preset, "name": name,
             "params": self._material_params_from_ui()},
        )
        self.material_labels[seg_id] = name
        self._mat_map_dirty = self._dirty = True
        self._observe_history()

    def _reset_material_params(self):
        preset = dpg.get_value("_material_type_select")
        self._set_material_params_ui(MATERIAL_PARAM_PRESETS.get(preset))
        if dpg.get_value("_mat_live_preview"): self._apply_material_assignment()

    def _copy_material_params(self):
        self._material_clipboard = copy.deepcopy(self._material_params_from_ui())

    def _paste_material_params(self):
        if self._material_clipboard is not None:
            self._set_material_params_ui(copy.deepcopy(self._material_clipboard))
            if dpg.get_value("_mat_live_preview"): self._apply_material_assignment()

    def _material_param_changed(self):
        if dpg.get_value("_mat_live_preview"): self._apply_material_assignment()

    def _set_adaptive_rt_preview(self, enabled):
        self._interactive_rt_preview.enabled = bool(enabled)
        self._dirty = True
        mode = "adaptive raster interaction" if enabled else "full OptiX interaction"
        self._set_io_status(f"RT preview mode: {mode}")

    # ──────────────────────────── CLIP auto-detection ───────────────────────

    def _load_clip_network(self):
        """Lazy-load the CLIP text encoder (ViT-B-16, one-time ~600 MB load)."""
        if self._clip_network is None:
            from clip_utils.clip_utils import OpenCLIPNetwork, OpenCLIPNetworkConfig
            print("[V2] Loading CLIP text encoder for material detection…")
            self._clip_network = OpenCLIPNetwork(OpenCLIPNetworkConfig())
            self._clip_network.eval()
        return self._clip_network

    def _construct_camera(self, orbit_camera=None):
        """Build a raster camera from an explicit orbit without mutating GUI state."""
        original = self.camera
        if orbit_camera is None:
            return super()._construct_camera()
        try:
            self.camera = orbit_camera
            return super()._construct_camera()
        finally:
            self.camera = original

    @torch.no_grad()
    def _auto_detect_material(self):
        """
        Render the selected segment into a screen-space mask, crop its current
        RGB appearance, and compare the crop's OpenCLIP image embedding with
        the material text prompts.

        The learned per-Gaussian SAGA features are 32-D and cannot be compared
        directly with OpenCLIP's 512-D text embeddings.  Using an image crop
        keeps both sides in the same embedding space.
        """
        val = (dpg.get_value("_material_segment_select")
               if dpg.does_item_exist("_material_segment_select") else None)
        seg_id = self._parse_segment_select_value(val)
        if seg_id < 1:
            print("[V2] Select a segment first.")
            return

        scene_mask = self.engine["scene"]._mask
        if scene_mask is None:
            print("[V2] No segmentation mask available.")
            return

        feat_mask = (scene_mask == seg_id + 1)
        if not feat_mask.any():
            print(f"[V2] Segment {seg_id} has no Gaussians.")
            return

        try:
            self._set_io_status("Loading CLIP (uses HF_TOKEN when configured)…")
            clip_net = self._load_clip_network()
        except Exception as exc:
            self._clip_network = None
            self._set_io_status(
                f"CLIP unavailable/offline: {exc}. Check network/cache or HF_TOKEN, then retry.",
                error=True,
            )
            return

        pipe = type('Pipe', (), {
            'convert_SHs_python': self.opt.convert_SHs_python,
            'compute_cov3D_python': self.opt.compute_cov3D_python,
            'debug': self.opt.debug,
        })()
        # Render both inputs independently of the current GUI view. This avoids
        # classifying segmentation colours or an already edited material frame.
        self._sh_editor.restore()
        point_mask_color = feat_mask.float().unsqueeze(1).expand(-1, 3)
        mask_bg = torch.zeros(3, device="cuda", dtype=torch.float32)
        per_view_scores = []
        # Three independent views centred on the current view. The GUI camera
        # itself is never changed, and edited/material buffers are never read.
        for yaw_degrees in (-10.0, 0.0, 10.0):
            orbit = copy.deepcopy(self.camera)
            orbit.rot = Rotation.from_rotvec(np.array([0.0, np.radians(yaw_degrees), 0.0])) * orbit.rot
            cam = self._construct_camera(orbit)
            original_rgb = render(cam, self.engine["scene"], pipe, self.bg_color)["render"]
            mask_image = render(
                cam, self.engine["scene"], pipe, mask_bg,
                override_color=point_mask_color,
            )["render"][0]
            try:
                crop = masked_original_rgb_crop(original_rgb, mask_image).unsqueeze(0)
            except ValueError:
                continue
            per_view_scores.append(score_crop_against_material_prompts(clip_net, crop))
        if not per_view_scores:
            self._set_io_status(f"Segment {seg_id} is not visible in sampled views", error=True)
            return

        scores = aggregate_multiview_scores(per_view_scores)

        # Update score labels in UI
        for mat_name, score in scores.items():
            tag = f"_clip_score_{mat_name}"
            if dpg.does_item_exist(tag):
                pct = max(0, min(100, int((score + 0.5) * 100)))
                dpg.set_value(tag, f"{mat_name}: {pct}%")

        top_mat, margin = select_confident_material(scores)
        self._clip_detected_material = top_mat

        # Show detected result in the apply button label
        if dpg.does_item_exist("_clip_apply_btn"):
            label = (f"Confirm [{top_mat}] Δ={margin:.3f}" if top_mat else
                     f"Low confidence Δ={margin:.3f}")
            dpg.configure_item("_clip_apply_btn", label=label, enabled=top_mat is not None)

        self._dirty = True
        print(f"[V2] CLIP Top-K: {topk_materials(scores, 3)}; margin={margin:.3f}; "
              f"views={len(per_view_scores)}; candidate={top_mat or 'none'}")

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
            print("[V2] Select a segment first.")
            return

        prompt = (dpg.get_value("_clip_prompt_input").strip()
                  if dpg.does_item_exist("_clip_prompt_input") else "")

        if prompt:
            params = params_from_text_prompt(prompt)
            self._replace_material_assignment(seg_id, {
                "type": "Custom",
                "name": f"[CLIP] {prompt}",
                "params": params,
            })
            self.material_labels[seg_id] = f"[CLIP] {prompt}"
            self._mat_map_dirty = True
            self._dirty = True
            self._observe_history()
            print(f"[V2] Seg {seg_id} ← prompt '{prompt}' → {params}")

        elif self._clip_detected_material:
            mat = self._clip_detected_material
            self._replace_material_assignment(
                seg_id, {"type": mat, "name": f"[CLIP→{mat}]"})
            self.material_labels[seg_id] = f"[CLIP→{mat}]"
            self._mat_map_dirty = True
            self._dirty = True
            self._observe_history()
            print(f"[V2] Seg {seg_id} ← CLIP auto '{mat}'")

        else:
            print("[V2] Run Auto Detect or enter a prompt first.")

    # ──────────────────────────────────── UI sub-mode + CLIP panel override --

    def _register_dpg(self):
        """Override: update sub-mode combo and inject CLIP Material panel."""
        super()._register_dpg()

        # Reuse the parent's material selector as the single source of truth.
        # It now selects a preset, initializes the parameter controls and is
        # also the value persisted by _apply_material_assignment().
        if dpg.does_item_exist("_material_type_select"):
            dpg.configure_item(
                "_material_type_select",
                label="Material preset",
                items=["Default", *MATERIAL_PARAM_PRESETS.keys()],
                default_value="Default",
                callback=lambda s, v: self._reset_material_params(),
            )
        dpg.add_separator(parent="_material_header")
        dpg.add_input_floatx(parent="_material_header", tag="_mat_tint", size=3,
                             default_value=[1.0, 1.0, 1.0], label="Tint",
                             callback=lambda: self._material_param_changed())
        for label, tag, default, minimum, maximum in (
            ("Strength", "_mat_strength", 1.0, 0.0, 1.0),
            ("Saturation", "_mat_saturation", 1.0, 0.0, 2.5),
            ("Higher-order SH gain", "_mat_sh_gain", 1.0, 0.0, 3.0),
            ("Opacity scale", "_mat_opacity", 1.0, 0.0, 1.5)):
            dpg.add_slider_float(parent="_material_header", tag=tag, label=label,
                                 default_value=default, min_value=minimum, max_value=maximum,
                                 callback=lambda: self._material_param_changed())
        dpg.add_checkbox(parent="_material_header", tag="_mat_live_preview",
                         label="Live preview", default_value=True)
        with dpg.group(parent="_material_header", horizontal=True):
            dpg.add_button(label="Reset preset", callback=lambda: self._reset_material_params())
            dpg.add_button(label="Copy", callback=lambda: self._copy_material_params())
            dpg.add_button(label="Paste", callback=lambda: self._paste_material_params())

        # ── sub-mode combo ───────────────────────────────────────────────────
        if dpg.does_item_exist("_rt_submode"):
            dpg.configure_item(
                "_rt_submode",
                items=["Material (SH+RT)", "Original", "Compare"],
                default_value="Material (SH+RT)",
            )
        with dpg.collapsing_header(parent="_rt_group", label="Interactive RT preview",
                                   default_open=True, tag="_interactive_rt_header"):
            dpg.add_checkbox(
                label="Adaptive RT preview",
                default_value=self._interactive_rt_preview.enabled,
                callback=lambda s, value: self._set_adaptive_rt_preview(value),
            )
            dpg.add_text("Raster while moving; full OptiX after release.",
                         color=(170, 190, 210), wrap=330)
        # Parent mouse handlers update the moving flags. This additional handler
        # guarantees a final full-quality frame even when the pose did not change
        # between the last move event and button release.
        with dpg.handler_registry():
            dpg.add_mouse_release_handler(
                callback=lambda s, a: setattr(self, "_dirty", True))

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

        # ── complete project state + offline export ────────────────────────
        scene_name = os.path.basename(os.path.normpath(self.opt.MODEL_PATH))
        with dpg.collapsing_header(
            label="Project & Export", default_open=False,
            before="_rt_group", tag="_project_export_header",
        ):
            dpg.add_text("Segmentation + materials + hidden segments + camera",
                         color=(180, 180, 180), wrap=340)
            dpg.add_input_text(
                tag="_project_state_path", width=-1,
                default_value=f"./segmentation_res/{scene_name}_project.npz",
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save project", callback=lambda: self._save_project())
                dpg.add_button(label="Load project", tag="_load_project_button",
                               callback=lambda: self._load_project())
            with dpg.group(horizontal=True):
                dpg.add_button(label="Undo (Ctrl+Z)", tag="_undo_button", enabled=False,
                               callback=lambda: self._undo())
                dpg.add_button(label="Redo (Ctrl+Y)", tag="_redo_button", enabled=False,
                               callback=lambda: self._redo())

            dpg.add_separator()
            dpg.add_text("High-resolution export", color=(180, 200, 255))
            dpg.add_combo(label="Channel", tag="_export_channel", width=-1,
                          items=["RGB", "RGBA", "Depth", "Normals", "Segmentation ID",
                                 "Material ID", "Original / Edited"], default_value="RGB")
            with dpg.group(horizontal=True):
                dpg.add_input_int(label="W", tag="_export_width", width=145,
                                  default_value=1920, min_value=16, max_value=8192)
                dpg.add_input_int(label="H", tag="_export_height", width=145,
                                  default_value=1080, min_value=16, max_value=8192)
            dpg.add_input_text(tag="_export_image_path", width=-1,
                               default_value=f"./exports/{scene_name}.png")
            dpg.add_button(label="Export PNG", width=-1, tag="_export_png_button",
                           callback=lambda: self._export_image())

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_input_int(label="Frames", tag="_export_frames", width=145,
                                  default_value=120, min_value=2, max_value=1440)
                dpg.add_input_int(label="FPS", tag="_export_fps", width=145,
                                  default_value=30, min_value=1, max_value=120)
            dpg.add_input_text(tag="_export_video_path", width=-1,
                               default_value=f"./exports/{scene_name}_turntable.mp4")
            dpg.add_checkbox(label="Also write PNG sequence", tag="_export_png_sequence",
                             default_value=False)
            dpg.add_button(label="Export turntable MP4", width=-1, tag="_export_video_button",
                           callback=lambda: self._export_video())
            dpg.add_progress_bar(tag="_export_progress", default_value=0.0, width=-1)
            dpg.add_button(label="Cancel export", width=-1, tag="_export_cancel_button",
                           enabled=False, callback=lambda: self._cancel_export())
            dpg.add_text("Ready", tag="_project_io_status",
                         color=(160, 160, 160), wrap=340)
        with dpg.handler_registry():
            dpg.add_key_press_handler(key=dpg.mvKey_Z,
                                      callback=lambda s, a: self._history_key(dpg.mvKey_Z))
            dpg.add_key_press_handler(key=dpg.mvKey_Y,
                                      callback=lambda s, a: self._history_key(dpg.mvKey_Y))


# ──────────────────────────────────────────────────────────── entry point ──

def main():
    parser = ArgumentParser(description="3DGS-RTMaterial V2 — SH Material Viewer")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="Trained scene directory. If omitted, use the first valid scene in ./output.")
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
    model_path = args.model_path
    if model_path is None:
        output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        candidates = []
        if os.path.isdir(output_root):
            for entry in sorted(os.listdir(output_root)):
                candidate = os.path.join(output_root, entry)
                required = (
                    os.path.join(candidate, "point_cloud", f"iteration_{args.scene_iteration}", "scene_point_cloud.ply"),
                    os.path.join(candidate, "point_cloud", f"iteration_{args.feature_iteration}", "contrastive_feature_point_cloud.ply"),
                    os.path.join(candidate, "point_cloud", f"iteration_{args.feature_iteration}", "scale_gate.pt"),
                )
                if all(os.path.isfile(path) for path in required):
                    candidates.append(candidate)
        if not candidates:
            parser.error("no valid trained scene found under ./output; pass -m MODEL_PATH")
        model_path = candidates[0]
        print(f"No model path supplied; using {model_path}")
    opt.MODEL_PATH = model_path
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
