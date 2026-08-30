# Ray-Tracing 3D Gaussian Viewer (Modular)
# Based on SAGA GUI + "Ray-Tracing 3D Gaussian" proposal.
# Phase 1: Baseline viewer with RGB, Segmentation, Material views.
# Phase 2: Ray-Tracing view with depth-based normals and Blinn-Phong material shading.

# Keep --help CPU-safe: CI and users can inspect the entry point before CUDA
# extensions and GUI dependencies are installed.
if __name__ == "__main__":
    import sys as _early_sys
    if any(arg in {"-h", "--help"} for arg in _early_sys.argv[1:]):
        from argparse import ArgumentParser as _EarlyArgumentParser
        _early_parser = _EarlyArgumentParser(description="3DGS ray-tracing viewer")
        _early_parser.add_argument("-m", "--model_path", help="trained model directory")
        _early_parser.print_help()
        raise SystemExit(0)

import torch
import os
import math
import numpy as np
import dearpygui.dearpygui as dpg
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from viewer.gui import REPOSITORY_ROOT

try:
    from sklearn.cluster import MiniBatchKMeans as KMeans  # faster for large scenes
except ImportError:
    from sklearn.cluster import KMeans

try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from scene import GaussianModel, FeatureGaussianModel
from gaussian_renderer import render, render_contrastive_feature
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal
from segmentation.membership_io import load_gaussian_membership
from viewer.scene_import import build_proxy_features

# Optional: depth rasterizer for ray-tracing view
try:
    from gaussian_renderer import render_with_depth
    DEPTH_RENDER_AVAILABLE = True
except Exception:
    DEPTH_RENDER_AVAILABLE = False

# Optional: OptiX 3DGRT ray-tracing integration
try:
    import sys as _optix_sys
    _3dgrut_root = str(REPOSITORY_ROOT / "3dgrut")
    if _3dgrut_root not in _optix_sys.path:
        _optix_sys.path.insert(0, _3dgrut_root)
    from optix_integration import OptiXRenderer
    OPTIX_INTEGRATION_AVAILABLE = True
except Exception as _optix_import_err:
    OPTIX_INTEGRATION_AVAILABLE = False
    OptiXRenderer = None
else:
    _optix_import_err = None

# Keyword -> material type mapping for auto semantic-to-material assignment
KEYWORD_MATERIAL_MAP = [
    (["mirror", "glass", "window", "glazing", "pane"], "Glass"),
    (["metal", "steel", "iron", "aluminum", "chrome", "copper", "brass", "silver", "gold"], "Metal"),
    (["plastic", "rubber", "resin", "acrylic", "polymer"], "Plastic"),
    (["wood", "fabric", "cloth", "textile", "leather", "carpet", "foam", "paper",
      "concrete", "stone", "brick", "ceramic", "tile", "wall", "floor", "ceiling",
      "matte", "diffuse"], "Matte"),
]


def _align_heatmap_to_image(heat, image_hw):
    """Return a 2D heatmap aligned to an image's ``(height, width)``.

    The contrastive-feature rasterizer can expose its spatial axes as
    ``(width, height)`` while the RGB rasterizer returns ``(height, width)``.
    This is invisible for square viewports but must be corrected before the
    two results are composited.  The interpolation fallback also keeps the
    preview safe if the renderers use different resolutions in the future.
    """
    if heat.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {tuple(heat.shape)}")

    target_hw = tuple(int(value) for value in image_hw)
    if tuple(heat.shape) == target_hw:
        return heat
    if tuple(reversed(heat.shape)) == target_hw:
        return heat.transpose(0, 1)

    return torch.nn.functional.interpolate(
        heat.unsqueeze(0).unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def _reduce_prompt_score_map(scores):
    """Collapse only the prompt axis, preserving both spatial dimensions."""
    if scores.ndim == 2:
        return scores
    if scores.ndim == 3:
        return torch.max(scores, dim=-1).values
    raise ValueError(f"prompt score map must be 2D or 3D, got shape {tuple(scores.shape)}")


# ---------- Config ----------
class RTGSConfig:
    r = 2
    window_width = int(2160 / r)
    window_height = int(1200 / r)
    width = window_width
    height = window_height
    control_width = int(350 * (2 / r))
    control_height = int(700 * (2 / r))
    font_size = min(28, max(16, int(18 * (2 / r))))
    radius = 2
    debug = False
    dt_gamma = 0.2
    sh_degree = 3
    convert_SHs_python = False
    compute_cov3D_python = False
    white_background = False
    FEATURE_DIM = 32
    MODEL_PATH = "./output/figurines"
    FEATURE_GAUSSIAN_ITERATION = 10000
    SCENE_GAUSSIAN_ITERATION = 30000
    RESOLVED_SCENE_PCD = None
    RESOLVED_FEATURE_PCD = None
    RESOLVED_SCALE_GATE = None
    HAS_SEMANTIC_FEATURES = True
    AUTO_FIT_CAMERA = False

    @property
    def SCALE_GATE_PATH(self):
        return self.RESOLVED_SCALE_GATE or os.path.join(
            self.MODEL_PATH, f"point_cloud/iteration_{self.FEATURE_GAUSSIAN_ITERATION}/scale_gate.pt")

    @property
    def FEATURE_PCD_PATH(self):
        return self.RESOLVED_FEATURE_PCD or os.path.join(
            self.MODEL_PATH,
            f"point_cloud/iteration_{self.FEATURE_GAUSSIAN_ITERATION}/contrastive_feature_point_cloud.ply")

    @property
    def SCENE_PCD_PATH(self):
        return self.RESOLVED_SCENE_PCD or os.path.join(
            self.MODEL_PATH, f"point_cloud/iteration_{self.SCENE_GAUSSIAN_ITERATION}/scene_point_cloud.ply")


# ---------- Orbit Camera (same as SAGA) ----------
class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W, self.H = W, H
        self.radius = r
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.rot = R.from_quat([0, 0, 0, 1])
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.right = np.array([1, 0, 0], dtype=np.float32)
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0
        self.rot_mode = 1

    @property
    def pose_movecenter(self):
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        res[:3, 3] -= self.center
        res[:3, 3] = -rot[:3, :3].T @ res[:3, 3]
        return res

    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        res[2, 3] += self.radius
        res[:3, 3] -= self.center
        res[:3, :3] = rot[:3, :3].T
        return res

    def orbit(self, dx, dy):
        if self.rot_mode == 1:
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        else:
            up, side = -self.up, -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius -= 0.1 * delta

    def pan(self, dx, dy, dz=0):
        if self.rot_mode == 1:
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        else:
            self.center += 0.0005 * np.array([-dx, dy, dz])


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-7)
    import cv2
    return cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


# ---------- Main GUI ----------
class RTGSViewerGUI:
    VIEW_RGB = "RGB (Baseline)"
    VIEW_SEGMENTATION = "Segmentation"
    VIEW_MATERIAL = "Material"
    VIEW_RAYTRACING = "Ray-Tracing"

    # Shading params: (ambient, kd, ks, shininess)
    # ambient + kd ≈ 1.0 so the lit side preserves original scene brightness.
    # Materials are distinguished by: specular intensity/sharpness, color modification.
    MATERIAL_SHADING = {
        "Default":  (0.25, 0.75, 0.20,  32.0),  # standard Lambertian + soft gloss
        "Metal":    (0.10, 0.60, 3.00, 128.0),  # dark body, strong sharp specular
        "Glass":    (0.10, 0.70, 4.00, 256.0),  # blue-tinted, very sharp specular
        "Plastic":  (0.20, 0.80, 0.60,  32.0),  # saturated color, medium gloss
        "Matte":    (0.35, 0.65, 0.00,   1.0),  # even diffuse, zero specular
    }
    # Specular highlight color per material
    MATERIAL_SPEC_COLOR = {
        "Default":  [1.0, 1.0, 1.0],
        "Metal":    [1.0, 1.0, 1.0],
        "Glass":    [0.9, 0.95, 1.0],  # slightly blue specular
        "Plastic":  [1.0, 1.0, 1.0],
        "Matte":    [0.0, 0.0, 0.0],
    }
    # Color-space modification applied to the SH base BEFORE shading.
    # None = keep original SH color.
    # "desaturate" = shift toward luminance (metallic grey appearance).
    # [r,g,b] tint = multiply channels (glass blue tint etc.).
    MATERIAL_COLOR_MOD = {
        "Default":  None,
        "Metal":    "desaturate",        # grey-ish, physically metallic
        "Glass":    [0.82, 0.93, 1.05],  # cool blue tint
        "Plastic":  None,                # keep scene color
        "Matte":    None,
    }

    def __init__(self, opt, gs_model: GaussianModel, feat_model: FeatureGaussianModel, scale_gate):
        self.opt = opt
        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.control_width = opt.control_width
        self.control_height = opt.control_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.bg_feature = torch.zeros(opt.FEATURE_DIM, dtype=torch.float32, device="cuda")

        self.engine = {
            "scene": gs_model,
            "feature": feat_model,
            "scale_gate": scale_gate,
        }
        self.render_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.load_model = False

        # View mode
        self.view_mode = self.VIEW_RGB
        # Segmentation state (from SAGA)
        self.clickmode_button = False
        self.clickmode_multi_button = False
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        self.clear_edit = False
        self.roll_back = False
        self.preview = False
        self.segment3d_flag = False
        self.auto_segment_flag = False
        self.sam_driven_flag = False
        self.auto_segment_K = 8
        self.save_flag = False
        self.save_full_mask_flag = False  # NEW: save entire _mask tensor
        self.load_segment_flag = False
        # Optional confidence values affect only Segmentation preview rendering.
        # Integer labels remain authoritative for editing, hiding, and materials.
        self.soft_segment_membership = None
        self.soft_segment_source = None
        self.chosen_feature = None
        self.gates = None
        self.proj_mat = None
        # Material: segment_id -> display name (for panel)
        self.material_labels = {}
        # Assign material: segment_id -> {"type": ..., "name": ...}
        self.material_assignments = {}
        self.MATERIAL_TYPES = ["Default", "Metal", "Glass", "Plastic", "Matte"]
        # RGB colors per material type for segmentation/material view
        self.MATERIAL_TYPE_COLORS = {
            "Default": [0.5, 0.5, 0.5],
            "Metal": [0.78, 0.78, 0.82],
            "Glass": [0.55, 0.82, 1.0],
            "Plastic": [0.95, 0.65, 0.45],
            "Matte": [0.35, 0.35, 0.38],
        }
        self._last_segment_times = -1
        self.segment_colors = None
        self.hidden_segments = set()
        self.confirm_hide_flag = False
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # Dirty flag: only re-render when state/camera changes
        self._dirty = True
        self._last_pose = None
        self._last_view_mode = self.VIEW_RGB
        self._last_rt_submode = "Material"
        # mat_map cache: reuse projected material map when pose/assignments unchanged
        self._mat_map_cache = None          # (H, W) long tensor
        self._mat_map_cache_pose = None     # pose tuple when cache was built
        self._mat_map_dirty = True          # force recompute on next frame

        # Load model
        print("Loading model...")
        self.engine["scene"].load_ply(opt.SCENE_PCD_PATH)
        if getattr(opt, "HAS_SEMANTIC_FEATURES", True):
            self.engine["feature"].load_ply(opt.FEATURE_PCD_PATH)
            self.engine["scale_gate"].load_state_dict(
                torch.load(opt.SCALE_GATE_PATH, map_location="cuda"))
            print("Segmentation features: learned SAGA contrastive features.")
        else:
            self.engine["feature"].load_ply_from_3dgs(opt.SCENE_PCD_PATH)
            proxy = build_proxy_features(self.engine["scene"], opt.FEATURE_DIM)
            self.engine["feature"]._point_features = torch.nn.Parameter(
                proxy.contiguous(), requires_grad=False)
            count = int(self.engine["feature"].get_xyz.shape[0])
            self.engine["feature"].segment_times = 0
            self.engine["feature"]._mask = torch.ones(count, dtype=torch.float32, device="cuda")
            self.engine["feature"].old_mask = []
            with torch.no_grad():
                for parameter in self.engine["scale_gate"].parameters():
                    parameter.zero_()
                linear = next(
                    (module for module in self.engine["scale_gate"].modules()
                     if isinstance(module, torch.nn.Linear)), None)
                if linear is not None and linear.bias is not None:
                    linear.bias.fill_(4.0)
            print("Segmentation features: generated geometry/appearance proxy (plain 3DGS input).")
        self._init_pca()
        if getattr(opt, "AUTO_FIT_CAMERA", False):
            self._fit_camera_to_scene()
        self.load_model = True
        print("Model loaded.")

        # OptiX 3DGRT renderer (initialized after model is loaded into GPU)
        self._optix_renderer = None
        self._backend_error = None
        if OPTIX_INTEGRATION_AVAILABLE:
            try:
                self._optix_renderer = OptiXRenderer(
                    self.engine["scene"],
                    width=self.width,
                    height=self.height,
                    sh_degree=self.opt.sh_degree,
                    bg_color=self.bg_color,
                )
                if self._optix_renderer.available:
                    self._optix_renderer.build_bvh()
                    print("OptiX 3DGRT: ready (true ray tracing active).")
                else:
                    self._backend_error = "3DGRT tracer unavailable"
                    print("OptiX 3DGRT: plugin not compiled — run build_plugin.py. "
                          "Falling back to Blinn-Phong.")
            except Exception as _e:
                self._backend_error = str(_e)
                print(f"OptiX renderer init failed: {_e}")
                self._optix_renderer = None

        dpg.create_context()
        self._register_dpg()

    def _init_pca(self):
        sems = self.engine["feature"].get_point_features.clone().squeeze()
        N, C = sems.shape
        torch.manual_seed(0)
        idx = torch.randint(0, N, [min(200_000, N)])
        sem_chosen = sems[idx]
        sem_chosen = sem_chosen / (torch.norm(sem_chosen, dim=1, keepdim=True) + 1e-6)
        cov = (1 / sem_chosen.shape[0]) * (sem_chosen.T @ sem_chosen).float()
        L, V = torch.linalg.eig(cov)
        L, V = L.real, V.real
        idx_sort = torch.argsort(-L)
        V = V[:, idx_sort]
        self.proj_mat = V[:, :3]

    def _fit_camera_to_scene(self):
        """Robustly frame an imported scene without assuming origin or scale."""
        xyz = self.engine["scene"].get_xyz.detach().float()
        if xyz.numel() == 0:
            return
        center = xyz.median(dim=0).values
        distances = torch.linalg.vector_norm(xyz - center, dim=1)
        extent = torch.quantile(distances, 0.95).clamp_min(1e-3)
        fovy = math.radians(float(self.camera.fovy))
        radius = extent / max(math.sin(fovy * 0.5), 1e-3) * 1.10
        # OrbitCamera stores the pan offset (the negative of the world target).
        self.camera.center = (-center).detach().cpu().numpy().astype(np.float32)
        self.camera.radius = float(radius.item())
        print(f"Camera auto-fit: center={self.camera.center.tolist()}, radius={self.camera.radius:.4f}")

    def _construct_camera(self):
        pose = self.camera.pose_movecenter if self.camera.rot_mode == 1 else self.camera.pose_objcenter
        R_mat = pose[:3, :3]
        t = pose[:3, 3]
        fovy_rad = self.camera.fovy * math.pi / 180.0
        fy = fov2focal(fovy_rad, self.height)
        fovx = focal2fov(fy, self.width)
        cam = Camera(
            colmap_id=0, R=R_mat, T=t, FoVx=fovx, FoVy=fovy_rad,
            image=torch.zeros(3, self.height, self.width),
            gt_alpha_mask=None, image_name=None, uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam

    def _should_update(self):
        """Return True if a re-render is needed (dirty flag check)."""
        # Always update if any action flag is pending
        if any([self.segment3d_flag, self.auto_segment_flag, self.clear_edit,
                self.roll_back, self.save_flag, self.save_full_mask_flag,
                self.load_segment_flag, self.confirm_hide_flag,
                self.sam_driven_flag, self._dirty]):
            return True
        # Camera moved
        pose = self.camera.pose_movecenter
        if self._last_pose is None or not np.allclose(pose, self._last_pose, atol=1e-6):
            self._last_pose = pose.copy()
            return True
        # View mode changed
        if dpg.does_item_exist("_view_mode"):
            view = dpg.get_value("_view_mode")
            if view != self._last_view_mode:
                self._last_view_mode = view
                return True
        # RT submode changed
        if dpg.does_item_exist("_rt_submode"):
            rt_sub = dpg.get_value("_rt_submode")
            if rt_sub != self._last_rt_submode:
                self._last_rt_submode = rt_sub
                return True
        return False

    def _get_hide_mask(self):
        """Returns (N,) bool, True for points that belong to hidden segments."""
        if not self.hidden_segments:
            return None
        scene = self.engine["scene"]
        m = scene._mask
        if m is None or m.numel() == 0:
            return None
        if m.shape[0] != scene.get_xyz.shape[0]:
            import warnings
            warnings.warn(
                f"scene._mask length ({m.shape[0]}) != scene point count ({scene.get_xyz.shape[0]}), "
                "skipping hide_mask. Clear segments and re-segment to fix."
            )
            return None
        hide = torch.zeros(m.shape[0], dtype=torch.bool, device=m.device)
        for sid in self.hidden_segments:
            hide = hide | (m == (sid + 1))
        return hide

    @torch.no_grad()
    def _run_auto_segment(self):
        """Auto segment by clustering on contrastive point features.
        Uses HDBSCAN (density-based, automatic K) when available, with MiniBatchKMeans as fallback."""
        scene = self.engine["scene"]
        feat = self.engine["feature"]
        if scene.get_xyz.shape[0] != feat.get_xyz.shape[0]:
            print("Auto segment: scene and feature point counts differ, skipping.")
            return

        try:
            scene.clear_segment()
            feat.clear_segment()
            self.hidden_segments = set()
        except Exception as e:
            print("Clear segment failed:", e)

        scale = dpg.get_value("_Scale") if dpg.does_item_exist("_Scale") else 0.5
        gates = self.engine["scale_gate"](torch.tensor([scale], device="cuda"))
        feat_pts = feat.get_point_features.squeeze()
        scale_pts = feat_pts * gates.unsqueeze(0)
        scale_pts = torch.nn.functional.normalize(scale_pts, dim=-1, p=2)
        X = scale_pts.cpu().numpy().astype(np.float32)
        if X.shape[0] < 2:
            print("Auto segment: at least two Gaussians are required.")
            return

        use_hdbscan = HDBSCAN_AVAILABLE
        if dpg.does_item_exist("_ClusterAlgo"):
            use_hdbscan = (dpg.get_value("_ClusterAlgo") == "HDBSCAN (auto K)") and HDBSCAN_AVAILABLE

        if use_hdbscan:
            N = X.shape[0]
            sample_size = min(N, max(500, int(0.02 * N)))
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(N, size=sample_size, replace=False)
            X_sample = X[sample_idx]

            min_cluster_size = int(dpg.get_value("_MinClusterSize")) \
                if dpg.does_item_exist("_MinClusterSize") else 50
            min_cluster_size = max(5, min_cluster_size)

            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                core_dist_n_jobs=1,
            )
            sample_labels = clusterer.fit_predict(X_sample)

            # Explicitly exclude noise label -1 (avoids the legacy GUI's off-by-one bug)
            valid_labels = sorted(set(sample_labels.tolist()) - {-1})
            if len(valid_labels) < 2:
                print(f"HDBSCAN found {len(valid_labels)} cluster(s); falling back to MiniBatchKMeans K=8")
                use_hdbscan = False
            else:
                # Full-scene assignment via cosine similarity to cluster centres
                # X is L2-normalised so dot product == cosine similarity
                centres = np.stack([
                    X_sample[sample_labels == lbl].mean(axis=0)
                    for lbl in valid_labels
                ])  # (n_clusters, D)
                sims = X @ centres.T          # (N, n_clusters)
                full_labels = sims.argmax(axis=1)
                labels_tensor = torch.from_numpy(full_labels).cuda()

                for i in range(len(valid_labels)):
                    seg_mask = (labels_tensor == i)
                    if seg_mask.sum() == 0:
                        continue
                    scene.segment(seg_mask)
                    feat.segment(seg_mask)

                self._update_material_labels()
                self._auto_assign_materials()
                print(f"HDBSCAN auto-segment done: {len(valid_labels)} clusters "
                      f"-> {scene.segment_times} segments")

        if not use_hdbscan:
            K = int(dpg.get_value("_AutoSegmentK")) if dpg.does_item_exist("_AutoSegmentK") else 8
            K = max(2, min(50, K, X.shape[0]))
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            labels_t = torch.from_numpy(labels).cuda()
            for i in range(K):
                seg_mask = (labels_t == i)
                if seg_mask.sum() == 0:
                    continue
                scene.segment(seg_mask)
                feat.segment(seg_mask)
            self._update_material_labels()
            self._auto_assign_materials()
            print(f"MiniBatchKMeans auto-segment done: {K} clusters -> {scene.segment_times} segments")

    def _auto_assign_materials(self):
        """Keyword-based automatic semantic-to-material mapping.
        Checks segment display names against KEYWORD_MATERIAL_MAP."""
        changed = False
        for sid, info in self.material_assignments.items():
            name = info.get("name", "").lower()
            if not name:
                continue
            for keywords, mat_type in KEYWORD_MATERIAL_MAP:
                if any(kw in name for kw in keywords):
                    self.material_assignments[sid]["type"] = mat_type
                    changed = True
                    break
        if changed:
            self._mat_map_dirty = True

    def _run_sam_driven_segment(self):
        """SAM-driven segmentation: project 2D SAM masks to 3D via multi-view voting."""
        try:
            from segmentation.sam_driven import (
                run_multiview_selection, run_sam_driven_segment)
            manifest = (dpg.get_value("_multiview_selection_path").strip()
                        if dpg.does_item_exist("_multiview_selection_path") else "")
            if manifest:
                n = run_multiview_selection(
                    manifest, self.opt.MODEL_PATH,
                    self.engine["scene"], self.engine["feature"],
                    diagnostics_path="./segmentation_res/multiview_diagnostics.json")
            else:
                n = run_sam_driven_segment(
                    self.opt.MODEL_PATH,
                    self.engine["scene"],
                    self.engine["feature"],
                    min_votes=2,
                    sample_rate=1.0,
                )
            if n >= 0:
                self.hidden_segments = set()
                self._update_material_labels()
                self._auto_assign_materials()
                self._last_segment_times = -1
        except Exception as e:
            print("SAM-driven segment failed:", e)
            import traceback
            traceback.print_exc()

    def _segment_colors_from_mask(self, mask, material=False):
        """mask: (N,) segment id per point. Returns (N, 3) RGB.
        Uses material type colors when assigned, else random segment colors."""
        if mask is None or mask.numel() == 0:
            return None
        ids = mask.cpu().numpy().astype(int)
        max_id = max(1, ids.max())
        if self.segment_colors is None or self.segment_colors.shape[0] < max_id + 1:
            np.random.seed(42)
            self.segment_colors = np.random.rand(max_id + 1, 3).astype(np.float32)
            # Mask value 1 is the unsegmented/background convention used by
            # GaussianModel. Keep it neutral so selected objects stand out.
            self.segment_colors[0] = [0.2, 0.2, 0.2]
            if self.segment_colors.shape[0] > 1:
                self.segment_colors[1] = [0.12, 0.12, 0.12]
        colors = self.segment_colors.copy()
        if material:
            for sid in self.material_assignments:
                mask_val = sid + 1
                if mask_val < colors.shape[0]:
                    mat_type = self.material_assignments[sid].get("type", "Default")
                    colors[mask_val] = np.array(
                        self.MATERIAL_TYPE_COLORS.get(
                            mat_type, self.MATERIAL_TYPE_COLORS["Default"]),
                        dtype=np.float32
                    )
        return torch.from_numpy(colors[ids]).cuda().float()

    def _update_material_labels(self):
        st = self.engine["scene"].segment_times
        for i in range(1, st + 1):
            if i not in self.material_labels:
                self.material_labels[i] = f"Material_{i}"
            if i not in self.material_assignments:
                self.material_assignments[i] = {"type": "Default", "name": self.material_labels[i]}
                self._mat_map_dirty = True

    def _prune_segment_state(self):
        """Drop UI/material state for segment IDs removed by rollback."""
        last_segment = int(self.engine["scene"].segment_times)
        self.hidden_segments = {sid for sid in self.hidden_segments if sid <= last_segment}
        self.material_labels = {
            sid: value for sid, value in self.material_labels.items() if sid <= last_segment}
        self.material_assignments = {
            sid: value for sid, value in self.material_assignments.items() if sid <= last_segment}
        self._last_segment_times = -1
        self._mat_map_dirty = True

    def _parse_segment_select_value(self, val):
        if val is None or val == "(No segment)":
            return 0
        if isinstance(val, int):
            return val
        try:
            return int(val.replace("Segment ", "").strip())
        except Exception:
            return 0

    def _on_material_segment_select(self, value):
        """When user selects a segment in dropdown, prefill name and type."""
        seg_id = self._parse_segment_select_value(value)
        if seg_id >= 1 and seg_id in self.material_assignments:
            info = self.material_assignments[seg_id]
            if dpg.does_item_exist("_material_type_select"):
                dpg.set_value("_material_type_select", info["type"])
            if dpg.does_item_exist("_material_name_input"):
                dpg.set_value("_material_name_input", info["name"])

    def _apply_material_assignment(self):
        """Apply current UI selection to the chosen segment."""
        val = dpg.get_value("_material_segment_select")
        seg_id = self._parse_segment_select_value(val)
        if seg_id < 1:
            return
        mat_type = dpg.get_value("_material_type_select")
        name = dpg.get_value("_material_name_input")
        if name is None or str(name).strip() == "":
            name = f"Material_{seg_id}"
        self.material_assignments[seg_id] = {"type": mat_type, "name": str(name).strip()}
        self.material_labels[seg_id] = str(name).strip()
        self._last_segment_times = -1
        self._dirty = True
        self._mat_map_dirty = True  # mat_map must be rebuilt with new assignment

    # ---------- Ray-Tracing Rendering ----------

    @torch.no_grad()
    def _compute_depth_normals(self, depth, cam):
        """Compute surface normals from a depth map via finite differences.
        Returns normals tensor of shape (3, H, W) in view space (z pointing toward camera)."""
        if depth.dim() == 3:
            depth = depth.squeeze(0)
        H, W = depth.shape

        dz_dx = torch.zeros_like(depth)
        dz_dy = torch.zeros_like(depth)
        dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) * 0.5
        dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) * 0.5

        # Scale gradients to account for FOV (pixel -> angle)
        fx = 1.0 / math.tan(cam.FoVx * 0.5) * (W * 0.5)
        fy = 1.0 / math.tan(cam.FoVy * 0.5) * (H * 0.5)

        nx = -dz_dx * fx
        ny = -dz_dy * fy
        nz = torch.ones_like(depth)

        norm_len = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-6
        normals = torch.stack([nx / norm_len, ny / norm_len, nz / norm_len], dim=0)  # (3, H, W)
        return normals

    @torch.no_grad()
    def _render_raytracing(self, cam, pipe):
        """Material-aware rendering: depth-based normals + Blinn-Phong per-material shading.

        Sub-modes (selected via _rt_submode combo):
          'Material' - physically shaded render with per-segment material properties
          'Depth'    - false-color depth visualization (Phase 2 validation)
          'Normals'  - surface normals visualized as RGB (Phase 2 validation)
        """
        rt_submode = dpg.get_value("_rt_submode") if dpg.does_item_exist("_rt_submode") else "Material"
        hide_mask = self._get_hide_mask()

        # --- Step 1: OptiX ray trace; SH rasterizer only as fallback ---
        # Avoids a redundant SH render every frame when OptiX is active.
        rgb = None
        depth = None
        normals = None

        if self._optix_renderer is not None and self._optix_renderer.available:
            try:
                optix_out = self._optix_renderer.render(self.camera)
                if optix_out is not None:
                    depth = optix_out["depth"]  # (H, W) hit distance, 0 = background
                    # Convert world-space normals (H, W, 3) -> view-space (3, H, W)
                    # cam.R = R_w2c; row-vector form: n_cam_row = n_world_row @ R_w2c.T
                    R_w2c = torch.tensor(cam.R, device="cuda", dtype=torch.float32)
                    normals_cam = (optix_out["normals"] @ R_w2c.T).permute(2, 0, 1)  # (3, H, W)
                    # Ensure normals point into screen (+Z) to match Blinn-Phong convention
                    flip = (normals_cam[2] < 0).unsqueeze(0).expand(3, -1, -1)
                    normals = torch.where(flip, -normals_cam, normals_cam)
                    rgb = optix_out["rgb"].permute(2, 0, 1)  # (3, H, W)
            except Exception as e:
                self._backend_error = str(e)
                if dpg.does_item_exist("_optix_status"):
                    dpg.set_value("_optix_status", f"Backend: rasterizer fallback ({e})")
                print(f"OptiX render failed: {e}, falling back to rasterizer")

        if rgb is None:
            # SH rasterization fallback (OptiX unavailable or failed)
            scene_out = render(cam, self.engine["scene"], pipe, self.bg_color, filtered_mask=hide_mask)
            rgb = scene_out["render"]

        if depth is None and DEPTH_RENDER_AVAILABLE:
            try:
                depth_out = render_with_depth(cam, self.engine["scene"], pipe, self.bg_color,
                                              filtered_mask=hide_mask)
                depth = depth_out["depth"]
                if depth.dim() == 3:
                    depth = depth.squeeze(0)
                normals = self._compute_depth_normals(depth, cam)
            except Exception as e:
                print(f"Depth render failed: {e}")

        # --- Depth sub-mode ---
        if rt_submode == "Depth":
            if depth is not None:
                valid = depth > 0
                if valid.any():
                    d_min = depth[valid].min()
                    d_max = depth[valid].max()
                else:
                    d_min, d_max = depth.min(), depth.max()
                depth_norm = ((depth - d_min) / (d_max - d_min + 1e-6)).clamp(0, 1)
                # Turbo-like colormap via RGB interpolation: near=warm, far=cool
                r = (1.0 - depth_norm).clamp(0, 1)
                g = (1.0 - (depth_norm - 0.5).abs() * 2).clamp(0, 1)
                b = depth_norm.clamp(0, 1)
                return torch.stack([r, g, b], dim=0)
            else:
                # No depth: show a grey placeholder
                return rgb * 0.5 + 0.25

        # --- Normals sub-mode ---
        if rt_submode == "Normals":
            if normals is not None:
                return ((normals + 1.0) / 2.0).clamp(0, 1)
            else:
                return rgb * 0.5 + 0.25

        # --- Material shading sub-mode ---
        # During camera drag, skip the expensive mat_map render (a full 3DGS rasterization)
        # and return the raw OptiX/SH RGB. Full shading is applied only on still frames.
        if self.moving or self.moving_middle:
            return rgb.clamp(0, 1)

        # Fallback if depth not available: return tinted RGB by material color
        if normals is None:
            seg_mask_tensor = self.engine["scene"]._mask
            seg_colors = self._segment_colors_from_mask(seg_mask_tensor, material=True)
            if seg_colors is not None:
                mat_render = render(cam, self.engine["scene"], pipe, self.bg_color,
                                    override_color=seg_colors, filtered_mask=hide_mask)
                return mat_render["render"]
            return rgb

        # --- Full Blinn-Phong shading ---
        H, W = normals.shape[1], normals.shape[2]
        scene = self.engine["scene"]
        N_pts = scene.get_xyz.shape[0]

        # ── Main light (UI sliders) ──────────────────────────────────────
        light_az = float(dpg.get_value("_LightAz")) if dpg.does_item_exist("_LightAz") else 45.0
        light_el = float(dpg.get_value("_LightEl")) if dpg.does_item_exist("_LightEl") else 60.0
        fill_intensity = float(dpg.get_value("_FillIntensity")) if dpg.does_item_exist("_FillIntensity") else 0.4

        def _make_light(az_deg, el_deg):
            az, el = math.radians(az_deg), math.radians(el_deg)
            return torch.tensor([math.cos(el)*math.sin(az),
                                  -math.sin(el),
                                  math.cos(el)*math.cos(az)],
                                 device="cuda", dtype=torch.float32)

        R_w2c = torch.tensor(cam.R, device="cuda", dtype=torch.float32)  # (3, 3)
        V = torch.tensor([0.0, 0.0, -1.0], device="cuda", dtype=torch.float32)

        def _light_terms(L_world):
            L = R_w2c @ (L_world / (L_world.norm() + 1e-6))
            H = L + V;  H = H / (H.norm() + 1e-6)
            ndotl = (normals[0]*L[0] + normals[1]*L[1] + normals[2]*L[2]).clamp(0, 1)
            ndoth = (normals[0]*H[0] + normals[1]*H[1] + normals[2]*H[2]).clamp(0, 1)
            return ndotl, ndoth

        # Main key light
        NdotL,  NdotH  = _light_terms(_make_light(light_az, light_el))
        # Fill light: opposite azimuth, lower elevation, softer
        NdotL2, NdotH2 = _light_terms(_make_light(light_az + 160.0, 25.0))
        # Rim light: from behind-above camera (camera space: [0, -0.5, -1] normalised)
        # defined directly in camera space — no world transform needed
        L_rim = torch.tensor([0.0, -0.5, -1.0], device="cuda", dtype=torch.float32)
        L_rim = L_rim / L_rim.norm()
        H_rim = (L_rim + V); H_rim = H_rim / H_rim.norm()
        NdotL_rim = (normals[0]*L_rim[0] + normals[1]*L_rim[1] + normals[2]*L_rim[2]).clamp(0, 1)
        NdotH_rim = (normals[0]*H_rim[0] + normals[1]*H_rim[1] + normals[2]*H_rim[2]).clamp(0, 1)

        # Mask for pixels with meaningful normals (valid geometry).
        # Also exclude depth discontinuities: at object edges the central-difference
        # normal computation mixes foreground depth with background (depth=0), producing
        # nearly-horizontal normals that barely pass the nz threshold but yield NdotL≈0,
        # darkening those pixels to ~ambient and causing black spots.
        valid_depth = depth > 0
        depth_diff_x = torch.zeros_like(depth)
        depth_diff_y = torch.zeros_like(depth)
        depth_diff_x[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]).abs()
        depth_diff_y[1:-1, :] = (depth[2:, :] - depth[:-2, :]).abs()
        if valid_depth.any():
            disc_thresh = depth[valid_depth].median().clamp(min=1e-6) * 0.15
        else:
            disc_thresh = depth.new_tensor(1e-3)
        is_discontinuity = (depth_diff_x > disc_thresh) | (depth_diff_y > disc_thresh) | ~valid_depth
        has_normal = (normals[2] > 0.05) & ~is_discontinuity

        result = rgb.clone()

        # mat_map: per-pixel material type (0=Default … 4=Matte).
        # Expensive to compute (full 3DGS rasterization); cache by pose + assignment state.
        oc = self.camera  # OrbitCamera has rot / radius / center
        pose_key = (oc.rot.as_quat().tobytes(), round(float(oc.radius), 5),
                    tuple(round(float(v), 5) for v in oc.center))
        if (self._mat_map_cache is None or self._mat_map_dirty
                or self._mat_map_cache_pose != pose_key):
            mat_idx_gpu = torch.zeros(N_pts, device="cuda", dtype=torch.float32)
            for sid, info in self.material_assignments.items():
                mask_val = sid + 1
                seg_mask_pts = (scene._mask == mask_val)
                midx = float(self.MATERIAL_TYPES.index(info["type"]))
                mat_idx_gpu[seg_mask_pts] = midx
            mat_color_pts = torch.zeros(N_pts, 3, device="cuda")
            mat_color_pts[:, 0] = mat_idx_gpu / 4.0
            mat_img_out = render(cam, scene, pipe, self.bg_color,
                                 override_color=mat_color_pts, filtered_mask=hide_mask)
            self._mat_map_cache = (mat_img_out["render"][0] * 4.0).round().long().clamp(0, 4)
            self._mat_map_cache_pose = pose_key
            self._mat_map_dirty = False
        mat_map = self._mat_map_cache

        # Per-material shading — each material uses view-dependent (NdotV) and/or
        # light-dependent (NdotL/NdotH) terms so differences are always visible:
        #   Default  — standard Blinn-Phong
        #   Metal    — desaturated + matcap (view-dependent brightness) + sharp specular
        #   Glass    — Fresnel rim glow (bright at edges) + sharp specular
        #   Plastic  — SH color + medium gloss
        #   Matte    — SH color + ZERO specular, pure diffuse
        # Broadcast to (1, H, W) for per-pixel ops
        NdotL3     = NdotL.unsqueeze(0)
        NdotH3     = NdotH.unsqueeze(0)
        NdotL2_3   = NdotL2.unsqueeze(0)
        NdotH2_3   = NdotH2.unsqueeze(0)
        NdotLr3    = NdotL_rim.unsqueeze(0)
        NdotHr3    = NdotH_rim.unsqueeze(0)
        # NdotV: view-dependent; view dir = +Z in camera space = normals[2]
        NdotV      = normals[2].clamp(0, 1)
        NdotV3     = NdotV.unsqueeze(0)
        white      = torch.ones(3, 1, 1, device="cuda")
        fi         = fill_intensity   # fill light weight (0-1 from UI)

        for mat_idx, mat_name in enumerate(self.MATERIAL_TYPES):
            active = (mat_map == mat_idx) & has_normal
            if not active.any():
                continue
            active3 = active.unsqueeze(0).expand(3, -1, -1)

            if mat_name == "Default":
                diff = NdotL3 + fi * NdotL2_3
                spec = NdotH3 ** 32 + fi * NdotH2_3 ** 32
                shaded = result * (0.20 + 0.80 * diff) + white * (0.20 * spec)

            elif mat_name == "Metal":
                # Desaturate to grey (metals reflect without hue shift)
                lum = (result[0]*0.299 + result[1]*0.587 + result[2]*0.114).unsqueeze(0)
                base = result * 0.1 + lum.expand(3, -1, -1) * 0.9
                # MatCap: NdotV² → view-dependent bright patch that moves with camera
                matcap = NdotV3 ** 2
                # Specular from all three lights: key + fill + rim
                spec = (4.5 * NdotH3 ** 80
                        + fi * 2.5 * NdotH2_3 ** 80
                        + 0.35 * NdotHr3 ** 60)
                shaded = base * (0.05 + 0.35*NdotL3 + fi*0.20*NdotL2_3 + 0.55*matcap) + white * spec

            elif mat_name == "Glass":
                # Fresnel rim: bright at grazing angles
                fresnel = (1.0 - NdotV3).clamp(0, 1) ** 3
                tint = torch.tensor([0.72, 0.88, 1.0], device="cuda").view(3, 1, 1)
                # Specular from key + rim (rim especially pronounced for glass edges)
                spec = (5.5 * NdotH3 ** 200
                        + fi * 3.0 * NdotH2_3 ** 200
                        + 1.8 * NdotHr3 ** 150)
                shaded = (result * tint * (0.04 + 0.20*NdotL3 + fi*0.10*NdotL2_3)
                          + tint * (1.8 * fresnel)
                          + white * spec)

            elif mat_name == "Plastic":
                diff = NdotL3 + fi * NdotL2_3
                spec = NdotH3 ** 28 + fi * NdotH2_3 ** 28
                shaded = result * (0.15 + 0.85 * diff) + white * (0.60 * spec)

            elif mat_name == "Matte":
                # Zero specular — purely diffuse from key + fill
                diff = NdotL3 + fi * NdotL2_3
                shaded = result * (0.30 + 0.70 * diff)

            else:
                shaded = result

            result = torch.where(active3, shaded, result)

        return result.clamp(0, 1)

    # ---------- Main per-frame data fetch + render ----------

    @torch.no_grad()
    def fetch_data(self, cam):
        pipe = type('Pipe', (), {
            'convert_SHs_python': self.opt.convert_SHs_python,
            'compute_cov3D_python': self.opt.compute_cov3D_python,
            'debug': self.opt.debug,
        })()

        hide_mask = self._get_hide_mask()

        # Feature render (always needed for segmentation interaction)
        feat_out = render_contrastive_feature(cam, self.engine["feature"], pipe, self.bg_feature)
        sems = feat_out["render"].permute(1, 2, 0)  # (H, W, C)
        H, W, C = sems.shape
        sems = sems / (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)

        scale = dpg.get_value("_Scale") if dpg.does_item_exist("_Scale") else 0.5
        self.gates = self.engine["scale_gate"](torch.tensor([scale], device="cuda"))
        scale_feat = sems * self.gates.unsqueeze(0).unsqueeze(0)
        scale_feat = torch.nn.functional.normalize(scale_feat, dim=-1, p=2)

        # ----- State mutations -----
        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            self.prompt_num = 0
            self.hidden_segments = set()
            self.material_labels = {}
            self.material_assignments = {}
            self.soft_segment_membership = None
            self.soft_segment_source = None
            self.segment_colors = None
            self._mat_map_cache = None
            self._mat_map_dirty = True
            try:
                self.engine["scene"].clear_segment()
                self.engine["feature"].clear_segment()
            except Exception as e:
                print("Clear segment failed:", e)
            self._last_segment_times = -1
            self._dirty = True

        if self.roll_back:
            self.new_click_xy = []
            self.roll_back = False
            self.prompt_num = 0
            self.soft_segment_membership = None
            self.soft_segment_source = None
            try:
                self.engine["scene"].roll_back()
                self.engine["feature"].roll_back()
                self._prune_segment_state()
            except Exception as e:
                print("Roll back failed:", e)
            self._dirty = True

        score_map = None
        if len(self.new_click_xy) > 0:
            featmap = scale_feat.reshape(H, W, -1)
            if self.new_click:
                xy = self.new_click_xy
                px = int(np.clip(xy[0], 0, W - 1))
                py = int(np.clip(xy[1], 0, H - 1))
                new_feat = featmap[py, px, :].reshape(-1, 1)
                if self.prompt_num == 0 or not self.clickmode_multi_button:
                    self.chosen_feature = new_feat
                else:
                    self.chosen_feature = torch.cat([self.chosen_feature, new_feat], dim=-1)
                self.prompt_num += 1
                self.new_click = False
                self._dirty = True
            if self.chosen_feature is not None:
                score_map = featmap @ self.chosen_feature
                score_map = (score_map + 1.0) / 2.0
                score_map = _reduce_prompt_score_map(score_map)

        if self.auto_segment_flag:
            self.auto_segment_flag = False
            self.soft_segment_membership = None
            self.soft_segment_source = None
            self._run_auto_segment()
            self._dirty = True

        if self.sam_driven_flag:
            self.sam_driven_flag = False
            self.soft_segment_membership = None
            self.soft_segment_source = None
            self._run_sam_driven_segment()
            self._dirty = True

        if self.segment3d_flag:
            self.segment3d_flag = False
            self.soft_segment_membership = None
            self.soft_segment_source = None
            if len(self.new_click_xy) == 0:
                print("Please right-click on an object to add a prompt, then click Segment 3D")
            else:
                feat_pts = self.engine["feature"].get_point_features.squeeze()
                scale_pts = feat_pts * self.gates.unsqueeze(0)
                scale_pts = torch.nn.functional.normalize(scale_pts, dim=-1, p=2)
                score_pts = scale_pts @ self.chosen_feature
                score_pts = (score_pts + 1.0) / 2.0
                score_thres = dpg.get_value("_ScoreThres") if dpg.does_item_exist("_ScoreThres") else 0.0
                binary_pts = (score_pts > score_thres).sum(1) > 0
                scene = self.engine["scene"]
                feat = self.engine["feature"]
                if scene.get_xyz.shape[0] == feat.get_xyz.shape[0]:
                    segment_mask = binary_pts & (scene._mask == 1)
                else:
                    segment_mask = binary_pts
                self.engine["scene"].segment(segment_mask)
                self.engine["feature"].segment(segment_mask)
                self._update_material_labels()
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

        if self.save_flag:
            self.save_flag = False
            try:
                os.makedirs("./segmentation_res", exist_ok=True)
                name = dpg.get_value("save_name") if dpg.does_item_exist("save_name") else "precomputed_mask"
                save_mask = self.engine["scene"]._mask == self.engine["scene"].segment_times + 1
                torch.save(save_mask.cpu(), f"./segmentation_res/{name}.pt")
                print(f"Saved current segment to ./segmentation_res/{name}.pt")
            except Exception as e:
                print("Save failed (segment first):", e)

        if self.save_full_mask_flag:
            self.save_full_mask_flag = False
            try:
                os.makedirs("./segmentation_res", exist_ok=True)
                name = dpg.get_value("save_name") if dpg.does_item_exist("save_name") else "precomputed_mask"
                full_mask = self.engine["scene"]._mask
                torch.save(full_mask.cpu(), f"./segmentation_res/{name}_full.pt")
                print(f"Saved full mask ({self.engine['scene'].segment_times} segments) "
                      f"to ./segmentation_res/{name}_full.pt")
            except Exception as e:
                print("Save full mask failed:", e)

        if self.load_segment_flag:
            self.load_segment_flag = False
            path = dpg.get_value("load_segment_path") if dpg.does_item_exist("load_segment_path") else ""
            if path and os.path.isfile(path):
                try:
                    scene = self.engine["scene"]
                    feat = self.engine["feature"]
                    use_soft = (dpg.get_value("_load_soft_membership")
                                if dpg.does_item_exist("_load_soft_membership") else False)
                    membership = load_gaussian_membership(
                        path, mode="soft" if use_soft else "binary",
                        expected_count=scene.get_xyz.shape[0])
                    loaded = torch.from_numpy(membership.labels).float()
                    mask_cuda = loaded.cuda()
                    scene._mask = mask_cuda.clone()
                    scene.segment_times = max(0, int(loaded.max().item()) - 1)
                    scene.old_mask = []
                    if feat.get_xyz.shape[0] == scene.get_xyz.shape[0]:
                        feat._mask = mask_cuda.clone()
                        feat.segment_times = scene.segment_times
                        feat.old_mask = []
                    self.soft_segment_membership = (
                        torch.from_numpy(membership.values).to(
                            device=scene.get_xyz.device, dtype=torch.float32)
                        if membership.mode == "soft" else None)
                    self.soft_segment_source = (
                        str(membership.confidence_path)
                        if membership.confidence_path is not None else None)
                    self.hidden_segments = set()
                    self._update_material_labels()
                    self._last_segment_times = -1
                    suffix = (f"; soft preview: {self.soft_segment_source}"
                              if self.soft_segment_source else "")
                    print(f"Loaded segmentation from {path} "
                          f"({scene.segment_times} segments{suffix})")
                except Exception as e:
                    print("Load segmentation failed:", e)
            else:
                print("Load failed: file not found or path empty")
            self._dirty = True

        # ----- Render based on view mode -----
        view = dpg.get_value("_view_mode") if dpg.does_item_exist("_view_mode") else self.VIEW_RGB

        if view == self.VIEW_RGB:
            scene_out = render(cam, self.engine["scene"], pipe, self.bg_color, filtered_mask=hide_mask)
            img = scene_out["render"].permute(1, 2, 0)  # (H, W, 3)

        elif view == self.VIEW_RAYTRACING:
            img = self._render_raytracing(cam, pipe).permute(1, 2, 0)  # (H, W, 3)

        elif view in (self.VIEW_SEGMENTATION, self.VIEW_MATERIAL):
            mask = self.engine["scene"]._mask
            seg_colors = self._segment_colors_from_mask(
                mask, material=view == self.VIEW_MATERIAL)
            if seg_colors is not None:
                seg_out = render(cam, self.engine["scene"], pipe, self.bg_color,
                                 override_color=seg_colors, filtered_mask=hide_mask)
                img = seg_out["render"].permute(1, 2, 0)
                if (view == self.VIEW_SEGMENTATION
                        and self.soft_segment_membership is not None):
                    values = self.soft_segment_membership[:, None].expand(-1, 3)
                    soft_out = render(
                        cam, self.engine["scene"], pipe,
                        torch.zeros_like(self.bg_color), override_color=values,
                        filtered_mask=hide_mask)
                    threshold = (float(dpg.get_value("_soft_membership_threshold"))
                                 if dpg.does_item_exist("_soft_membership_threshold")
                                 else 0.35)
                    coverage = (soft_out["render"][:1] >= threshold).float()
                    rgb_out = render(
                        cam, self.engine["scene"], pipe, self.bg_color,
                        filtered_mask=hide_mask)["render"]
                    composed = rgb_out * (1.0 - coverage) + seg_out["render"] * coverage
                    img = composed.permute(1, 2, 0)
            else:
                scene_out = render(cam, self.engine["scene"], pipe, self.bg_color,
                                   filtered_mask=hide_mask)
                img = scene_out["render"].permute(1, 2, 0)

            # Preview 2D: overlay score_map as a heatmap in Segmentation view
            if view == self.VIEW_SEGMENTATION and self.preview and score_map is not None:
                heat = _align_heatmap_to_image(score_map.clamp(0, 1), img.shape[:2])
                # Red channel boost for heatmap (warm = high score)
                heat_rgb = torch.stack([heat, heat * 0.4, 1.0 - heat], dim=0)  # (3, H, W)
                img_t = img.permute(2, 0, 1)  # (3, H, W)
                alpha = 0.5
                img_t = img_t * (1 - alpha) + heat_rgb * alpha
                img = img_t.permute(1, 2, 0)

        else:
            scene_out = render(cam, self.engine["scene"], pipe, self.bg_color, filtered_mask=hide_mask)
            img = scene_out["render"].permute(1, 2, 0)

        self.render_buffer = img.cpu().numpy().astype(np.float32)
        if dpg.does_item_exist("_texture"):
            dpg.set_value("_texture", self.render_buffer.reshape(-1))

        self._dirty = False

        # Update material list UI only when segment count changed
        self._update_material_labels()
        st = self.engine["scene"].segment_times
        if dpg.does_item_exist("_material_list") and st != self._last_segment_times:
            self._last_segment_times = st
            try:
                dpg.delete_item("_material_list", children_only=True)
            except Exception:
                pass
            if dpg.does_item_exist("_material_segment_select"):
                seg_items = ["(No segment)"] + [f"Segment {i}" for i in range(1, st + 1)]
                dpg.configure_item("_material_segment_select", items=seg_items)
                if st >= 1 and dpg.get_value("_material_segment_select") == "(No segment)":
                    dpg.set_value("_material_segment_select", "Segment 1")
            for sid in sorted(self.material_assignments.keys()):
                info = self.material_assignments[sid]
                dpg.add_text(f"Segment {sid}: {info['type']} — {info['name']}",
                             parent="_material_list")

    def _register_dpg(self):
        self._control_font = None
        for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
            if os.path.isfile(fp):
                try:
                    with dpg.font_registry():
                        self._control_font = dpg.add_font(fp, self.opt.font_size)
                    break
                except Exception:
                    pass

        cw = getattr(self.opt, "control_width", 350)
        ch = getattr(self.opt, "control_height", 700)

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer,
                                format=dpg.mvFormat_Float_rgb, tag="_texture")

        with dpg.window(tag="_main_window", width=self.window_width + cw, height=self.window_height):
            dpg.add_image("_texture", tag="_main_image")
        dpg.set_primary_window("_main_window", True)

        def _screen_to_image_coords(self_ref, screen_x, screen_y):
            if not dpg.does_item_exist("_main_image"):
                return screen_x, screen_y
            try:
                rect_min = dpg.get_item_pos("_main_image")
                rect_size = dpg.get_item_rect_size("_main_image")
                if rect_size[0] <= 0 or rect_size[1] <= 0:
                    return screen_x, screen_y
                local_x = screen_x - rect_min[0]
                local_y = screen_y - rect_min[1]
                if local_x < 0 or local_x >= rect_size[0] or local_y < 0 or local_y >= rect_size[1]:
                    return None, None
                img_x = local_x * (self_ref.width / rect_size[0])
                img_y = local_y * (self_ref.height / rect_size[1])
                return img_x, img_y
            except Exception:
                return screen_x, screen_y

        def click_cb(sender, app_data):
            xy = dpg.get_mouse_pos(local=False)
            if dpg.does_item_exist("pos_item"):
                dpg.set_value("pos_item", f"({xy[0]:.0f}, {xy[1]:.0f})")
            if self.clickmode_button and app_data == 1:
                img_xy = _screen_to_image_coords(self, xy[0], xy[1])
                if img_xy[0] is not None:
                    self.new_click_xy = np.array(img_xy)
                    self.new_click = True

        def toggle_left():
            self.moving = not self.moving
        def toggle_middle():
            self.moving_middle = not self.moving_middle
        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_main_window"):
                dx, dy = self.mouse_pos[0] - pos[0], self.mouse_pos[1] - pos[1]
                if dx != 0 or dy != 0:
                    self.camera.orbit(-dx * 30, dy * 30)
            if self.moving_middle and dpg.is_item_focused("_main_window"):
                dx, dy = self.mouse_pos[0] - pos[0], self.mouse_pos[1] - pos[1]
                if dx != 0 or dy != 0:
                    self.camera.pan(-dx * 20, dy * 20)
            self.mouse_pos = pos
        def wheel_cb(sender, app_data):
            if dpg.is_item_focused("_main_window"):
                self.camera.scale(app_data)

        def on_view_mode_change():
            v = dpg.get_value("_view_mode")
            if v == self.VIEW_RAYTRACING:
                dpg.show_item("_rt_group")
            else:
                dpg.hide_item("_rt_group")
            self._dirty = True

        def on_segment_mode_change(sender, value):
            dpg.hide_item("_manual_segment_group")
            dpg.hide_item("_auto_segment_group")
            dpg.hide_item("_sam_driven_group")
            if value == "Manual":
                dpg.show_item("_manual_segment_group")
            elif value == "Auto":
                dpg.show_item("_auto_segment_group")
            elif value == "SAM-driven":
                dpg.show_item("_sam_driven_group")

        with dpg.window(label="Control", tag="_control_window",
                        width=cw, height=ch, pos=[self.window_width + 10, 0]):

            # --- Status bar ---
            with dpg.group(horizontal=True):
                dpg.add_text("Mouse:")
                dpg.add_text("(0, 0)", tag="pos_item")
            dpg.add_separator()

            # --- View mode ---
            dpg.add_text("View")
            dpg.add_radio_button(
                [self.VIEW_RGB, self.VIEW_SEGMENTATION, self.VIEW_MATERIAL, self.VIEW_RAYTRACING],
                tag="_view_mode",
                default_value=self.VIEW_RGB,
                callback=on_view_mode_change,
            )
            dpg.add_separator()

            # --- Segmentation section ---
            with dpg.collapsing_header(label="Segmentation", default_open=True):
                dpg.add_slider_float(label="Scale", default_value=0.5, min_value=0.0,
                                     max_value=1.0, tag="_Scale")
                dpg.add_slider_float(label="Threshold", default_value=0.0, min_value=0.0,
                                     max_value=1.0, tag="_ScoreThres")
                dpg.add_combo(items=["Manual", "Auto", "SAM-driven"], default_value="Manual",
                              tag="_segment_mode", callback=on_segment_mode_change)

                with dpg.group(tag="_manual_segment_group"):
                    def set_clickmode(s, v): self.clickmode_button = v
                    def set_multiclick(s, v): self.clickmode_multi_button = v
                    def set_preview(s, v):
                        self.preview = v
                        self._dirty = True
                    dpg.add_checkbox(label="Click mode", callback=set_clickmode)
                    dpg.add_checkbox(label="Multi-click", callback=set_multiclick)
                    dpg.add_checkbox(label="Preview heatmap", callback=set_preview)
                    dpg.add_text("Right-click to add prompt", color=[160, 160, 160])
                    dpg.add_button(label="Segment 3D",
                                   callback=lambda: setattr(self, 'segment3d_flag', True))

                with dpg.group(tag="_auto_segment_group", show=False):
                    if HDBSCAN_AVAILABLE:
                        dpg.add_combo(items=["HDBSCAN (auto K)", "KMeans (fixed K)"],
                                      default_value="HDBSCAN (auto K)", tag="_ClusterAlgo")
                        dpg.add_slider_int(label="Min cluster size", default_value=50,
                                           min_value=5, max_value=500, tag="_MinClusterSize")
                    else:
                        dpg.add_combo(items=["KMeans (fixed K)"], default_value="KMeans (fixed K)",
                                      tag="_ClusterAlgo")
                    dpg.add_slider_int(label="K (KMeans)", default_value=8,
                                       min_value=2, max_value=30, tag="_AutoSegmentK")
                    dpg.add_button(label="Auto Segment",
                                   callback=lambda: setattr(self, 'auto_segment_flag', True))

                with dpg.group(tag="_sam_driven_group", show=False):
                    dpg.add_text("Needs sam_masks + mask_scales", color=[160, 160, 160])
                    dpg.add_text("V2.2: cross-view anchor graph", color=[160, 160, 160])
                    dpg.add_input_text(
                        label="Selection JSON", tag="_multiview_selection_path",
                        default_value="", hint="multi-view selection manifest", width=-1)
                    dpg.add_button(label="Run SAM Instance Graph",
                                   callback=lambda: setattr(self, 'sam_driven_flag', True))

                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Roll back",
                                   callback=lambda: setattr(self, 'roll_back', True))
                    dpg.add_button(label="Clear all",
                                   callback=lambda: setattr(self, 'clear_edit', True))

            # --- Material section ---
            with dpg.collapsing_header(label="Material", default_open=False, tag="_material_header"):
                dpg.add_combo(tag="_material_segment_select", items=["(No segment)"],
                              default_value="(No segment)",
                              callback=lambda s, v: self._on_material_segment_select(v))
                dpg.add_combo(tag="_material_type_select", items=self.MATERIAL_TYPES,
                              default_value="Default")
                dpg.add_input_text(tag="_material_name_input", default_value="", hint="Name")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Apply",
                                   callback=lambda: self._apply_material_assignment())
                    dpg.add_button(label="Confirm & hide",
                                   callback=lambda: setattr(self, 'confirm_hide_flag', True))
                dpg.add_child_window(tag="_material_list", height=100)

            # --- Ray-Tracing section (only shown in RT view mode) ---
            with dpg.group(tag="_rt_group", show=False):
                with dpg.collapsing_header(label="Ray-Tracing", default_open=True):
                    # OptiX status indicator
                    _optix_ready = self._optix_renderer is not None and self._optix_renderer.available
                    if _optix_ready:
                        dpg.add_text("Backend: OptiX 3DGRT (true ray tracing)",
                                     color=[100, 220, 100], tag="_optix_status")
                    elif OPTIX_INTEGRATION_AVAILABLE:
                        reason = self._backend_error or "3DGRT tracer unavailable"
                        dpg.add_text(f"Backend: rasterizer fallback ({reason})",
                                     color=[220, 160, 60], tag="_optix_status",
                                     wrap=self.control_width - 20)
                    else:
                        dpg.add_text(f"Backend: rasterizer fallback ({_optix_import_err})",
                                     color=[200, 100, 100], tag="_optix_status",
                                     wrap=self.control_width - 20)

                    rt_items = ["Material", "Depth", "Normals"]
                    if not DEPTH_RENDER_AVAILABLE and not _optix_ready:
                        rt_items = ["Material (no depth/OptiX)"]
                    dpg.add_combo(items=rt_items, default_value=rt_items[0], tag="_rt_submode",
                                  callback=lambda: setattr(self, '_dirty', True))
                    dpg.add_slider_float(label="Light Az", default_value=45.0, min_value=0.0,
                                         max_value=360.0, tag="_LightAz",
                                         callback=lambda: setattr(self, '_dirty', True))
                    dpg.add_slider_float(label="Light El", default_value=60.0, min_value=0.0,
                                         max_value=90.0, tag="_LightEl",
                                         callback=lambda: setattr(self, '_dirty', True))
                    dpg.add_slider_float(label="Fill Light", default_value=0.4, min_value=0.0,
                                         max_value=1.2, tag="_FillIntensity",
                                         callback=lambda: setattr(self, '_dirty', True))

            # --- Save / Load section ---
            with dpg.collapsing_header(label="Save / Load", default_open=False):
                dpg.add_input_text(label="", default_value="precomputed_mask", tag="save_name")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save segment",
                                   callback=lambda: setattr(self, 'save_flag', True))
                    dpg.add_button(label="Save all",
                                   callback=lambda: setattr(self, 'save_full_mask_flag', True))
                dpg.add_separator()
                dpg.add_input_text(label="", default_value="./segmentation_res/sam_driven_mask.pt",
                                   tag="load_segment_path")
                dpg.add_checkbox(
                    label="Confidence-aware preview", default_value=False,
                    tag="_load_soft_membership",
                    callback=lambda: setattr(self, '_dirty', True))
                dpg.add_slider_float(
                    label="Soft pixel threshold", default_value=0.35,
                    min_value=0.05, max_value=0.95, format="%.3f",
                    tag="_soft_membership_threshold",
                    callback=lambda: setattr(self, '_dirty', True))
                dpg.add_button(label="Load segmentation",
                               callback=lambda: setattr(self, 'load_segment_flag', True))

            dpg.add_separator()
            dpg.add_text("Drag:rotate  Mid:pan  Wheel:zoom", color=[130, 130, 130])

        if self._control_font:
            dpg.bind_item_font("_control_window", self._control_font)

        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=wheel_cb)
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda: toggle_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda: toggle_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda: toggle_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda: toggle_middle())
            dpg.add_mouse_move_handler(callback=move_handler)
            dpg.add_mouse_click_handler(callback=click_cb)

        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0,
                                    category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_main_window", theme_no_padding)

        dpg.create_viewport(title="Ray-Tracing 3D Gaussian Viewer",
                            width=self.window_width + cw + 20, height=self.window_height,
                            resizable=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            poll_exports = getattr(self, "_poll_export_events", None)
            if poll_exports is not None:
                poll_exports()
            export_busy = getattr(getattr(self, "_export_manager", None), "running", False)
            if self.load_model and not export_busy and self._should_update():
                cam = self._construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def main():
    parser = ArgumentParser(description="Ray-Tracing 3D Gaussian Viewer (modular)")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="Trained scene directory. If omitted, use the first valid scene in ./output.")
    parser.add_argument("-f", "--feature_iteration", type=int, default=10000)
    parser.add_argument("-s", "--scene_iteration", type=int, default=30000)
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Window scale divisor (2=half resolution, 1=full)")
    args = parser.parse_args()

    opt = RTGSConfig()
    opt.r = args.scale
    opt.window_width = int(2160 / opt.r)
    opt.window_height = int(1200 / opt.r)
    opt.width = opt.window_width
    opt.height = opt.window_height
    opt.control_width = int(350 * (2 / opt.r))
    opt.control_height = int(700 * (2 / opt.r))
    opt.font_size = min(28, max(16, int(18 * (2 / opt.r))))
    model_path = args.model_path
    if model_path is None:
        output_root = str(REPOSITORY_ROOT / "output")
        candidates = []
        if os.path.isdir(output_root):
            for entry in sorted(os.listdir(output_root)):
                candidate = os.path.join(output_root, entry)
                scene_ply = os.path.join(
                    candidate, "point_cloud", f"iteration_{args.scene_iteration}",
                    "scene_point_cloud.ply",
                )
                feature_ply = os.path.join(
                    candidate, "point_cloud", f"iteration_{args.feature_iteration}",
                    "contrastive_feature_point_cloud.ply",
                )
                scale_gate = os.path.join(
                    candidate, "point_cloud", f"iteration_{args.feature_iteration}",
                    "scale_gate.pt",
                )
                if all(os.path.isfile(p) for p in (scene_ply, feature_ply, scale_gate)):
                    candidates.append(candidate)
        if not candidates:
            parser.error("no valid trained scene found under ./output; pass -m MODEL_PATH")
        model_path = candidates[0]
        print(f"No model path supplied; using {model_path}")

    opt.MODEL_PATH = model_path
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration

    # Validate paths before loading
    for name, path in [
        ("Scene PLY", opt.SCENE_PCD_PATH),
        ("Feature PLY", opt.FEATURE_PCD_PATH),
        ("Scale gate", opt.SCALE_GATE_PATH),
    ]:
        if not os.path.isfile(path):
            print(f"Error: {name} not found: {path}")
            print("Please check -m, -f, -s arguments and ensure the model is trained.")
            return

    gs_model = GaussianModel(opt.sh_degree)
    feat_model = FeatureGaussianModel(opt.FEATURE_DIM)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, opt.FEATURE_DIM, bias=True),
        torch.nn.Sigmoid(),
    ).cuda()

    gui = RTGSViewerGUI(opt, gs_model, feat_model, scale_gate)
    gui.render()


if __name__ == "__main__":
    main()
