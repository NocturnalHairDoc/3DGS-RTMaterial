"""
OptiXRenderer
=============
High-level renderer that ties together GaussianAdapter, RayGenerator, and the
3DGRT Tracer.  Designed as a drop-in replacement for the Blinn-Phong
approximation currently used in ``rt_gs_gui.py``.

Usage::

    from optix_integration import OptiXRenderer, build_3dgrt_plugin

    # 1. Build the OptiX plugin once (requires OptiX SDK + CUDA)
    build_3dgrt_plugin()

    # 2. Create renderer
    renderer = OptiXRenderer(gs_model, width=1080, height=600)

    # 3. Build BVH (call after loading / changing the Gaussian model)
    renderer.build_bvh()

    # 4. Render a frame
    result = renderer.render(orbit_cam)
    # result["rgb"]     (H, W, 3)  float32 — composited colour
    # result["normals"] (H, W, 3)  float32 — world-space surface normals
    # result["depth"]   (H, W)     float32 — hit distance (or 0 = miss)
    # result["opacity"] (H, W, 1)  float32 — accumulated opacity

The renderer falls back gracefully to None if the 3DGRT plugin is not yet
compiled, allowing ``rt_gs_gui.py`` to detect availability and fall back to
the existing Blinn-Phong path.
"""

import logging
import sys
import os
import torch

from .gaussian_adapter import GaussianAdapter
from .ray_generator import RayGenerator

logger = logging.getLogger(__name__)


def sanitize_normals(normals: torch.Tensor) -> torch.Tensor:
    """Replace invalid normals and normalize every non-degenerate vector."""
    finite = torch.isfinite(normals).all(dim=-1, keepdim=True)
    clean = torch.where(finite, normals, torch.zeros_like(normals))
    length = torch.linalg.vector_norm(clean, dim=-1, keepdim=True)
    return torch.where(
        length > 1e-8,
        clean / length.clamp_min(1e-8),
        torch.zeros_like(clean),
    )

# Minimal 3DGRT render configuration.  Mirrors the defaults in the 3dgrut repo.
_DEFAULT_RENDER_CONF = {
    "pipeline_type": "reference",
    "backward_pipeline_type": "referenceBwd",
    "primitive_type": "gaussian",        # 'gaussian' | 'trisurfel'
    "particle_kernel_degree": 2,
    "particle_kernel_min_response": 1e-3,
    "particle_kernel_density_clamping": False,
    "particle_radiance_sph_degree": 3,
    "enable_normals": True,
    "enable_hitcounts": False,
    "min_transmittance": 1e-3,
    "max_consecutive_bvh_update": 16,
    "enable_kernel_timings": False,
    "particle_kernel_min_alpha": 1e-4,
    "particle_kernel_max_alpha": 1.0,
    "particle_feature_half": False,
    "feature_output_half": False,
}


class _RenderConf:
    """Simple namespace-like config consumed by setup_3dgrt and Tracer."""
    def __init__(self, d: dict):
        self.render = type("RenderConf", (), d)()
        progressive = type("ProgressiveTrainingConf", (), {"max_n_features": 3})()
        activation = type("ActivationConf", (), {"type": "none", "num_frequencies": 1})()
        nht = type("NHTConf", (), {
            "dim": 48, "activation": activation, "interpolation_type": "none"
        })()
        self.model = type("ModelConf", (), {
            "feature_type": "sh",
            "progressive_training": progressive,
            "nht_features": nht,
        })()


class OptiXRenderer:
    """Ray-traces a SAGA GaussianModel using the 3DGRT OptiX backend.

    Args:
        gs_model:  A loaded SAGA ``GaussianModel``.
        width:     Render width in pixels.
        height:    Render height in pixels.
        sh_degree: Active SH degree (0–3).
        bg_color:  Background colour Tensor (3,) on CUDA.
        render_conf: Dict overriding ``_DEFAULT_RENDER_CONF`` entries.
    """

    def __init__(
        self,
        gs_model,
        width: int = 1080,
        height: int = 600,
        sh_degree: int = 3,
        bg_color: torch.Tensor | None = None,
        render_conf: dict | None = None,
    ):
        self._gs_model = gs_model
        self.width = width
        self.height = height

        if bg_color is None:
            bg_color = torch.zeros(3, device="cuda")

        # Adapter wraps SAGA model for 3DGRT
        self.adapter = GaussianAdapter(gs_model, sh_degree=sh_degree, bg_color=bg_color)
        # Ray generator builds pixel rays from camera params
        self.ray_gen = RayGenerator(width, height, device="cuda")

        # Build render config
        conf_dict = dict(_DEFAULT_RENDER_CONF)
        if render_conf:
            conf_dict.update(render_conf)
        self._conf = _RenderConf(conf_dict)

        # 3DGRT Tracer (None until plugin is compiled)
        self._tracer = None
        self._bvh_built = False
        self._active_mask_key = None

        self._try_load_tracer()

    # ------------------------------------------------------------------
    # Plugin loading
    # ------------------------------------------------------------------

    def _try_load_tracer(self):
        """Attempt to import and instantiate the 3DGRT Tracer."""
        try:
            # Insert the 3dgrut directory so its modules are importable
            _3dgrut_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "3dgrut")
            )
            if _3dgrut_root not in sys.path:
                sys.path.insert(0, _3dgrut_root)

            from threedgrt_tracer.tracer import Tracer
            self._tracer = Tracer(self._conf)
            logger.info("3DGRT OptiX Tracer loaded successfully.")
        except ImportError as e:
            logger.warning(
                f"3DGRT plugin not available ({e}). "
                "Run optix_integration.build_3dgrt_plugin() to compile it."
            )
            self._tracer = None
        except Exception as e:
            logger.warning(f"3DGRT Tracer init failed: {e}")
            self._tracer = None

    @property
    def available(self) -> bool:
        """True if the OptiX backend is ready to render."""
        return self._tracer is not None

    # ------------------------------------------------------------------
    # BVH
    # ------------------------------------------------------------------

    def build_bvh(self, rebuild: bool = True):
        """Build (or update) the OptiX BVH acceleration structure.

        Must be called once after loading the Gaussian model, and again any
        time the Gaussians change significantly (e.g. after densification).

        Args:
            rebuild: Force a full BVH rebuild (vs. incremental update).
        """
        if not self.available:
            logger.warning("build_bvh() skipped — 3DGRT plugin not loaded.")
            return
        with torch.no_grad():
            self._tracer.build_acc(self.adapter, rebuild=rebuild)
        self._bvh_built = True
        logger.info(f"BVH built for {self.adapter.num_gaussians:,} Gaussians.")

    @staticmethod
    def _mask_key(mask: torch.Tensor | None):
        if mask is None:
            return None
        return (mask.data_ptr(), mask.numel(), getattr(mask, "_version", 0))

    def update_bvh(self):
        """Incremental BVH update (cheaper than full rebuild)."""
        self.build_bvh(rebuild=False)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, camera, segment_mask=None) -> dict | None:
        """Render one frame using OptiX ray tracing.

        Args:
            camera: Either an ``OrbitCamera`` (from rt_gs_gui.py) or a SAGA
                    ``Camera`` object.
            segment_mask: Optional (N,) int tensor. When provided, only
                          Gaussians with mask > 0 are ray-traced (for
                          per-segment material rendering).

        Returns:
            dict with keys:
                "rgb"     (H, W, 3) composited colour, float32
                "normals" (H, W, 3) world-space surface normals, float32
                "depth"   (H, W)    ray hit distance (0 = background), float32
                "opacity" (H, W, 1) accumulated opacity, float32
            or ``None`` if the plugin is unavailable or BVH not built.
        """
        if not self.available:
            return None
        # The OptiX acceleration structure contains only the active Gaussian
        # subset. Rebuild when visibility changes, then reuse it on every frame.
        bool_mask = segment_mask.bool().contiguous() if segment_mask is not None else None
        desired_key = self._mask_key(bool_mask)
        if not self._bvh_built or desired_key != self._active_mask_key:
            self.adapter.update_mask(bool_mask)
            self.build_bvh(rebuild=True)
            self._active_mask_key = desired_key

        # Build ray batch
        batch = self._ray_batch(camera)
        return self._trace_batch(batch)

    def _ray_batch(self, camera, region=None):
        if hasattr(camera, "pose_movecenter"):
            return self.ray_gen.from_orbit_camera(camera, region=region)
        return self.ray_gen.from_saga_camera(camera, region=region)

    def _trace_batch(self, batch):

        with torch.no_grad():
            out = self._tracer.render(self.adapter, batch, train=False)

        H, W = self.height, self.width
        # Current 3DGRT names the ray colour tensor ``pred_features``; older
        # integration revisions used ``pred_rgb``. Support both explicitly.
        rgb_tensor = out.get("pred_features", out.get("pred_rgb"))
        if rgb_tensor is None:
            raise RuntimeError(f"3DGRT output has no RGB/features tensor: {sorted(out)}")
        rgb = rgb_tensor.squeeze(0)        # (H, W, 3)
        opacity = out["pred_opacity"].squeeze(0)  # (H, W, 1)
        depth = out["pred_dist"].squeeze(0).squeeze(-1)   # (H, W)
        normals = out["pred_normals"].squeeze(0)  # (H, W, 3)
        # 3DGRT can emit NaNs for rays whose normal estimate is degenerate.
        # Never let those values enter visualisation or BRDF calculations.
        normals = sanitize_normals(normals)

        rgb = rgb + (1.0 - opacity) * self.adapter._bg_color.view(1, 1, 3)
        return {
            "rgb": torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0),
            "normals": normals,
            "depth": torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0),
            "opacity": torch.nan_to_num(opacity, nan=0.0).clamp(0.0, 1.0),
        }

    def render_tiled(self, camera, tile_size=1024, segment_mask=None, progress=None):
        """Render bounded ray batches and stitch exact full-frame coordinates on CPU."""
        if tile_size < 16:
            raise ValueError("tile_size must be at least 16")
        bool_mask = segment_mask.bool().contiguous() if segment_mask is not None else None
        desired_key = self._mask_key(bool_mask)
        if not self._bvh_built or desired_key != self._active_mask_key:
            self.adapter.update_mask(bool_mask)
            self.build_bvh(rebuild=True)
            self._active_mask_key = desired_key
        result = {
            "rgb": torch.empty((self.height, self.width, 3), dtype=torch.float32),
            "normals": torch.empty((self.height, self.width, 3), dtype=torch.float32),
            "depth": torch.empty((self.height, self.width), dtype=torch.float32),
            "opacity": torch.empty((self.height, self.width, 1), dtype=torch.float32),
        }
        regions = [(x, y, min(tile_size, self.width-x), min(tile_size, self.height-y))
                   for y in range(0, self.height, tile_size)
                   for x in range(0, self.width, tile_size)]
        for index, (x, y, w, h) in enumerate(regions, 1):
            tile = self._trace_batch(self._ray_batch(camera, (x, y, w, h)))
            for key, value in tile.items():
                result[key][y:y+h, x:x+w].copy_(value.detach().cpu())
            if progress is not None:
                progress(index, len(regions))
        return result

    # ------------------------------------------------------------------
    # Convenience: render to numpy (H, W, 3) uint8 for DearPyGUI
    # ------------------------------------------------------------------

    def render_to_numpy(self, camera, segment_mask=None):
        """Render and return a uint8 numpy array (H, W, 3) for display."""
        import numpy as np
        result = self.render(camera, segment_mask=segment_mask)
        if result is None:
            return None
        rgb_np = result["rgb"].cpu().numpy()  # (H, W, 3) float
        return (rgb_np * 255).clip(0, 255).astype(np.uint8)
