"""
GaussianAdapter
===============
Wraps the SAGA GaussianModel (and optionally a hidden-segment mask) so that it
satisfies the interface expected by the 3DGRT Tracer:

    tracer.build_acc(gaussians)
    tracer.render(gaussians, gpu_batch)

3DGRT expects the object to expose:
    .positions           Tensor (N, 3)
    .rotation            raw (pre-activation) quaternion Tensor (N, 4)
    .scale               raw (pre-activation) scale Tensor  (N, 3)
    .density             raw (pre-activation) opacity Tensor (N, 1)
    .rotation_activation callable  quat -> normalized quat
    .scale_activation    callable  log-scale -> scale
    .density_activation  callable  logit -> sigmoid opacity
    .get_rotation()      -> activated rotation  (N, 4)
    .get_scale()         -> activated scale     (N, 3)
    .get_density()       -> activated density   (N, 1)
    .get_features()      -> SH coefficients     (N, F)
    .n_active_features   int  (SH degree^2 used by renderer)
    .background()        composites background color into the ray output

SAGA GaussianModel stores:
    ._xyz                positions
    ._rotation           raw quaternion
    ._scaling            raw log-scale
    ._opacity            raw logit-opacity
    .get_features        property -> (N, F) SH coefficients (DC + rest)
    .get_xyz             property -> positions
    .get_rotation        property -> normalized quaternion
    .get_scaling         property -> exp(log-scale)
    .get_opacity         property -> sigmoid(logit-opacity)
"""

import torch
import torch.nn.functional as F
import math


class GaussianAdapter:
    """Adapts a SAGA GaussianModel to the 3DGRT Tracer interface.

    Args:
        gs_model: A SAGA ``GaussianModel`` instance (scene PLY loaded).
        mask: Optional boolean Tensor of shape (N,). When provided, only the
              Gaussians where mask==True are exposed to the tracer (useful for
              hiding segmented objects or focusing ray tracing on a subset).
        sh_degree: Active SH degree (0-3).  Defaults to the model's sh_degree.
        bg_color: Background RGB Tensor (3,). Defaults to black.
    """

    def __init__(self, gs_model, mask=None, sh_degree: int | None = None, bg_color=None):
        self._model = gs_model
        self._mask = mask  # (N,) bool or None
        self._sh_degree = sh_degree if sh_degree is not None else getattr(gs_model, "active_sh_degree", 3)
        self._bg_color = bg_color if bg_color is not None else torch.zeros(3, device="cuda")

        # 3DGRT activation callables (match SAGA conventions)
        self.rotation_activation = F.normalize
        self.scale_activation = torch.exp
        self.density_activation = torch.sigmoid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._mask is None:
            return tensor
        return tensor[self._mask]

    # ------------------------------------------------------------------
    # Raw (pre-activation) fields — required by build_acc()
    # ------------------------------------------------------------------

    @property
    def positions(self) -> torch.Tensor:
        return self._apply_mask(self._model._xyz)

    @property
    def rotation(self) -> torch.Tensor:
        return self._apply_mask(self._model._rotation)

    @property
    def scale(self) -> torch.Tensor:
        return self._apply_mask(self._model._scaling)

    @property
    def density(self) -> torch.Tensor:
        return self._apply_mask(self._model._opacity)

    # ------------------------------------------------------------------
    # Activated fields — used inside Tracer._Autograd.apply()
    # ------------------------------------------------------------------

    def get_rotation(self) -> torch.Tensor:
        """Normalized quaternion (N, 4)."""
        return self.rotation_activation(self.rotation, dim=1)

    def get_scale(self) -> torch.Tensor:
        """Positive scale (N, 3)."""
        return self.scale_activation(self.scale)

    def get_density(self) -> torch.Tensor:
        """Opacity in [0, 1] (N, 1)."""
        return self.density_activation(self.density)

    def get_features(self) -> torch.Tensor:
        """SH coefficients (N, n_active_features).

        SAGA stores features as (N, 3, (sh_degree+1)^2) or (N, F).
        We flatten and trim to the number of coefficients the active SH
        degree requires.
        """
        feats = self._apply_mask(self._model.get_features)  # SAGA property
        # feats shape: (N, 3*(max_sh+1)^2) — treat as flat SH block
        # Trim to active degree
        n_active = self.n_active_features
        if feats.shape[1] > n_active:
            feats = feats[:, :n_active]
        return feats.contiguous()

    @property
    def n_active_features(self) -> int:
        """Number of SH coefficients used for rendering."""
        return (self._sh_degree + 1) ** 2 * 3  # 3 channels × n_coeffs

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    # ------------------------------------------------------------------
    # Background compositing — required by Tracer.render()
    # ------------------------------------------------------------------

    def background(
        self,
        T_to_world: torch.Tensor,
        rays_dir: torch.Tensor,
        pred_rgb: torch.Tensor,
        pred_opacity: torch.Tensor,
        train: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple constant-color background compositing.

        Args:
            T_to_world: (1, 4, 4) camera-to-world transform (unused here).
            rays_dir:   (H, W, 3) ray directions (unused here).
            pred_rgb:   (H, W, 3) accumulated radiance.
            pred_opacity: (H, W, 1) accumulated opacity.
            train:      ignored.

        Returns:
            composited_rgb (H, W, 3), pred_opacity (H, W, 1)
        """
        bg = self._bg_color.to(pred_rgb.device)
        # Alpha-composite: out = alpha * pred + (1 - alpha) * bg
        composited = pred_rgb + (1.0 - pred_opacity) * bg.view(1, 1, 3)
        return composited, pred_opacity

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def update_mask(self, mask):
        """Replace the Gaussian subset mask (None = use all Gaussians)."""
        self._mask = mask

    def mask_from_segment_ids(self, segment_mask_tensor: torch.Tensor, segment_ids: list[int]) -> torch.Tensor:
        """Build a boolean mask selecting Gaussians belonging to given segment IDs.

        Args:
            segment_mask_tensor: (N,) int tensor from SAGA segmentation.
            segment_ids: list of segment label values to include.

        Returns:
            Boolean Tensor (N,).
        """
        mask = torch.zeros(segment_mask_tensor.shape[0], dtype=torch.bool, device=segment_mask_tensor.device)
        for sid in segment_ids:
            mask |= (segment_mask_tensor == sid)
        return mask
