"""
MaterialSHEditor
================
Edits Gaussian SH coefficients in-place to simulate material appearance,
then restores the original parameters after rendering.

Why this works better than post-process Blinn-Phong:
- Changes live inside the 3DGS representation, so alpha-compositing, depth-sorting,
  and view-dependent SH evaluation all happen correctly for free.
- Matte vs Metal is immediately obvious: Matte zeroes all higher-order SH
  (same colour from every angle), Metal keeps/amplifies them (shimmers as camera moves).
- Glass reduces opacity → you see objects behind it → real transparency.
- No light direction needed; effects are purely view-dependent (matcap-style).

Material recipes
----------------
Metal   : desaturate DC (achromatic grey base) + keep higher-order SH
           → grey + view-dependent shimmer that moves with camera
Glass   : blue-tint DC + reduce opacity to ~30 %
           + keep higher-order SH for glassy view-dependent glints
Plastic : slight saturation boost + reduce higher-order SH to 15 %
           → vivid flat colour with almost no view-dependence
Matte   : zero all higher-order SH
           → purely diffuse, identical from every angle, "dead flat"
Default : no modification
"""

import torch

# Zeroth-order SH coefficient (same constant used in 3DGS codebase)
_SH_C0 = 0.28209479177387814


def _dc_to_rgb(dc: torch.Tensor) -> torch.Tensor:
    """Convert _features_dc[:, 0, :] (N, 3) → linear RGB (N, 3)."""
    return (0.5 + _SH_C0 * dc).clamp(0.0, 1.0)


def _rgb_to_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Convert target linear RGB (N, 3) → _features_dc[:, 0, :] values."""
    return (rgb - 0.5) / _SH_C0


class MaterialSHEditor:
    """
    Apply / restore material edits on a SAGA GaussianModel.

    Usage::

        editor = MaterialSHEditor(scene)

        # every frame in Material view mode:
        editor.apply_all(material_assignments, scene._mask)
        img = render(cam, scene, pipe, bg)["render"]
        editor.restore()
    """

    def __init__(self, gs_model):
        self._model = gs_model
        # Snapshot originals once at construction time
        self._orig_dc   = gs_model._features_dc.data.clone()
        self._orig_rest = gs_model._features_rest.data.clone()
        self._orig_opa  = gs_model._opacity.data.clone()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def restore(self):
        """Restore all Gaussian parameters to their original values."""
        self._model._features_dc.data.copy_(self._orig_dc)
        self._model._features_rest.data.copy_(self._orig_rest)
        self._model._opacity.data.copy_(self._orig_opa)

    def apply_all(self, material_assignments: dict, scene_mask: torch.Tensor):
        """
        Edit Gaussians according to material_assignments.

        Each entry in material_assignments can carry either:
          {"type": "Metal"}                       — discrete preset
          {"type": "Custom", "params": {...}}      — continuous params from material_clip
            params keys: specular_gain, saturation, opacity_scale, tint, strength

        Parameters
        ----------
        material_assignments : {seg_id: {"type": str, ...}}
        scene_mask           : (N,) int tensor, value = seg_id+1 for each Gaussian
        """
        self.restore()   # start from a clean slate each call

        if scene_mask is None:
            return

        for seg_id, info in material_assignments.items():
            mat_type = info.get("type", "Default")
            if mat_type == "Default":
                continue
            mask = (scene_mask == (seg_id + 1))
            if not mask.any():
                continue
            # Custom continuous params (from material_clip text/CLIP pipeline)
            if mat_type == "Custom" and "params" in info:
                self.apply_params(mask, info["params"])
            else:
                self._apply(mask, mat_type)

    def apply_params(self, mask: torch.Tensor, params: dict):
        """
        Apply material edits using continuous parameters from material_clip.

        params keys (all optional, defaults to neutral):
          specular_gain  float  — scale factor for higher-order SH (0=matte, >1=metallic)
          saturation     float  — colour saturation multiplier (1=unchanged, >1=vivid)
          opacity_scale  float  — opacity multiplier (1=opaque, 0.28=glass-like)
          tint           list[3]— RGB multiplicative tint applied to DC
          strength       float  — blend weight between original and edited (0–1)
        """
        strength      = float(params.get("strength", 1.0))
        specular_gain = float(params.get("specular_gain", 1.0))
        saturation    = float(params.get("saturation", 1.0))
        opacity_scale = float(params.get("opacity_scale", 1.0))
        tint          = params.get("tint", [1.0, 1.0, 1.0])

        dev = self._model._features_dc.data.device

        # ── DC: tint + saturation ─────────────────────────────────────────
        dc       = self._model._features_dc.data[mask]   # (N, 1, 3)
        base_rgb = _dc_to_rgb(dc[:, 0, :])               # (N, 3)

        tint_t   = torch.tensor(tint, device=dev, dtype=torch.float32)
        tinted   = (base_rgb * tint_t).clamp(0, 1)

        mean     = tinted.mean(dim=1, keepdim=True)
        saturated = (mean + (tinted - mean) * saturation).clamp(0, 1)

        edited_rgb = base_rgb * (1 - strength) + saturated * strength
        self._model._features_dc.data[mask, 0, :] = _rgb_to_dc(edited_rgb.clamp(0, 1))

        # ── Higher-order SH: specular_gain ───────────────────────────────
        orig_rest = self._orig_rest[mask]
        new_rest  = orig_rest * (1 - strength + strength * specular_gain)
        self._model._features_rest.data[mask] = new_rest

        # ── Opacity ───────────────────────────────────────────────────────
        if abs(opacity_scale - 1.0) > 1e-4:
            orig_p = torch.sigmoid(self._orig_opa[mask])
            new_p  = (orig_p * (1 - strength + strength * opacity_scale)).clamp(1e-6, 1 - 1e-6)
            self._model._opacity.data[mask] = torch.log(new_p / (1.0 - new_p))

    # ------------------------------------------------------------------
    # Per-material recipes
    # ------------------------------------------------------------------

    def _apply(self, mask: torch.Tensor, material_type: str):
        dc       = self._model._features_dc.data[mask]       # (N, 1, 3)
        base_rgb = _dc_to_rgb(dc[:, 0, :])                   # (N, 3)

        if material_type == "Metal":
            self._apply_metal(mask, base_rgb)
        elif material_type == "Glass":
            self._apply_glass(mask, base_rgb)
        elif material_type == "Plastic":
            self._apply_plastic(mask, base_rgb)
        elif material_type == "Matte":
            self._apply_matte(mask)

    def _apply_metal(self, mask, base_rgb):
        """Achromatic grey base + amplified view-dependent SH → metallic shimmer."""
        lum = (base_rgb[:, 0] * 0.299
               + base_rgb[:, 1] * 0.587
               + base_rgb[:, 2] * 0.114).unsqueeze(1)          # (N, 1)
        # Desaturate: 5 % original colour + 95 % luminance
        metal_rgb = (base_rgb * 0.05 + lum * 0.95).clamp(0, 1)
        self._model._features_dc.data[mask, 0, :] = _rgb_to_dc(metal_rgb)
        # Amplify higher-order SH → stronger view-dependent shimmer
        self._model._features_rest.data[mask] = self._orig_rest[mask] * 1.5

    def _apply_glass(self, mask, base_rgb):
        """Transparency (opacity ~28 %) + very subtle cool tint + kept view-dependent SH.
        The primary effect is see-through; colour should stay close to original."""
        # Extremely subtle cool tint — barely perceptible shift toward blue/grey
        tint = torch.tensor([0.94, 0.97, 1.02], device=base_rgb.device)
        glass_rgb = (base_rgb * tint).clamp(0, 1)
        self._model._features_dc.data[mask, 0, :] = _rgb_to_dc(glass_rgb)

        # Reduce opacity to ~28 % of original → transparent
        orig_p = torch.sigmoid(self._orig_opa[mask])           # (N, 1)
        new_p  = (orig_p * 0.28).clamp(1e-6, 1.0 - 1e-6)
        self._model._opacity.data[mask] = torch.log(new_p / (1.0 - new_p))

        # Keep higher-order SH (glassy specular-like glints)
        # (already restored to orig_rest by restore(); no action needed)

    def _apply_plastic(self, mask, base_rgb):
        """Saturated colour + almost no higher-order SH → vivid flat surface."""
        # Boost saturation: push away from grey mean
        mean = base_rgb.mean(dim=1, keepdim=True)
        plastic_rgb = (mean + (base_rgb - mean) * 1.5).clamp(0, 1)
        self._model._features_dc.data[mask, 0, :] = _rgb_to_dc(plastic_rgb)
        # Kill 85 % of higher-order SH → nearly view-independent (plastic look)
        self._model._features_rest.data[mask] = self._orig_rest[mask] * 0.15

    def _apply_matte(self, mask):
        """Zero all higher-order SH → purely diffuse, same from every angle."""
        # DC (base colour) is intentionally kept unchanged
        self._model._features_rest.data[mask] = 0.0
