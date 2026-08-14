"""PBR-lite material storage and G-buffer shading for 3D Gaussian scenes.

The module keeps inverse rendering out of scope: albedo, roughness, metallic and
opacity are explicit user-editable parameters.  It consumes 3DGRT normals/depth
and optionally blends traced reflection/refraction rays in the style of the
3DGRUT playground hybrid renderer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


_PI = float(np.pi)


def _normalise(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values, dim=-1, eps=1e-8)


def linear_to_srgb(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(0.0)
    return torch.where(values <= 0.0031308, 12.92 * values,
                       1.055 * values.pow(1.0 / 2.4) - 0.055)


def srgb_to_linear(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(0.0, 1.0)
    return torch.where(values <= 0.04045, values / 12.92,
                       ((values + 0.055) / 1.055).pow(2.4))


def aces_tonemap(values: torch.Tensor) -> torch.Tensor:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ((values * (a * values + b)) / (values * (c * values + d) + e)).clamp(0, 1)


@dataclass(frozen=True)
class PBRMaterial:
    albedo: tuple[float, float, float] = (0.8, 0.8, 0.8)
    roughness: float = 0.5
    metallic: float = 0.0
    opacity: float = 1.0
    ior: float = 1.5

    def validated(self) -> "PBRMaterial":
        return PBRMaterial(
            tuple(float(np.clip(value, 0.0, 1.0)) for value in self.albedo),
            float(np.clip(self.roughness, 0.04, 1.0)),
            float(np.clip(self.metallic, 0.0, 1.0)),
            float(np.clip(self.opacity, 0.0, 1.0)),
            float(np.clip(self.ior, 1.0, 2.5)),
        )


class PBRParameterStore:
    """Dense per-Gaussian PBR fields with segment-level editing helpers."""

    def __init__(self, albedo: torch.Tensor):
        if albedo.ndim != 2 or albedo.shape[1] != 3:
            raise ValueError("albedo must have shape (N, 3)")
        self.albedo = albedo.detach().float().clamp(0, 1).clone()
        count, device = len(albedo), albedo.device
        self.roughness = torch.full((count, 1), 0.5, device=device)
        self.metallic = torch.zeros((count, 1), device=device)
        self.opacity = torch.ones((count, 1), device=device)
        self.ior = torch.full((count, 1), 1.5, device=device)
        self.segment_materials: dict[int, PBRMaterial] = {}

    @classmethod
    def from_gaussian_model(cls, model) -> "PBRParameterStore":
        # 3DGS DC is stored as (N, 1, 3); SH evaluation adds 0.5.
        dc = model._features_dc.detach()
        if dc.ndim == 3:
            dc = dc[:, 0, :]
        return cls((0.5 + 0.28209479177387814 * dc).clamp(0, 1))

    def apply_to_gaussians(self, gaussian_ids, material: PBRMaterial) -> None:
        material = material.validated()
        ids = torch.as_tensor(gaussian_ids, device=self.albedo.device)
        if ids.dtype == torch.bool:
            if ids.numel() != len(self.albedo):
                raise ValueError("boolean Gaussian mask has the wrong size")
        else:
            ids = ids.long()
        self.albedo[ids] = torch.tensor(material.albedo, device=self.albedo.device)
        self.roughness[ids] = material.roughness
        self.metallic[ids] = material.metallic
        self.opacity[ids] = material.opacity
        self.ior[ids] = material.ior

    def apply_to_segment(self, segment_id: int, scene_mask: torch.Tensor,
                         material: PBRMaterial) -> int:
        # UI segment 1 corresponds to Gaussian mask value 2.
        selected = scene_mask.long() == int(segment_id) + 1
        count = int(selected.sum().item())
        if count:
            self.apply_to_gaussians(selected, material)
            self.segment_materials[int(segment_id)] = material.validated()
        return count

    def packed_properties(self) -> torch.Tensor:
        return torch.cat((self.roughness, self.metallic, self.opacity), dim=1)

    def metadata(self) -> dict:
        return {str(key): asdict(value) for key, value in self.segment_materials.items()}

    def restore_metadata(self, metadata: dict, scene_mask: torch.Tensor) -> None:
        for key, value in metadata.items():
            self.apply_to_segment(int(key), scene_mask, PBRMaterial(**value))


class HDREnvironment:
    """Latitude-longitude floating-point environment with procedural fallback."""

    def __init__(self, pixels: torch.Tensor, exposure: float = 0.0):
        if pixels.ndim != 3 or pixels.shape[-1] < 3:
            raise ValueError("environment must have shape (H, W, >=3)")
        self.pixels = pixels[..., :3].float().clamp_min(0.0)
        self.exposure = float(exposure)

    @classmethod
    def procedural(cls, height=128, width=256, device="cpu") -> "HDREnvironment":
        """Create a deterministic HDR sky/ground map with a high-energy sun."""
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=device),
            torch.linspace(0, 1, width, device=device), indexing="ij",
        )
        horizon = torch.exp(-((yy - 0.52) / 0.22).square())
        sky = torch.stack((0.12 + 0.28 * horizon, 0.20 + 0.38 * horizon,
                           0.38 + 0.52 * horizon), dim=-1)
        ground = torch.stack((0.10 + 0.12 * horizon, 0.08 + 0.10 * horizon,
                              0.06 + 0.08 * horizon), dim=-1)
        pixels = torch.where((yy < 0.55)[..., None], sky, ground)
        sun = torch.exp(-(((xx - 0.72) / 0.025).square() + ((yy - 0.23) / 0.035).square()))
        pixels = pixels + sun[..., None] * torch.tensor([18.0, 14.0, 8.0], device=device)
        return cls(pixels)

    @classmethod
    def load(cls, path, device="cpu", exposure=0.0) -> "HDREnvironment":
        import imageio.v3 as iio
        source = Path(path).expanduser()
        pixels = np.asarray(iio.imread(source), dtype=np.float32)
        if pixels.max(initial=0) > 255:
            pixels /= 65535.0
        elif pixels.max(initial=0) > 2.0 and source.suffix.lower() not in {".hdr", ".exr"}:
            pixels /= 255.0
        return cls(torch.from_numpy(pixels).to(device), exposure=exposure)

    @property
    def device(self):
        return self.pixels.device

    def to(self, device) -> "HDREnvironment":
        self.pixels = self.pixels.to(device)
        return self

    def sample(self, directions: torch.Tensor, roughness=None) -> torch.Tensor:
        directions = _normalise(directions)
        x, y, z = directions.unbind(-1)
        u = torch.atan2(x, z) / (2 * torch.pi) + 0.5
        v = torch.acos(y.clamp(-1, 1)) / torch.pi
        grid = torch.stack((u * 2 - 1, v * 2 - 1), dim=-1)
        original_shape = grid.shape[:-1]
        sampled = F.grid_sample(
            self.pixels.permute(2, 0, 1).unsqueeze(0),
            grid.reshape(1, 1, -1, 2), mode="bilinear", padding_mode="border",
            align_corners=True,
        ).reshape(3, -1).T.reshape(*original_shape, 3)
        sampled = sampled * (2.0 ** self.exposure)
        if roughness is not None:
            mean = self.pixels.mean((0, 1)) * (2.0 ** self.exposure)
            blur = torch.as_tensor(roughness, device=sampled.device).clamp(0, 1)
            while blur.ndim < sampled.ndim:
                blur = blur.unsqueeze(-1)
            sampled = sampled * (1 - blur.square()) + mean * blur.square()
        return sampled


def reconstruct_world_positions(origins: torch.Tensor, directions: torch.Tensor,
                                depth: torch.Tensor) -> torch.Tensor:
    return origins + _normalise(directions) * depth.unsqueeze(-1)


def stabilize_gbuffer_normals(normals: torch.Tensor, positions: torch.Tensor,
                              depth: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
    """Fuse 3DGRT normals with depth tangents away from discontinuities.

    Gaussian covariance normals can vary sharply over a single splat.  The
    depth-derived component suppresses those ellipse contours while the traced
    normal remains authoritative at silhouettes and invalid depth neighbours.
    """
    traced = _normalise(torch.nan_to_num(normals))
    traced = torch.where(((traced * view_dirs).sum(-1, keepdim=True) < 0), -traced, traced)
    dx, dy = torch.zeros_like(positions), torch.zeros_like(positions)
    dx[:, 1:-1] = positions[:, 2:] - positions[:, :-2]
    dy[1:-1, :] = positions[2:, :] - positions[:-2, :]
    derived = _normalise(torch.cross(dx, dy, dim=-1))
    derived = torch.where(((derived * view_dirs).sum(-1, keepdim=True) < 0), -derived, derived)
    depth_dx, depth_dy = torch.zeros_like(depth), torch.zeros_like(depth)
    depth_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]).abs()
    depth_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]).abs()
    relative = torch.maximum(depth_dx, depth_dy) / depth.abs().clamp_min(1e-4)
    valid = ((depth > 0) & (relative < 0.08)
             & torch.isfinite(derived).all(-1) & (derived.norm(dim=-1) > 0.5))
    fused = _normalise(0.30 * traced + 0.70 * derived)
    return torch.where(valid.unsqueeze(-1), fused, traced)


def reflection_directions(incident: torch.Tensor, normals: torch.Tensor,
                          roughness: torch.Tensor | None = None) -> torch.Tensor:
    incident, normals = _normalise(incident), _normalise(normals)
    reflected = incident - 2.0 * (incident * normals).sum(-1, keepdim=True) * normals
    if roughness is not None:
        amount = torch.as_tensor(roughness, device=incident.device).clamp(0, 1).square() * 0.18
        reflected = reflected * (1 - amount) + normals * amount
    return _normalise(reflected)


def refraction_directions(incident: torch.Tensor, normals: torch.Tensor,
                          ior: torch.Tensor) -> torch.Tensor:
    """Snell refraction with total-internal-reflection fallback."""
    incident, normals = _normalise(incident), _normalise(normals)
    cosi = (incident * normals).sum(-1, keepdim=True).clamp(-1, 1)
    entering = cosi < 0
    oriented_n = torch.where(entering, normals, -normals)
    eta = torch.where(entering, 1.0 / ior.clamp_min(1.0), ior).clamp(0.4, 2.5)
    cosi = cosi.abs()
    k = 1.0 - eta.square() * (1.0 - cosi.square())
    refracted = eta * incident + (eta * cosi - torch.sqrt(k.clamp_min(0))) * oriented_n
    reflected = reflection_directions(incident, oriented_n)
    return _normalise(torch.where(k >= 0, refracted, reflected))


def cook_torrance_ggx(albedo, roughness, metallic, normals, view_dirs,
                      light_dirs, radiance, visibility=1.0):
    """Cook–Torrance BRDF with GGX NDF, Smith masking and Schlick Fresnel."""
    n, v, l = _normalise(normals), _normalise(view_dirs), _normalise(light_dirs)
    h = _normalise(v + l)
    ndotl = (n * l).sum(-1, keepdim=True).clamp(0, 1)
    ndotv = (n * v).sum(-1, keepdim=True).clamp(1e-4, 1)
    ndoth = (n * h).sum(-1, keepdim=True).clamp(0, 1)
    vdoth = (v * h).sum(-1, keepdim=True).clamp(0, 1)
    alpha = roughness.clamp(0.04, 1).square()
    alpha2 = alpha.square()
    denom = (ndoth.square() * (alpha2 - 1) + 1).square()
    distribution = alpha2 / (_PI * denom + 1e-7)
    k = (roughness + 1).square() / 8.0
    g_v = ndotv / (ndotv * (1 - k) + k + 1e-7)
    g_l = ndotl / (ndotl * (1 - k) + k + 1e-7)
    geometry = g_v * g_l
    f0 = 0.04 * (1 - metallic) + albedo * metallic
    fresnel = f0 + (1 - f0) * (1 - vdoth).pow(5)
    specular = distribution * geometry * fresnel / (4 * ndotv * ndotl + 1e-6)
    diffuse = (1 - fresnel) * (1 - metallic) * albedo / _PI
    return (diffuse + specular) * radiance * ndotl * visibility


class PBRLiteCompositor:
    def __init__(self, environment: HDREnvironment | None = None, exposure=0.0):
        self.environment = environment or HDREnvironment.procedural()
        self.exposure = float(exposure)

    def shade(self, albedo, roughness, metallic, material_opacity, normals, depth,
              view_dirs, light_dir, light_radiance, shadow_visibility=None,
              primary_rgb=None, reflected_linear=None, refracted_linear=None,
              gbuffer_opacity=None):
        """Shade linear PBR maps and blend optional traced secondary rays."""
        device = albedo.device
        self.environment.to(device)
        n, v = _normalise(normals), _normalise(view_dirs)
        l = _normalise(torch.as_tensor(light_dir, device=device, dtype=albedo.dtype))
        l = l.view(*([1] * (albedo.ndim - 1)), 3).expand_as(albedo)
        radiance = torch.as_tensor(light_radiance, device=device, dtype=albedo.dtype)
        radiance = radiance.view(*([1] * (albedo.ndim - 1)), 3)
        visibility = 1.0 if shadow_visibility is None else shadow_visibility
        direct = cook_torrance_ggx(albedo, roughness, metallic, n, v, l, radiance, visibility)

        reflected_dir = reflection_directions(-v, n, roughness)
        diffuse_env = self.environment.sample(n, torch.ones_like(roughness) * 0.85)
        spec_env = self.environment.sample(reflected_dir, roughness)
        ndotv = (n * v).sum(-1, keepdim=True).clamp(0, 1)
        f0 = 0.04 * (1 - metallic) + albedo * metallic
        fresnel = f0 + (1 - f0) * (1 - ndotv).pow(5)
        ibl = diffuse_env * albedo * (1 - metallic) * (1 - fresnel) + spec_env * fresnel
        surface = direct + ibl

        if reflected_linear is not None:
            surface = surface + reflected_linear * fresnel * (0.15 + 0.55 * metallic)
        opacity = material_opacity.clamp(0, 1)
        if refracted_linear is not None:
            transmission = (1 - opacity) * (1 - metallic) * (1 - fresnel)
            surface = surface * opacity + refracted_linear * transmission
        elif primary_rgb is not None:
            surface = surface * opacity + srgb_to_linear(primary_rgb) * (1 - opacity)

        hit = (depth > 0).unsqueeze(-1)
        coverage = torch.ones_like(material_opacity)
        if gbuffer_opacity is not None:
            coverage = gbuffer_opacity.clamp(0, 1)
            hit = hit & (coverage > 0.03)
        if primary_rgb is not None:
            background = srgb_to_linear(primary_rgb)
            surface = surface * coverage + background * (1 - coverage)
            surface = torch.where(hit, surface, background)
        surface = aces_tonemap(surface * (2.0 ** self.exposure))
        return linear_to_srgb(surface).clamp(0, 1)
