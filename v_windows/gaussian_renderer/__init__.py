#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, override_mask = None, filtered_mask = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    if filtered_mask is not None:
        if filtered_mask.shape[0] != opacity.shape[0]:
            raise IndexError(
                f"filtered_mask length ({filtered_mask.shape[0]}) must match scene opacity length ({opacity.shape[0]}). "
                "hide_mask should be built from the same point cloud (scene) that is being rendered."
            )
        new_opacity = opacity.detach().clone()
        new_opacity[filtered_mask, :] = 0
        opacity = new_opacity

    # The SAGA rasterizer used by this project extends the original 3DGS API
    # with one scalar mask per Gaussian and returns a rendered mask as well.
    # Base scene training does not optimize masks, so the model's all-ones
    # mask preserves the original RGB rendering behavior.
    mask = pc.get_mask if override_mask is None else override_mask
    if mask.numel() != means3D.shape[0]:
        raise RuntimeError(
            f"Gaussian mask/point count mismatch: mask={mask.numel()}, points={means3D.shape[0]}. "
            "Densification must update both tensors together."
        )

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_mask, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        mask = mask,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}



def render_mask(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, precomputed_mask = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # start_time  = time.time()
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # print("Render time checker: raster_settings", time.time() - start_time)
    # start_time  = time.time()


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    mask = pc.get_mask if precomputed_mask is None else precomputed_mask
    if mask.numel() != means3D.shape[0]:
        raise RuntimeError(
            f"Gaussian mask/point count mismatch: mask={mask.numel()}, points={means3D.shape[0]}."
        )
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # print("Render time checker: prepare vars", time.time() - start_time)


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_mask, radii = rasterizer.forward_mask(
        means3D = means3D,
        means2D = means2D,
        opacities = opacity,
        mask = mask,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # print("Render time checker: main render", time.time() - start_time)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

try:
    from diff_gaussian_rasterization_depth import (
        GaussianRasterizationSettings as GaussianRasterizationSettingsDepth,
        GaussianRasterizer as GaussianRasterizerDepth,
    )
    DEPTH_RASTERIZER_AVAILABLE = True
except (ImportError, OSError):
    # The project-specific depth rasterizer is not published with this checkout.
    # Keep RGB/segmentation usable on Windows; the GUI will use OptiX when present.
    DEPTH_RASTERIZER_AVAILABLE = False

def _render_with_depth_impl(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, override_mask = None, filtered_mask = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # start_time  = time.time()
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsDepth(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # print("Render time checker: raster_settings", time.time() - start_time)
    # start_time  = time.time()


    rasterizer = GaussianRasterizerDepth(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    if filtered_mask is not None:
        new_opacity = opacity.detach().clone()
        new_opacity[filtered_mask, :] = -1.
        opacity = new_opacity

    mask = pc.get_mask if override_mask is None else override_mask

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # print("Render time checker: prepare vars", time.time() - start_time)


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_mask, rendered_depth, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        mask = mask,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # print("Render time checker: main render", time.time() - start_time)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_mask,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def _render_with_depth_fallback(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, override_mask = None, filtered_mask = None):
    """Render SAGA-compatible accumulated camera-space depth without its optional depth extension.

    SAGA's depth rasterizer uses the Gaussian centre's view-space z value as
    its feature and composites it with the same alpha/transmittance equation
    as RGB.  Feeding that value through the regular rasterizer therefore
    produces the same (non-normalized) depth convention.
    """
    del bg_color, override_color

    xyz = pc.get_xyz
    xyz_homogeneous = torch.cat(
        (xyz, torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)),
        dim=1,
    )
    camera_depth = (xyz_homogeneous @ viewpoint_camera.world_view_transform)[:, 2:3]
    depth_as_color = camera_depth.expand(-1, 3).contiguous()
    black = torch.zeros(3, dtype=xyz.dtype, device=xyz.device)

    result = render(
        viewpoint_camera,
        pc,
        pipe,
        black,
        scaling_modifier=scaling_modifier,
        override_color=depth_as_color,
        override_mask=override_mask,
        filtered_mask=filtered_mask,
    )
    result["depth"] = result["render"][0:1]
    return result


if DEPTH_RASTERIZER_AVAILABLE:
    render_with_depth = _render_with_depth_impl
else:
    render_with_depth = _render_with_depth_fallback

from diff_gaussian_rasterization_contrastive_f import GaussianRasterizationSettings as GaussianRasterizationSettingsContrastiveF
from diff_gaussian_rasterization_contrastive_f import GaussianRasterizer as GaussianRasterizerContrastiveF
from scene.gaussian_model_ff import FeatureGaussianModel

def render_contrastive_feature(viewpoint_camera, pc : FeatureGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, norm_point_features = False, smooth_type = None, smooth_weights = None, smooth_K = 16, filtered_mask = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsContrastiveF(
        image_height=int(viewpoint_camera.feature_height),
        image_width=int(viewpoint_camera.feature_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizerContrastiveF(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    if filtered_mask is not None:
        new_opacity = opacity.detach().clone()
        new_opacity[filtered_mask, :] = 0
        opacity = new_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if smooth_type is None:
        colors_precomp = pc.get_point_features
    elif smooth_type == 'multi_res':
        colors_precomp = pc.get_multi_resolution_smoothed_point_features(smooth_weights = smooth_weights)
    elif smooth_type == 'traditional':
        colors_precomp = pc.get_smoothed_point_features(K = smooth_K, dropout=0.5)

    if norm_point_features:
        colors_precomp = colors_precomp / (colors_precomp.norm(dim=1, keepdim=True) + 1e-9)
    # colors_precomp = torch.nn.functional.normalize(colors_precomp, dim=1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
