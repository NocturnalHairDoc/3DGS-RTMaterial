"""
OptiX 3DGRT Integration for SegAnyGAussians
============================================

Bridges the SAGA GaussianModel with the NVIDIA 3DGRT OptiX ray tracer.

Modules:
    gaussian_adapter   - Wraps SAGA GaussianModel for the 3DGRT Tracer API
    ray_generator      - Generates camera rays from OrbitCamera / Camera objects
    optix_renderer     - Builds BVH and dispatches OptiX ray tracing
    build_plugin       - Compiles the 3DGRT OptiX C++/CUDA plugin

Quick start::

    from optix_integration import OptiXRenderer, build_3dgrt_plugin
    build_3dgrt_plugin()                        # compile once
    renderer = OptiXRenderer(gs_model)
    renderer.build_bvh()
    result = renderer.render(camera, material_assignments)
"""

from .optix_renderer import OptiXRenderer
from .gaussian_adapter import GaussianAdapter
from .ray_generator import RayGenerator
from .build_plugin import build_3dgrt_plugin

__all__ = [
    "OptiXRenderer",
    "GaussianAdapter",
    "RayGenerator",
    "build_3dgrt_plugin",
]
