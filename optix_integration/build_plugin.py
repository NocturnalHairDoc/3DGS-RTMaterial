"""
build_plugin.py
===============
One-call helper to compile the 3DGRT OptiX C++/CUDA plugin from source.

Prerequisites (must be met before calling build_3dgrt_plugin()):
  - CUDA 11.8+ toolkit (nvcc on PATH, or CUDA_HOME set)
  - NVIDIA OptiX SDK 8.x — set OPTIX_HOME env variable to the SDK root
  - slang compiler (slangtorch) — install with: pip install slangtorch
  - gcc <= 11  (Ubuntu 24.04: sudo apt install gcc-11 g++-11)
  - Python packages: torch, threedgrut (from 3dgrut/setup.py)

Usage::

    python -c "from optix_integration.build_plugin import build_3dgrt_plugin; build_3dgrt_plugin()"

Or interactively in the viewer::

    from optix_integration import build_3dgrt_plugin
    build_3dgrt_plugin(verbose=True)
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Absolute path to the bundled 3dgrut submodule
_3DGRUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "3dgrut"))

# Default render config used during compilation (controls CUDA #defines)
_DEFAULT_BUILD_CONF = {
    "pipeline_type": "reference",
    "backward_pipeline_type": "referenceBwd",
    "primitive_type": "gaussian",
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


class _Conf:
    def __init__(self, d):
        self.render = type("R", (), d)()
        progressive = type("P", (), {"max_n_features": 3})()
        activation = type("A", (), {"type": "none", "num_frequencies": 1})()
        nht = type("N", (), {"dim": 48, "activation": activation, "interpolation_type": "none"})()
        self.model = type("M", (), {"feature_type": "sh", "progressive_training": progressive,
                                      "nht_features": nht})()


def build_3dgrt_plugin(conf_overrides: dict | None = None, verbose: bool = True) -> bool:
    """Compile the 3DGRT OptiX plugin.

    Compilation happens via JIT (torch.utils.cpp_extension) and the compiled
    shared library is cached in PyTorch's extension cache directory so it is
    only rebuilt when the source changes.

    Args:
        conf_overrides: Optional dict to override ``_DEFAULT_BUILD_CONF``
                        entries (e.g. change SH degree or primitive type).
        verbose:        Print build log to stdout.

    Returns:
        True on success, False on failure.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Ensure 3dgrut is importable
    if _3DGRUT_ROOT not in sys.path:
        sys.path.insert(0, _3DGRUT_ROOT)

    # Check prerequisites
    issues = _check_prerequisites(verbose)
    if issues:
        logger.error("Build prerequisites not met:\n" + "\n".join(f"  - {i}" for i in issues))
        return False

    conf_dict = dict(_DEFAULT_BUILD_CONF)
    if conf_overrides:
        conf_dict.update(conf_overrides)
    conf = _Conf(conf_dict)

    try:
        from threedgrt_tracer.setup_3dgrt import setup_3dgrt
        logger.info("Compiling 3DGRT OptiX plugin (this may take a few minutes)...")
        plugin = setup_3dgrt(conf)
        logger.info(f"Plugin compiled successfully: {plugin}")
        return True
    except Exception as e:
        logger.error(f"Plugin compilation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def _check_prerequisites(verbose: bool) -> list[str]:
    """Return a list of unmet prerequisite descriptions (empty = all good)."""
    issues = []

    # 1. CUDA
    import torch
    if not torch.cuda.is_available():
        issues.append("CUDA not available (torch.cuda.is_available() == False)")
    cuda_home = (
        os.environ.get("CUDA_HOME")
        or os.environ.get("CUDA_PATH")
        or _find_cuda_home()
    )
    if cuda_home is None:
        issues.append("CUDA toolkit not found — set CUDA_HOME environment variable")
    elif verbose:
        logger.info(f"CUDA home: {cuda_home}")

    # 2. OptiX SDK
    optix_home = os.environ.get("OPTIX_HOME") or os.environ.get("OPTIX_PATH")
    optix_include = None
    if optix_home:
        optix_include = os.path.join(optix_home, "include", "optix.h")
        if not os.path.isfile(optix_include):
            issues.append(
                f"OptiX SDK header not found at {optix_include}. "
                "Download from https://developer.nvidia.com/designworks/optix/downloads/legacy"
            )
        elif verbose:
            logger.info(f"OptiX SDK: {optix_home}")
    else:
        # Try the bundled stub in the 3dgrut submodule
        bundled = os.path.join(
            _3DGRUT_ROOT, "threedgrt_tracer", "dependencies", "optix-dev", "include", "optix.h"
        )
        if os.path.isfile(bundled):
            if verbose:
                logger.info(f"Using bundled OptiX stubs: {bundled}")
        else:
            issues.append(
                "OptiX SDK not found. Set OPTIX_HOME to the OptiX SDK root, "
                "or ensure the bundled stubs exist in "
                "3dgrut/threedgrt_tracer/dependencies/optix-dev/include/"
            )

    # 3. slangtorch
    try:
        import slangtorch  # noqa: F401
        if verbose:
            logger.info("slangtorch found.")
    except ImportError:
        issues.append("slangtorch not installed — run: pip install slangtorch")

    # 4. threedgrut package
    try:
        sys.path.insert(0, _3DGRUT_ROOT)
        import threedgrut  # noqa: F401
        if verbose:
            logger.info("threedgrut package importable.")
    except ImportError:
        issues.append(
            "threedgrut package not installed — run: "
            "cd 3dgrut && pip install -e . --no-build-isolation"
        )

    return issues


def _find_cuda_home() -> str | None:
    """Try to locate the CUDA home directory."""
    import shutil
    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc is at <CUDA_HOME>/bin/nvcc
        return os.path.dirname(os.path.dirname(os.path.abspath(nvcc)))
    for candidate in ["/usr/local/cuda", "/usr/cuda"]:
        if os.path.isdir(candidate):
            return candidate
    return None


def print_setup_instructions():
    """Print step-by-step setup instructions for the OptiX plugin."""
    print("""
=======================================================================
  3DGRT OptiX Plugin — Setup Instructions
=======================================================================

Step 1: Install CUDA 11.8+ toolkit
    sudo apt install nvidia-cuda-toolkit
    # or download from https://developer.nvidia.com/cuda-downloads

Step 2: Download NVIDIA OptiX SDK 8.x
    https://developer.nvidia.com/designworks/optix/downloads/legacy
    export OPTIX_HOME=/path/to/OptiX-SDK-8.x.x

Step 3: Install slangtorch (Slang shader compiler)
    pip install slangtorch

Step 4: Install the threedgrut Python package
    cd 3dgrut
    pip install -e . --no-build-isolation

Step 5: (Ubuntu 24.04 only) Install gcc-11
    sudo apt install gcc-11 g++-11
    export CC=gcc-11 CXX=g++-11

Step 6: Compile the plugin
    python -c "from optix_integration import build_3dgrt_plugin; build_3dgrt_plugin()"

=======================================================================
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the 3DGRT OptiX plugin")
    parser.add_argument("--check", action="store_true", help="Only check prerequisites")
    args = parser.parse_args()

    if args.check:
        issues = _check_prerequisites(verbose=True)
        if issues:
            print("\nUnmet prerequisites:")
            for i in issues:
                print(f"  ✗ {i}")
        else:
            print("\nAll prerequisites met.  Run without --check to compile.")
    else:
        success = build_3dgrt_plugin(verbose=True)
        sys.exit(0 if success else 1)
