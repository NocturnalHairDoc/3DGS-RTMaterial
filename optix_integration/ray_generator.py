"""
RayGenerator
============
Converts camera parameters (from OrbitCamera / SAGA Camera objects) into the
ray tensors required by the 3DGRT Tracer:

    gpu_batch.rays_ori  (H, W, 3)  — ray origin in world space
    gpu_batch.rays_dir  (H, W, 3)  — unit ray direction in world space
    gpu_batch.T_to_world (1, 4, 4) — camera-to-world transform

The 3DGRT Tracer shoots one ray per pixel.  For a pinhole camera the origin
is the camera centre (constant across the image) and the direction is computed
from the pixel grid via the intrinsics.
"""

import math
import numpy as np
import torch


class Batch:
    """Minimal stand-in for threedgrut.datasets.protocols.Batch."""

    def __init__(self, rays_ori: torch.Tensor, rays_dir: torch.Tensor, T_to_world: torch.Tensor):
        self.rays_ori = rays_ori          # (H, W, 3)
        self.rays_dir = rays_dir          # (H, W, 3)
        self.T_to_world = T_to_world      # (1, 4, 4)


class RayGenerator:
    """Generate per-pixel rays from camera intrinsics and pose.

    Usage::

        gen = RayGenerator(width=1080, height=600, device="cuda")
        batch = gen.from_orbit_camera(orbit_cam)
        # or
        batch = gen.from_saga_camera(saga_cam)
    """

    def __init__(self, width: int, height: int, device: str = "cuda"):
        self.width = width
        self.height = height
        self.device = device

        # Pre-build pixel grid (stays constant across frames)
        # Pixel centres at (i + 0.5, j + 0.5)
        j_coords = torch.arange(height, dtype=torch.float32, device=device)  # rows
        i_coords = torch.arange(width, dtype=torch.float32, device=device)   # cols
        grid_j, grid_i = torch.meshgrid(j_coords, i_coords, indexing="ij")   # (H, W)
        self._grid_i = grid_i  # column index
        self._grid_j = grid_j  # row index

    # ------------------------------------------------------------------
    # Public: from OrbitCamera (as used in rt_gs_gui.py)
    # ------------------------------------------------------------------

    def from_orbit_camera(self, orbit_cam, region=None) -> Batch:
        """Build a Batch from an ``OrbitCamera`` instance.

        Args:
            orbit_cam: An ``OrbitCamera`` object with attributes:
                .fovy (degrees), .rot_mode, .pose_movecenter / .pose_objcenter

        Returns:
            Batch with rays in world space.
        """
        pose = orbit_cam.pose_movecenter if orbit_cam.rot_mode == 1 else orbit_cam.pose_objcenter
        # pose[:3,:3] = R_c2w (camera-to-world).
        # SAGA/COLMAP stores cam.R as R_c2w: dataset_readers.py does
        #   R = np.transpose(qvec2rotmat(...))
        # and getWorld2View2 transposes it again to get R_w2c.
        # Camera centre in world: C = -R_c2w @ T_w2c
        R_c2w = pose[:3, :3]
        T_w2c = pose[:3, 3]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w              # no transpose needed
        c2w[:3, 3] = -(R_c2w @ T_w2c)   # camera centre = -R_c2w @ T_w2c
        fovy_rad = math.radians(orbit_cam.fovy)
        return self._build_batch_from_c2w(c2w, fovy_rad, region=region)

    # ------------------------------------------------------------------
    # Public: from SAGA Camera object
    # ------------------------------------------------------------------

    def from_saga_camera(self, cam, region=None) -> Batch:
        """Build a Batch from a SAGA ``Camera`` object.

        Args:
            cam: SAGA Camera with .R (3,3), .T (3,), .FoVy, .image_width, .image_height

        Returns:
            Batch with rays in world space.
        """
        # SAGA Camera: cam.R = R_c2w (camera-to-world).
        # Confirmed: dataset_readers.py stores R = np.transpose(qvec2rotmat(...))
        # and getWorld2View2 transposes it internally to get R_w2c.
        # Camera centre in world: C = -R_c2w @ T_w2c
        R_c2w = cam.R  # (3, 3)  camera-to-world rotation
        T_w2c = cam.T  # (3,)    world-to-camera translation

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w              # no transpose needed
        c2w[:3, 3] = -(R_c2w @ T_w2c)   # camera centre = -R_c2w @ T_w2c

        return self._build_batch_from_c2w(c2w, cam.FoVy, region=region)

    # ------------------------------------------------------------------
    # Internal: build Batch from c2w + FoVy
    # ------------------------------------------------------------------

    def _build_batch_from_c2w(self, c2w: np.ndarray, fovy_rad: float, region=None) -> Batch:
        """Core ray generation from a 4×4 camera-to-world matrix.

        Args:
            c2w:      (4, 4) numpy array, column-major OpenGL convention.
            fovy_rad: Vertical field of view in radians.

        Returns:
            Batch object with rays_ori, rays_dir, T_to_world.
        """
        full_h, full_w = self.height, self.width
        if region is None:
            x0, y0, W, H = 0, 0, full_w, full_h
            grid_i, grid_j = self._grid_i, self._grid_j
        else:
            x0, y0, W, H = map(int, region)
            if x0 < 0 or y0 < 0 or W < 1 or H < 1 or x0 + W > full_w or y0 + H > full_h:
                raise ValueError(f"invalid tile region {region} for {full_w}x{full_h}")
            js = torch.arange(y0, y0 + H, dtype=torch.float32, device=self.device)
            ins = torch.arange(x0, x0 + W, dtype=torch.float32, device=self.device)
            grid_j, grid_i = torch.meshgrid(js, ins, indexing="ij")

        # Focal length from FoVy (square pixels: fx = fy, matches SAGA convention)
        fy = (full_h / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx, cy = full_w / 2.0, full_h / 2.0

        # Ray directions in camera space: (H, W, 3)
        # x right, y down, z into scene (OpenCV convention)
        dx = (grid_i + 0.5 - cx) / fx   # (H, W)
        dy = (grid_j + 0.5 - cy) / fy   # (H, W)
        dz = torch.ones_like(dx)               # (H, W)

        dirs_cam = torch.stack([dx, dy, dz], dim=-1)  # (H, W, 3)

        # Convert c2w to torch
        c2w_t = torch.tensor(c2w, dtype=torch.float32, device=self.device)  # (4, 4)

        # Official 3DGRT convention: rays are expressed in camera space;
        # the tracer transforms them exactly once using T_to_world.
        directions = torch.nn.functional.normalize(dirs_cam, dim=-1)
        origins = torch.zeros((H, W, 3), dtype=torch.float32, device=self.device)

        # T_to_world: (1, 4, 4)
        T_to_world = c2w_t.unsqueeze(0)  # (1, 4, 4)

        # OptiX tracer expects (1, H, W, 3) — add batch dimension
        return Batch(
            rays_ori=origins.unsqueeze(0),
            rays_dir=directions.unsqueeze(0).contiguous(),
            T_to_world=T_to_world,
        )
