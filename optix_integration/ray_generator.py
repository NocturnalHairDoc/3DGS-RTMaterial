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

    def from_orbit_camera(self, orbit_cam) -> Batch:
        """Build a Batch from an ``OrbitCamera`` instance.

        Args:
            orbit_cam: An ``OrbitCamera`` object with attributes:
                .fovy (degrees), .rot_mode, .pose_movecenter / .pose_objcenter

        Returns:
            Batch with rays in world space.
        """
        pose = orbit_cam.pose_movecenter if orbit_cam.rot_mode == 1 else orbit_cam.pose_objcenter
        # pose is w2c (COLMAP convention): R=pose[:3,:3] world->cam, T=pose[:3,3] cam-space translation
        # Convert to c2w for ray generation: R_c2w = R.T, cam_center = -R.T @ T
        R_w2c = pose[:3, :3]
        T_w2c = pose[:3, 3]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_w2c.T
        c2w[:3, 3] = -(R_w2c.T @ T_w2c)
        fovy_rad = math.radians(orbit_cam.fovy)
        return self._build_batch_from_c2w(c2w, fovy_rad)

    # ------------------------------------------------------------------
    # Public: from SAGA Camera object
    # ------------------------------------------------------------------

    def from_saga_camera(self, cam) -> Batch:
        """Build a Batch from a SAGA ``Camera`` object.

        Args:
            cam: SAGA Camera with .R (3,3), .T (3,), .FoVy, .image_width, .image_height

        Returns:
            Batch with rays in world space.
        """
        # SAGA Camera: R is world-to-camera rotation, T is world-to-camera translation
        # c2w = [R^T | -R^T T; 0 0 0 1]
        R = cam.R  # (3, 3)  world-to-cam rotation
        T = cam.T  # (3,)    world-to-cam translation

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -(R.T @ T)

        return self._build_batch_from_c2w(c2w, cam.FoVy)

    # ------------------------------------------------------------------
    # Internal: build Batch from c2w + FoVy
    # ------------------------------------------------------------------

    def _build_batch_from_c2w(self, c2w: np.ndarray, fovy_rad: float) -> Batch:
        """Core ray generation from a 4×4 camera-to-world matrix.

        Args:
            c2w:      (4, 4) numpy array, column-major OpenGL convention.
            fovy_rad: Vertical field of view in radians.

        Returns:
            Batch object with rays_ori, rays_dir, T_to_world.
        """
        H, W = self.height, self.width

        # Focal length from FoVy (square pixels: fx = fy, matches SAGA convention)
        fy = (H / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx, cy = W / 2.0, H / 2.0

        # Ray directions in camera space: (H, W, 3)
        # x right, y down, z into scene (OpenCV convention)
        dx = (self._grid_i + 0.5 - cx) / fx   # (H, W)
        dy = (self._grid_j + 0.5 - cy) / fy   # (H, W)
        dz = torch.ones_like(dx)               # (H, W)

        dirs_cam = torch.stack([dx, dy, dz], dim=-1)  # (H, W, 3)

        # Convert c2w to torch
        c2w_t = torch.tensor(c2w, dtype=torch.float32, device=self.device)  # (4, 4)

        # Rotate directions to world space
        R_c2w = c2w_t[:3, :3]  # (3, 3)
        # (H, W, 3) @ (3, 3)^T = (H, W, 3)
        dirs_world = (dirs_cam @ R_c2w.T)
        dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1)

        # Camera origin in world space — same for all pixels (pinhole)
        origin = c2w_t[:3, 3]  # (3,)
        origins = origin.view(1, 1, 3).expand(H, W, 3).contiguous()

        # T_to_world: (1, 4, 4)
        T_to_world = c2w_t.unsqueeze(0)  # (1, 4, 4)

        # OptiX tracer expects (1, H, W, 3) — add batch dimension
        return Batch(
            rays_ori=origins.unsqueeze(0),
            rays_dir=dirs_world.unsqueeze(0).contiguous(),
            T_to_world=T_to_world,
        )
