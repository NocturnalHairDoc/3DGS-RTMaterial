import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from optix_integration.material_compositor import MATERIAL_PARAMS, MaterialCompositor
from optix_integration.ray_generator import RayGenerator
from optix_integration.optix_renderer import sanitize_normals
from clip_utils.material_clip import (aggregate_multiview_scores, masked_original_rgb_crop,
                                      params_from_text_prompt, select_confident_material)


class RayGeneratorTests(unittest.TestCase):
    def test_identity_camera_generates_normalized_forward_rays(self):
        generator = RayGenerator(width=3, height=3, device="cpu")
        camera = SimpleNamespace(R=np.eye(3, dtype=np.float32),
                                 T=np.zeros(3, dtype=np.float32), FoVy=math.pi / 2)
        batch = generator.from_saga_camera(camera)
        self.assertEqual(tuple(batch.rays_dir.shape), (1, 3, 3, 3))
        self.assertTrue(torch.allclose(batch.rays_dir.norm(dim=-1), torch.ones(1, 3, 3)))
        self.assertTrue(torch.allclose(batch.rays_dir[0, 1, 1], torch.tensor([0.0, 0.0, 1.0])))

    def test_camera_space_rays_leave_extrinsic_transform_to_tracer(self):
        generator = RayGenerator(width=1, height=1, device="cpu")
        camera = SimpleNamespace(R=np.eye(3, dtype=np.float32),
                                 T=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                                 FoVy=math.pi / 2)
        batch = generator.from_saga_camera(camera)
        self.assertTrue(torch.equal(batch.rays_ori, torch.zeros_like(batch.rays_ori)))
        self.assertTrue(torch.allclose(batch.T_to_world[0, :3, 3],
                                       torch.tensor([-1.0, -2.0, -3.0])))

    def test_rotated_camera_is_not_applied_twice(self):
        generator = RayGenerator(width=1, height=1, device="cpu")
        rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0]], dtype=np.float32)
        camera = SimpleNamespace(R=rotation, T=np.zeros(3, dtype=np.float32),
                                 FoVy=math.pi / 2)
        batch = generator.from_saga_camera(camera)
        self.assertTrue(torch.allclose(batch.rays_dir[0, 0, 0],
                                       torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(batch.T_to_world[0, :3, :3],
                                       torch.from_numpy(rotation)))

    def test_tile_rays_exactly_match_full_frame_slice(self):
        generator = RayGenerator(width=11, height=7, device="cpu")
        camera = SimpleNamespace(R=np.eye(3, dtype=np.float32),
                                 T=np.zeros(3, dtype=np.float32), FoVy=math.pi / 3)
        full = generator.from_saga_camera(camera)
        tile = generator.from_saga_camera(camera, region=(3, 2, 5, 4))
        self.assertTrue(torch.equal(tile.rays_dir, full.rays_dir[:, 2:6, 3:8]))
        self.assertTrue(torch.equal(tile.rays_ori, full.rays_ori[:, 2:6, 3:8]))

    def test_invalid_normals_are_zeroed_and_valid_normals_are_normalized(self):
        normals = torch.tensor([[[float("nan"), 1.0, 0.0], [0.0, 0.0, 2.0]]])
        clean = sanitize_normals(normals)
        self.assertTrue(torch.isfinite(clean).all())
        self.assertTrue(torch.equal(clean[0, 0], torch.zeros(3)))
        self.assertTrue(torch.allclose(clean[0, 1], torch.tensor([0.0, 0.0, 1.0])))


class MaterialTests(unittest.TestCase):
    def test_documented_presets_are_complete(self):
        self.assertEqual(set(MATERIAL_PARAMS), {"Default", "Metal", "Glass", "Plastic", "Matte"})
        self.assertEqual(MATERIAL_PARAMS["Matte"][2], 0.0)
        self.assertGreater(MATERIAL_PARAMS["Glass"][3], MATERIAL_PARAMS["Metal"][3])

    def test_shading_preserves_background_and_range(self):
        compositor = MaterialCompositor()
        rgb = torch.full((2, 2, 3), 0.5)
        normals = torch.tensor([0.0, 0.0, 1.0]).expand(2, 2, 3).clone()
        depth = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        opacity = torch.ones(2, 2, 1)
        result = compositor.shade(rgb, normals, depth, opacity, None, {})
        self.assertTrue(torch.equal(result[0, 1], rgb[0, 1]))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_material_prompt_combines_finish_and_colour(self):
        params = params_from_text_prompt("shiny gold metal")
        self.assertEqual(params["specular_gain"], 1.7)
        self.assertLess(params["saturation"], 1.0)
        self.assertEqual(params["tint"], [1.0800000429153442, 0.9599999785423279, 0.7200000286102295])

    def test_masked_crop_uses_original_rgb_and_neutral_outside(self):
        original = torch.zeros(3, 6, 7)
        original[0] = 0.8
        mask = torch.zeros(6, 7)
        mask[2:4, 3:5] = 1
        crop = masked_original_rgb_crop(original, mask, padding=1)
        self.assertEqual(tuple(crop.shape), (3, 4, 4))
        self.assertTrue(torch.allclose(crop[:, 0, 0], torch.full((3,), 0.5)))
        self.assertAlmostEqual(float(crop[0, 1, 1]), 0.8, places=6)

    def test_clip_topk_confidence_threshold_and_multiview_mean(self):
        scores = aggregate_multiview_scores([
            {"Metal": 0.30, "Glass": 0.20}, {"Metal": 0.26, "Glass": 0.22}])
        material, margin = select_confident_material(scores, min_score=0.15, min_margin=0.02)
        self.assertEqual(material, "Metal")
        self.assertAlmostEqual(margin, 0.07)
        material, _ = select_confident_material({"Metal": 0.20, "Glass": 0.195}, min_margin=0.02)
        self.assertIsNone(material)


if __name__ == "__main__":
    unittest.main()
