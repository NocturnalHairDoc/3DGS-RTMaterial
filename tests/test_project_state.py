import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_state import STATE_VERSION, load_project_state, migrate_metadata, model_fingerprint, save_project_state


class ProjectStateTests(unittest.TestCase):
    def test_round_trip_preserves_mask_and_nested_material_metadata(self):
        mask = np.array([1, 2, 2, 3], dtype=np.int32)
        metadata = {
            "point_count": 4,
            "segment_times": 2,
            "material_assignments": {
                "1": {"type": "Custom", "name": "gold", "params": {"tint": [1.0, 0.8, 0.2]}},
            },
            "hidden_segments": [2],
            "camera": {"rotation_xyzw": [0, 0, 0, 1], "center": [1, 2, 3]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_project_state(Path(directory) / "scene", mask, metadata)
            loaded_mask, loaded_metadata = load_project_state(path)
        np.testing.assert_array_equal(loaded_mask, mask)
        self.assertEqual(loaded_metadata["material_assignments"], metadata["material_assignments"])
        self.assertEqual(loaded_metadata["hidden_segments"], [2])
        self.assertEqual(loaded_metadata["version"], STATE_VERSION)
        self.assertIn("saved_at", loaded_metadata)

    def test_state_schema_migration_preserves_material_parameters(self):
        old = {"version": 1, "material_assignments": {
            "1": {"type": "Custom", "params": {"strength": 0.4, "tint": [1, 0.5, 0.2]}}
        }}
        migrated = migrate_metadata(old)
        self.assertEqual(migrated["version"], 2)
        self.assertEqual(migrated["material_assignments"]["1"]["params"]["strength"], 0.4)

    def test_model_fingerprint_changes_with_model_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "point_cloud/iteration_1/scene_point_cloud.ply"
            feature = root / "point_cloud/iteration_2/contrastive_feature_point_cloud.ply"
            gate = root / "point_cloud/iteration_2/scale_gate.pt"
            for path, data in ((scene, b"scene-a"), (feature, b"feature"), (gate, b"gate")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            first = model_fingerprint(root, 1, 2)
            scene.write_bytes(b"scene-b")
            self.assertNotEqual(first, model_fingerprint(root, 1, 2))

    def test_rejects_wrong_mask_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez_compressed(path, mask=np.zeros((2, 2), dtype=np.int32),
                                metadata=np.asarray('{"version": 1}'))
            with self.assertRaisesRegex(ValueError, "mask shape"):
                load_project_state(path)


if __name__ == "__main__":
    unittest.main()
