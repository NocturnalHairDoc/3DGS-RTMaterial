import os
import unittest
from unittest.mock import patch

from render_policy import InteractiveRenderPolicy


class InteractiveRenderPolicyTests(unittest.TestCase):
    def test_uses_raster_only_during_camera_interaction(self):
        policy = InteractiveRenderPolicy(True)
        self.assertFalse(policy.use_raster(False, False))
        self.assertTrue(policy.use_raster(True, False))
        self.assertTrue(policy.use_raster(False, True))

    def test_environment_can_disable_adaptive_preview(self):
        with patch.dict(os.environ, {"RTM_ADAPTIVE_RT_PREVIEW": "0"}):
            policy = InteractiveRenderPolicy()
        self.assertFalse(policy.enabled)
        self.assertFalse(policy.use_raster(True, True))


if __name__ == "__main__":
    unittest.main()
