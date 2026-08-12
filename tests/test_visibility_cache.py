import types
import unittest
from unittest.mock import Mock

import torch

from rt_gs_gui_sh_clip import SHMaterialViewer


class VisibilityCacheTests(unittest.TestCase):
    def test_reuses_mask_until_hidden_set_changes(self):
        viewer = SHMaterialViewer.__new__(SHMaterialViewer)
        mask = torch.tensor([0, 2, 3], dtype=torch.int32)
        scene = types.SimpleNamespace(_mask=mask, segment_times=2)
        viewer.engine = {"scene": scene}
        viewer.hidden_segments = {1}
        viewer._visibility_cache_key = None
        viewer._visibility_cache = None
        viewer._get_hide_mask = Mock(return_value=torch.tensor([False, True, False]))
        first = viewer._visible_gaussian_mask()
        second = viewer._visible_gaussian_mask()
        self.assertIs(first, second)
        self.assertEqual(viewer._get_hide_mask.call_count, 1)
        viewer.hidden_segments = {2}
        viewer._visible_gaussian_mask()
        self.assertEqual(viewer._get_hide_mask.call_count, 2)


if __name__ == "__main__":
    unittest.main()
