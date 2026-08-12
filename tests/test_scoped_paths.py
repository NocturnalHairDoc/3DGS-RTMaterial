import os
import unittest

from rt_gs_gui_sh_clip import SHMaterialViewer


class ScopedPathTests(unittest.TestCase):
    def test_allows_only_requested_v2_output_roots(self):
        path = SHMaterialViewer._scoped_path("./exports/test.png", "exports")
        self.assertTrue(path.endswith(os.path.join("exports", "test.png")))
        with self.assertRaises(ValueError):
            SHMaterialViewer._scoped_path("../old-output/test.png", "exports")


if __name__ == "__main__":
    unittest.main()
