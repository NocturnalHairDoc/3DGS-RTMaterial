import os
import subprocess
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CLITests(unittest.TestCase):
    def test_help_does_not_initialize_cuda(self):
        for entry_point in ("rt_gs_gui.py", "rt_gs_gui_sh_clip.py"):
            with self.subTest(entry_point=entry_point):
                result = subprocess.run(
                    [sys.executable, os.path.join(ROOT, entry_point), "--help"],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--model_path", result.stdout)


if __name__ == "__main__":
    unittest.main()
