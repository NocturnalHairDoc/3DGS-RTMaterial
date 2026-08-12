import threading
import time
import unittest

import numpy as np

from export_manager import ExportCancelled, ExportManager, estimate_frame_bytes, linear_to_srgb


class ExportManagerTests(unittest.TestCase):
    def _wait(self, manager):
        deadline = time.monotonic() + 3
        events = []
        while time.monotonic() < deadline:
            events.extend(manager.drain_events())
            if events and events[-1].kind in {"completed", "cancelled", "failed"}:
                return events
            time.sleep(0.01)
        self.fail("export worker did not finish")

    def test_colour_conversion_and_estimate(self):
        values = linear_to_srgb(np.array([0.0, 0.0031308, 0.18, 1.0]))
        self.assertAlmostEqual(float(values[0]), 0.0, places=6)
        self.assertAlmostEqual(float(values[-1]), 1.0, places=6)
        self.assertAlmostEqual(float(values[2]), 0.461356, places=5)
        self.assertEqual(estimate_frame_bytes(64, 48), 64 * 48 * 4 * 4 * 8)

    def test_export_cancellation(self):
        manager = ExportManager()
        entered = threading.Event()
        def work(progress, cancelled):
            entered.set()
            while not cancelled.wait(0.01):
                pass
            raise ExportCancelled("cancelled by test")
        manager.start(3, work)
        self.assertTrue(entered.wait(1))
        manager.cancel()
        self.assertEqual(self._wait(manager)[-1].kind, "cancelled")

    def test_export_exception_recovery_allows_restart(self):
        manager = ExportManager()
        manager.start(1, lambda progress, cancel: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertEqual(self._wait(manager)[-1].kind, "failed")
        manager.start(1, lambda progress, cancel: progress(1))
        self.assertEqual(self._wait(manager)[-1].kind, "completed")


if __name__ == "__main__":
    unittest.main()
