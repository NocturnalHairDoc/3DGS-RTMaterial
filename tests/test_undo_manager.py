import unittest

from undo_manager import UndoManager


class UndoManagerTests(unittest.TestCase):
    def test_undo_redo_and_new_branch(self):
        history = UndoManager(limit=4)
        history.record({"value": 1})
        history.record({"value": 2})
        self.assertEqual(history.undo()["value"], 1)
        self.assertEqual(history.redo()["value"], 2)
        history.undo()
        history.record({"value": 3})
        self.assertFalse(history.can_redo)

    def test_coalesces_camera_and_bounds_memory(self):
        history = UndoManager(limit=3, coalesce_seconds=1.0)
        history.record({"value": 0})
        history.record({"value": 1}, key="camera", now=1.0)
        history.record({"value": 2}, key="camera", now=1.2)
        self.assertEqual(history.undo()["value"], 0)
        for value in range(5):
            history.record({"value": value})
        self.assertLessEqual(len(history._undo), 3)


if __name__ == "__main__":
    unittest.main()
