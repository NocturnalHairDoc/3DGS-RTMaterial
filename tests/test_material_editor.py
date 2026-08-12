import unittest
from types import SimpleNamespace

import torch

from material_sh_edit.material_sh_editor import MaterialSHEditor


class MaterialEditorTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            _features_dc=torch.nn.Parameter(torch.randn(5, 1, 3)),
            _features_rest=torch.nn.Parameter(torch.randn(5, 15, 3)),
            _opacity=torch.nn.Parameter(torch.randn(5, 1)),
        )
        self.original = tuple(x.detach().clone() for x in
                              (self.model._features_dc, self.model._features_rest, self.model._opacity))
        self.editor = MaterialSHEditor(self.model)
        self.mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)

    def assert_restored(self):
        for current, expected in zip((self.model._features_dc, self.model._features_rest,
                                      self.model._opacity), self.original):
            self.assertTrue(torch.equal(current, expected))

    def test_apply_and_restore_success(self):
        self.editor.apply_params(self.mask, {"tint": [1.1, .9, .8], "specular_gain": 2,
                                             "opacity_scale": .5, "strength": 1})
        self.assertFalse(torch.equal(self.model._features_dc, self.original[0]))
        self.editor.restore()
        self.assert_restored()

    def test_restore_after_renderer_exception(self):
        try:
            self.editor.apply_params(self.mask, {"specular_gain": 0, "strength": 1})
            raise RuntimeError("renderer failed")
        except RuntimeError:
            pass
        finally:
            self.editor.restore()
        self.assert_restored()


if __name__ == "__main__": unittest.main()
