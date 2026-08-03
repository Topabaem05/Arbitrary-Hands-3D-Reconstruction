import ast
import logging
from pathlib import Path
import types
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "acr" / "main.py"


def load_function(name, extra_globals=None):
    tree = ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(extra_globals or {})
    exec(compile(module, str(MAIN), "exec"), namespace)
    return namespace[name]


class FakeTorch:
    def __init__(self, cuda_available, mps_available):
        self.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
        self.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available)
        )

    @staticmethod
    def device(name):
        return name


class MPSMainTests(unittest.TestCase):
    def test_select_inference_device_prefers_cuda_then_mps_then_cpu(self):
        selector = load_function("select_inference_device", {"torch": FakeTorch(True, True)})
        self.assertEqual(selector(), "cuda")

        selector = load_function("select_inference_device", {"torch": FakeTorch(False, True)})
        self.assertEqual(selector(), "mps")

        selector = load_function("select_inference_device", {"torch": FakeTorch(False, False)})
        self.assertEqual(selector(), "cpu")

    def test_move_to_device_recurses_without_changing_metadata(self):
        mover = load_function("move_to_device", {"torch": torch})
        nested = {
            "image": torch.tensor([1.0]),
            "items": [torch.tensor([2]), (torch.tensor([3]), "keep")],
            "name": "frame.jpg",
        }

        moved = mover(nested, torch.device("cpu"))

        self.assertEqual(moved["image"].device.type, "cpu")
        self.assertEqual(moved["items"][0].device.type, "cpu")
        self.assertEqual(moved["items"][1][0].device.type, "cpu")
        self.assertEqual(moved["items"][1][1], "keep")
        self.assertEqual(moved["name"], "frame.jpg")

    def test_legacy_cuda_redirect_maps_tensor_cuda_to_selected_device(self):
        redirect = load_function(
            "redirect_legacy_tensor_cuda",
            {"torch": torch, "logging": logging},
        )
        original_cuda = torch.Tensor.cuda
        try:
            redirect(torch.device("cpu"))
            tensor = torch.tensor([7])
            self.assertEqual(tensor.cuda().device.type, "cpu")
            self.assertEqual(tensor.cuda().item(), 7)
        finally:
            torch.Tensor.cuda = original_cuda

    def test_mps_fallback_is_configured_before_torch_import(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertLess(
            source.index("PYTORCH_ENABLE_MPS_FALLBACK"),
            source.index("import torch"),
        )

    def test_model_build_uses_dataparallel_only_for_cuda(self):
        source = MAIN.read_text(encoding="utf-8")
        self.assertIn("if self.device.type == 'cuda':", source)
        self.assertIn("model = nn.DataParallel(model)", source)
        self.assertNotIn("model.cuda()", source)
        self.assertNotIn("MANOWrapper().cuda()", source)

    def test_preprocessed_metadata_is_moved_before_inference(self):
        source = MAIN.read_text(encoding="utf-8")
        move_index = source.index("meta_data = move_to_device(meta_data, self.device)")
        inference_index = source.index("outputs = self.model(meta_data, **self.demo_cfg)")
        self.assertLess(move_index, inference_index)

    def test_main_source_compiles(self):
        source = MAIN.read_text(encoding="utf-8")
        compile(source, str(MAIN), "exec")


if __name__ == "__main__":
    unittest.main()
