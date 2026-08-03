import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "acr" / "mps_realtime.py"


def load_runtime(config_object=None):
    config_object = config_object or types.SimpleNamespace(
        demo_mode="webcam",
        input_size=512,
        render_size=512,
        save_dict_results=False,
        focal_length=1265,
        centermap_size=64,
    )

    config_module = types.ModuleType("acr.config")
    config_module.args = lambda: config_object
    model_module = types.ModuleType("acr.model")
    model_module.get_coord_maps = lambda size: torch.zeros(1, 2, size, size)
    utils_module = types.ModuleType("acr.utils")
    utils_module.batch_orth_proj = lambda x, camera, mode="2d", keep_dim=False: x
    utils_module.convert_kp2d_from_input_to_orgimg = lambda x, offsets: x
    utils_module.estimate_translation = lambda *args, **kwargs: torch.zeros(1, 3)

    previous = {name: sys.modules.get(name) for name in ("acr", "acr.config", "acr.model", "acr.utils")}
    package = types.ModuleType("acr")
    package.__path__ = [str(ROOT / "acr")]
    sys.modules.update({
        "acr": package,
        "acr.config": config_module,
        "acr.model": model_module,
        "acr.utils": utils_module,
    })
    try:
        spec = importlib.util.spec_from_file_location("acr.mps_realtime", MODULE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class RealtimeOptionsTests(unittest.TestCase):
    def test_realtime_profile_defaults_to_256_keypoints_and_mps_amp(self):
        runtime = load_runtime()
        config = types.SimpleNamespace(
            demo_mode="webcam",
            input_size=512,
            render_size=512,
            save_dict_results=False,
        )
        options = runtime.build_realtime_options(
            config,
            torch.device("mps"),
            environ={},
            argv=[],
        )
        self.assertEqual(options.profile, "realtime")
        self.assertEqual(options.input_size, 256)
        self.assertEqual(options.live_visualization, "keypoints")
        self.assertTrue(options.mps_amp)
        self.assertFalse(options.compute_camera_translation)

    def test_target_60_uses_two_to_one_display_to_inference_stride(self):
        runtime = load_runtime()
        config = types.SimpleNamespace(
            demo_mode="webcam",
            input_size=512,
            render_size=512,
            save_dict_results=False,
        )
        options = runtime.build_realtime_options(
            config,
            torch.device("mps"),
            environ={"ACR_TARGET_FPS": "60"},
            argv=[],
        )
        self.assertEqual(options.inference_stride, 2)

    def test_explicit_input_size_is_preserved(self):
        runtime = load_runtime()
        config = types.SimpleNamespace(
            demo_mode="webcam",
            input_size=384,
            render_size=512,
            save_dict_results=False,
        )
        options = runtime.build_realtime_options(
            config,
            torch.device("mps"),
            environ={},
            argv=["--input_size", "384"],
        )
        self.assertEqual(options.input_size, 384)

    def test_spatial_sizes_support_256_384_and_512(self):
        runtime = load_runtime()
        self.assertEqual(runtime.derive_spatial_sizes(256), (64, 32))
        self.assertEqual(runtime.derive_spatial_sizes(384), (96, 48))
        self.assertEqual(runtime.derive_spatial_sizes(512), (128, 64))
        with self.assertRaises(ValueError):
            runtime.derive_spatial_sizes(300)

    def test_keypoint_overlay_draws_without_resizing_frame(self):
        runtime = load_runtime()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        points = np.array([[[20 + index * 2, 20 + index] for index in range(21)]], dtype=np.float32)
        hand_types = np.array([0], dtype=np.int32)
        output = runtime.draw_keypoint_overlay(
            frame,
            points,
            hand_types,
            display_fps=60.0,
            inference_fps=30.0,
            label="MPS FP16 256",
        )
        self.assertEqual(output.shape, frame.shape)
        self.assertGreater(int(output.sum()), 0)
        self.assertEqual(int(frame.sum()), 0)

    def test_rate_meter_uses_duration_as_fps(self):
        runtime = load_runtime()
        meter = runtime.ExponentialRate(smoothing=1.0)
        self.assertAlmostEqual(meter.update_duration(0.02), 50.0)
        self.assertAlmostEqual(meter.update_duration(0.04), 25.0)

    def test_dynamic_part_head_matches_actual_parameter_map_size(self):
        runtime = load_runtime()

        class OffsetHead(torch.nn.Module):
            def forward(self, value):
                return value.new_zeros((value.shape[0], 6, 16, 1))

        class FakeModel:
            def __init__(self):
                self.contact_layers = [
                    None,
                    torch.nn.Identity(),
                    OffsetHead(),
                    OffsetHead(),
                    torch.nn.Conv2d(218, 109, 1),
                    torch.nn.Conv2d(218, 109, 1),
                ]
                self.cam_shape_layers = [
                    None,
                    torch.nn.Conv2d(256, 64, 1),
                    torch.nn.Linear(64 * 16, 10),
                    torch.nn.Linear(64 * 16, 10),
                ]

            @staticmethod
            def Hadamard_product(features, heatmaps):
                batch, joints, height, width = heatmaps.shape
                normalized = torch.softmax(heatmaps.reshape(batch, joints, -1), dim=-1)
                flattened = features.reshape(batch, -1, height * width)
                return torch.matmul(normalized, flattened.transpose(2, 1)).transpose(2, 1)

        fake = FakeModel()
        x = torch.randn(1, 256, 8, 8)
        pred_segm = torch.randn(1, 33, 16, 16)
        left = torch.randn(1, 109, 4, 4)
        right = torch.randn(1, 109, 4, 4)
        left_out, right_out, logits = runtime._dynamic_part_forward(
            fake, x, None, pred_segm, left, right
        )
        self.assertEqual(tuple(left_out.shape), (1, 109, 4, 4))
        self.assertEqual(tuple(right_out.shape), (1, 109, 4, 4))
        self.assertIs(logits, pred_segm)

    def test_latest_frame_worker_returns_inference_result(self):
        runtime = load_runtime()
        worker = runtime.LatestFrameWorker(lambda frame: int(frame.sum())).start()
        try:
            worker.submit(np.ones((4, 4, 3), dtype=np.uint8))
            result = None
            for _ in range(100):
                result = worker.latest()
                if result is not None:
                    break
                __import__("time").sleep(0.005)
            self.assertIsNotNone(result)
            self.assertIsNone(result.error)
            self.assertEqual(result.payload, 48)
            self.assertGreater(result.duration, 0.0)
        finally:
            worker.stop()

    def test_source_uses_dynamic_map_extent_and_skips_solvepnp_when_disabled(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn("map_height, map_width = l_params_maps.shape[-2:]", source)
        self.assertIn("if compute_camera_translation:", source)
        self.assertIn("cam_trans = vertices.new_zeros", source)


if __name__ == "__main__":
    unittest.main()
