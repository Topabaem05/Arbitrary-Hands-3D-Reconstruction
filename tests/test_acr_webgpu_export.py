import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "export_acr_webgpu.py"
spec = importlib.util.spec_from_file_location("export_acr_webgpu", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class AcrWebGpuExportTests(unittest.TestCase):
    def test_static_decoder_selects_peak_and_applies_cross_prior(self):
        params_left = torch.zeros(1, 109, 4, 4)
        params_right = torch.zeros(1, 109, 4, 4)
        centers_left = torch.zeros(1, 1, 4, 4)
        centers_right = torch.zeros(1, 1, 4, 4)
        priors_left = torch.zeros(1, 106, 4, 4)
        priors_right = torch.zeros(1, 106, 4, 4)

        centers_left[0, 0, 1, 2] = 0.9
        centers_right[0, 0, 2, 1] = 0.8
        params_left[0, :, 1, 2] = 2.0
        params_right[0, :, 2, 1] = 3.0
        priors_left[0, :, 2, 1] = 0.5
        priors_right[0, :, 1, 2] = 0.25

        decoded = module.decode_static_maps(
            params_left,
            params_right,
            centers_left,
            centers_right,
            priors_left,
            priors_right,
            threshold=0.35,
        )

        self.assertEqual(tuple(decoded.params.shape), (1, 2, 109))
        self.assertTrue(torch.allclose(decoded.params[0, 0, :3], torch.full((3,), 2.0)))
        self.assertTrue(torch.allclose(decoded.params[0, 0, 3:], torch.full((106,), 2.5)))
        self.assertTrue(torch.allclose(decoded.params[0, 1, 3:], torch.full((106,), 3.25)))
        self.assertTrue(torch.equal(decoded.centers[0, 0], torch.tensor([2.0, 1.0])))
        self.assertTrue(torch.equal(decoded.centers[0, 1], torch.tensor([1.0, 2.0])))

    def test_low_confidence_hand_is_zeroed_without_dynamic_outputs(self):
        params = torch.ones(1, 109, 2, 2)
        centers = torch.zeros(1, 1, 2, 2)
        priors = torch.ones(1, 106, 2, 2)
        decoded = module.decode_static_maps(
            params,
            params,
            centers,
            centers,
            priors,
            priors,
            threshold=0.35,
        )
        self.assertEqual(tuple(decoded.params.shape), (1, 2, 109))
        self.assertEqual(decoded.valid.sum().item(), 0)
        self.assertEqual(decoded.params.abs().sum().item(), 0)

    def test_mano_embedding_requires_explicit_acknowledgement(self):
        with self.assertRaisesRegex(PermissionError, "MANO"):
            module.validate_export_policy(include_mano=True, acknowledge_mano_license=False)
        module.validate_export_policy(include_mano=True, acknowledge_mano_license=True)
        module.validate_export_policy(include_mano=False, acknowledge_mano_license=False)

    def test_manifest_records_private_artifact_and_static_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "acr.onnx"
            output.write_bytes(b"onnx")
            manifest = module.build_export_manifest(
                output=output,
                input_size=256,
                include_mano=False,
                checkpoint_sha256="abc",
            )
        self.assertEqual(manifest["input"]["shape"], [1, 256, 256, 3])
        self.assertEqual(manifest["outputs"]["params"], [1, 2, 109])
        self.assertFalse(manifest["redistributable"])
        self.assertEqual(manifest["checkpoint_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
