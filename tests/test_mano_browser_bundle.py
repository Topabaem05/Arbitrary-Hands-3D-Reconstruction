import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.convert_mano_browser_bundle import (
    MANO_BUNDLE_FORMAT,
    build_bundle_from_layers,
    validate_export_policy,
    write_bundle_atomic,
)


class FakeLayer:
    def __init__(self, side: str):
        vertices = torch.zeros(1, 778, 3, dtype=torch.float32)
        vertices[0, :, 0] = torch.linspace(-0.1, 0.1, 778)
        vertices[0, :, 1] = torch.linspace(0.0, 0.2, 778)
        if side == "left":
            vertices[0, :, 0] *= -1
        self.th_v_template = vertices
        self.th_faces = torch.tensor([[0, 1, 2], [2, 3, 4], [775, 776, 777]], dtype=torch.long)
        self.th_weights = torch.zeros(778, 16, dtype=torch.float32)
        for vertex in range(778):
            self.th_weights[vertex, vertex % 16] = 0.75
            self.th_weights[vertex, (vertex + 1) % 16] = 0.25
        self.th_J_regressor = torch.zeros(16, 778, dtype=torch.float32)
        for joint in range(16):
            self.th_J_regressor[joint, min(joint * 40, 777)] = 1.0


class ManoBrowserBundleTests(unittest.TestCase):
    def test_license_acknowledgement_is_required(self):
        with self.assertRaises(PermissionError):
            validate_export_policy(False)
        validate_export_policy(True)

    def test_build_bundle_has_static_private_contract(self):
        bundle = build_bundle_from_layers(FakeLayer("left"), FakeLayer("right"), source_hashes={"left": "a", "right": "b"})

        self.assertEqual(bundle["format"], MANO_BUNDLE_FORMAT)
        self.assertTrue(bundle["private"])
        self.assertFalse(bundle["redistributable"])
        self.assertEqual(bundle["vertexCount"], 778)
        self.assertEqual(bundle["skinJointCount"], 16)
        self.assertEqual(bundle["landmarkCount"], 21)
        self.assertEqual(bundle["jointParents"], [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])

        for side in ("left", "right"):
            hand = bundle["hands"][side]
            self.assertEqual(len(hand["vertices"]), 778)
            self.assertEqual(len(hand["vertices"][0]), 3)
            self.assertEqual(len(hand["weights"]), 778)
            self.assertEqual(len(hand["weights"][0]), 16)
            self.assertEqual(len(hand["restSkinJoints"]), 16)
            self.assertEqual(len(hand["restLandmarks"]), 21)
            self.assertTrue(all(0 <= index < 778 for face in hand["faces"] for index in face))
            for row in hand["weights"][:32]:
                self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_atomic_writer_produces_valid_json(self):
        bundle = build_bundle_from_layers(FakeLayer("left"), FakeLayer("right"), source_hashes={"left": "a", "right": "b"})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mano-browser-bundle.json"
            write_bundle_atomic(output, bundle)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["format"], MANO_BUNDLE_FORMAT)
            self.assertFalse(output.with_suffix(output.suffix + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
