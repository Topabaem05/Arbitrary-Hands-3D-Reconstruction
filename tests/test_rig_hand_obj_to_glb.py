import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.rig_hand_obj_to_glb import (
    BONE_NAMES,
    BONE_PARENTS,
    build_rigged_hand,
    parse_glb,
)


def write_tiny_hand_obj(path: Path) -> None:
    # Flat palm box plus five rectangular fingers. This is intentionally coarse;
    # the test verifies the asset contract rather than visual quality.
    vertices = []
    faces = []

    def box(x0, x1, y0, y1, z0=-0.5, z1=0.5):
        base = len(vertices) + 1
        vertices.extend([
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ])
        faces.extend([
            (base, base + 1, base + 2, base + 3),
            (base + 4, base + 7, base + 6, base + 5),
            (base, base + 4, base + 5, base + 1),
            (base + 1, base + 5, base + 6, base + 2),
            (base + 2, base + 6, base + 7, base + 3),
            (base + 4, base, base + 3, base + 7),
        ])

    box(-1, 3, -2, 2, -0.8, 0.8)
    box(-7, -1, 1.1, 2.0)
    box(-8, -1, 0.2, 1.0)
    box(-8.5, -1, -0.7, 0.1)
    box(-7.5, -1, -1.6, -0.8)
    box(-4, 0.5, -3.1, -2.0)
    lines = [*(f"v {x} {y} {z}" for x, y, z in vertices)]
    lines.extend("f " + " ".join(map(str, face)) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RigHandObjToGlbTests(unittest.TestCase):
    def test_bone_contract(self):
        self.assertEqual(len(BONE_NAMES), 21)
        self.assertEqual(BONE_NAMES[0], "wrist")
        self.assertEqual(BONE_NAMES[-1], "pinky_tip")
        self.assertEqual(BONE_PARENTS[0], -1)
        self.assertEqual(BONE_PARENTS[1], 0)
        self.assertEqual(BONE_PARENTS[5], 0)
        self.assertEqual(BONE_PARENTS[20], 19)

    def test_builds_valid_glb_with_normalized_top_four_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hand.obj"
            output = root / "hand.glb"
            write_tiny_hand_obj(source)
            report = build_rigged_hand(source, output, voxel_size=0.25)
            self.assertTrue(output.is_file())
            self.assertEqual(report.bone_count, 21)
            self.assertGreater(report.vertex_count, 0)
            self.assertGreater(report.triangle_count, 0)
            parsed = parse_glb(output)
            self.assertEqual(parsed["asset"]["version"], "2.0")
            self.assertEqual(len(parsed["skins"][0]["joints"]), 21)
            self.assertEqual(
                [node["name"] for node in parsed["nodes"][1:]],
                BONE_NAMES,
            )
            attributes = parsed["meshes"][0]["primitives"][0]["attributes"]
            self.assertIn("POSITION", attributes)
            self.assertIn("NORMAL", attributes)
            self.assertIn("JOINTS_0", attributes)
            self.assertIn("WEIGHTS_0", attributes)
            weights = np.asarray(report.weights)
            self.assertEqual(weights.shape[1], 4)
            self.assertTrue(np.all(np.isfinite(weights)))
            self.assertTrue(np.allclose(weights.sum(axis=1), 1.0, atol=1e-5))
            self.assertTrue(np.all(weights >= 0))
            self.assertTrue(np.all(report.joints >= 0))
            self.assertTrue(np.all(report.joints < 21))
            self.assertTrue(np.all(np.isfinite(report.joint_positions)))

    def test_glb_header_and_chunk_lengths_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hand.obj"
            output = root / "hand.glb"
            write_tiny_hand_obj(source)
            build_rigged_hand(source, output, voxel_size=0.3)
            raw = output.read_bytes()
            magic, version, total = struct.unpack_from("<4sII", raw, 0)
            self.assertEqual(magic, b"glTF")
            self.assertEqual(version, 2)
            self.assertEqual(total, len(raw))
            json_length, json_type = struct.unpack_from("<I4s", raw, 12)
            self.assertEqual(json_type, b"JSON")
            json.loads(raw[20:20 + json_length].decode("utf-8").rstrip(" \x00"))


if __name__ == "__main__":
    unittest.main()
