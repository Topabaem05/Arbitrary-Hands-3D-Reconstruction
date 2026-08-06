#!/usr/bin/env python3
"""Convert locally licensed MANO pickle data into a private browser bundle.

The output is intended for local browser import and must not be committed or
redistributed unless the user's MANO license explicitly permits it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

MANO_BUNDLE_FORMAT = "mano-browser-bundle-v1"
VERTEX_COUNT = 778
SKIN_JOINT_COUNT = 16
LANDMARK_COUNT = 21
JOINT_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
LANDMARK_REORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
TIP_INDICES = {
    "left": [745, 317, 445, 556, 673],
    "right": [745, 317, 444, 556, 673],
}


def validate_export_policy(acknowledge_license: bool) -> None:
    if not acknowledge_license:
        raise PermissionError(
            "Creating a browser MANO bundle requires --acknowledge-mano-license. "
            "The output is private and must not be redistributed unless the license permits it."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rounded_rows(tensor: torch.Tensor, digits: int = 8) -> list[list[float]]:
    scale = float(10**digits)
    rounded = torch.round(tensor.detach().cpu().float() * scale) / scale
    return rounded.tolist()


def _validated_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.detach().cpu().float().clamp_min(0.0)
    if tuple(weights.shape) != (VERTEX_COUNT, SKIN_JOINT_COUNT):
        raise ValueError(f"Expected weights {(VERTEX_COUNT, SKIN_JOINT_COUNT)}, got {tuple(weights.shape)}")
    totals = weights.sum(dim=1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError("Every MANO vertex must have at least one positive skinning weight.")
    return weights / totals


def build_hand_payload(layer: Any, side: str) -> dict[str, Any]:
    if side not in TIP_INDICES:
        raise ValueError(f"Unsupported MANO side: {side}")

    vertices = layer.th_v_template.detach().cpu().float()
    if vertices.ndim == 3:
        vertices = vertices[0]
    if tuple(vertices.shape) != (VERTEX_COUNT, 3):
        raise ValueError(f"Expected MANO vertices {(VERTEX_COUNT, 3)}, got {tuple(vertices.shape)}")

    faces = layer.th_faces.detach().cpu().long()
    if faces.ndim == 3:
        faces = faces[0]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces, got {tuple(faces.shape)}")
    if int(faces.min()) < 0 or int(faces.max()) >= VERTEX_COUNT:
        raise ValueError("MANO face index is outside the 778-vertex range.")

    weights = _validated_weights(layer.th_weights)
    joint_regressor = layer.th_J_regressor.detach().cpu().float()
    if tuple(joint_regressor.shape) != (SKIN_JOINT_COUNT, VERTEX_COUNT):
        raise ValueError(
            f"Expected joint regressor {(SKIN_JOINT_COUNT, VERTEX_COUNT)}, got {tuple(joint_regressor.shape)}"
        )

    rest_skin_joints = joint_regressor @ vertices
    tips = vertices[torch.tensor(TIP_INDICES[side], dtype=torch.long)]
    raw_landmarks = torch.cat((rest_skin_joints, tips), dim=0)
    rest_landmarks = raw_landmarks[torch.tensor(LANDMARK_REORDER, dtype=torch.long)]

    return {
        "side": side,
        "vertices": _rounded_rows(vertices),
        "faces": faces.tolist(),
        "weights": _rounded_rows(weights),
        "restSkinJoints": _rounded_rows(rest_skin_joints),
        "restLandmarks": _rounded_rows(rest_landmarks),
    }


def build_bundle_from_layers(
    left_layer: Any,
    right_layer: Any,
    *,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    left = build_hand_payload(left_layer, "left")
    right = build_hand_payload(right_layer, "right")
    if len(left["faces"]) != len(right["faces"]):
        raise ValueError("Left and right MANO meshes must have the same face count.")

    return {
        "format": MANO_BUNDLE_FORMAT,
        "private": True,
        "redistributable": False,
        "vertexCount": VERTEX_COUNT,
        "faceCount": len(left["faces"]),
        "skinJointCount": SKIN_JOINT_COUNT,
        "landmarkCount": LANDMARK_COUNT,
        "jointParents": JOINT_PARENTS,
        "sourceSha256": dict(source_hashes),
        "hands": {"left": left, "right": right},
        "notes": [
            "Generated from locally licensed MANO files.",
            "Import locally in the browser; do not upload or redistribute unless permitted by your MANO license.",
        ],
    }


def write_bundle_atomic(output: Path, bundle: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def convert_mano_bundle(
    mano_root: Path,
    output: Path,
    *,
    acknowledge_license: bool,
) -> dict[str, Any]:
    validate_export_policy(acknowledge_license)
    left_path = mano_root / "MANO_LEFT.pkl"
    right_path = mano_root / "MANO_RIGHT.pkl"
    for path in (left_path, right_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    from mano.manolayer import ManoLayer

    left_layer = ManoLayer(
        ncomps=45,
        center_idx=None,
        side="left",
        mano_root=str(mano_root),
        use_pca=False,
        flat_hand_mean=False,
    ).eval()
    right_layer = ManoLayer(
        ncomps=45,
        center_idx=None,
        side="right",
        mano_root=str(mano_root),
        use_pca=False,
        flat_hand_mean=False,
    ).eval()

    bundle = build_bundle_from_layers(
        left_layer,
        right_layer,
        source_hashes={
            "left": sha256_file(left_path),
            "right": sha256_file(right_path),
        },
    )
    write_bundle_atomic(output, bundle)
    return bundle


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano-root", type=Path, default=Path("mano"))
    parser.add_argument("--output", type=Path, default=Path("mano-browser-bundle.json"))
    parser.add_argument("--acknowledge-mano-license", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    bundle = convert_mano_bundle(
        args.mano_root,
        args.output,
        acknowledge_license=args.acknowledge_mano_license,
    )
    print(
        f"Wrote private {bundle['format']} with {bundle['vertexCount']} vertices per hand to {args.output}"
    )
    print("Do not commit or redistribute this file unless your MANO license permits it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
