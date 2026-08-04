#!/usr/bin/env python3
"""Export the ACR inference core to a fixed-shape ONNX model for WebGPU.

The default export contains only the ACR neural network and a static left/right
map decoder. It does not contain MANO assets. `--include-mano` embeds the
caller's locally licensed MANO data into the output and therefore requires an
explicit acknowledgement; generated artifacts remain private by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticDecoded(NamedTuple):
    params: torch.Tensor
    scores: torch.Tensor
    centers: torch.Tensor
    valid: torch.Tensor


def _sample_map(feature_map: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
    batch, channels, _, _ = feature_map.shape
    flattened = feature_map.reshape(batch, channels, -1)
    index = flat_index.reshape(batch, 1, 1).expand(batch, channels, 1)
    return torch.gather(flattened, 2, index).squeeze(-1)


def _peak(center_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, height, width = center_map.shape
    pooled = F.max_pool2d(center_map, kernel_size=5, stride=1, padding=2)
    suppressed = center_map * (center_map == pooled).to(center_map.dtype)
    scores, indices = suppressed.reshape(center_map.shape[0], -1).max(dim=1)
    x = torch.remainder(indices, width).to(center_map.dtype)
    y = torch.div(indices, width, rounding_mode="floor").to(center_map.dtype)
    return scores, indices, torch.stack((x, y), dim=1)


def decode_static_maps(
    left_params_map: torch.Tensor,
    right_params_map: torch.Tensor,
    left_center_map: torch.Tensor,
    right_center_map: torch.Tensor,
    left_prior_map: torch.Tensor | None,
    right_prior_map: torch.Tensor | None,
    *,
    threshold: float,
) -> StaticDecoded:
    """Decode exactly one left and one right hand with static output shapes."""
    left_score, left_index, left_center = _peak(left_center_map)
    right_score, right_index, right_center = _peak(right_center_map)
    left_params = _sample_map(left_params_map, left_index)
    right_params = _sample_map(right_params_map, right_index)

    if left_prior_map is not None and right_prior_map is not None:
        left_cross_prior = _sample_map(left_prior_map, right_index)
        right_cross_prior = _sample_map(right_prior_map, left_index)
        map_width = left_center_map.shape[-1]
        center_distance = torch.linalg.vector_norm(left_center - right_center, dim=1)
        both_valid = (left_score >= threshold) & (right_score >= threshold)
        close_enough = center_distance <= float(map_width) / 2.0
        cross_mask = (both_valid & close_enough).to(left_params.dtype).unsqueeze(1)
        left_params = torch.cat(
            (left_params[:, :3], left_params[:, 3:] + left_cross_prior * cross_mask),
            dim=1,
        )
        right_params = torch.cat(
            (right_params[:, :3], right_params[:, 3:] + right_cross_prior * cross_mask),
            dim=1,
        )

    valid = torch.stack((left_score >= threshold, right_score >= threshold), dim=1)
    params = torch.stack((left_params, right_params), dim=1)
    params = params * valid.to(params.dtype).unsqueeze(-1)
    scores = torch.stack((left_score, right_score), dim=1)
    centers = torch.stack((left_center, right_center), dim=1)
    return StaticDecoded(params=params, scores=scores, centers=centers, valid=valid)


def validate_export_policy(*, include_mano: bool, acknowledge_mano_license: bool) -> None:
    if include_mano and not acknowledge_mano_license:
        raise PermissionError(
            "Embedding MANO into ONNX requires --acknowledge-mano-license. "
            "The generated model must not be redistributed unless your MANO license permits it."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_export_manifest(
    *,
    output: Path,
    input_size: int,
    include_mano: bool,
    checkpoint_sha256: str,
) -> dict[str, object]:
    outputs: dict[str, list[int]] = {
        "params": [1, 2, 109],
        "scores": [1, 2],
        "centers": [1, 2, 2],
        "valid": [1, 2],
    }
    if include_mano:
        outputs.update({"vertices": [1, 2, 778, 3], "joints": [1, 2, 21, 3]})
    return {
        "format": "acr-webgpu-static-v1",
        "model": output.name,
        "input": {"name": "image", "shape": [1, input_size, input_size, 3], "dtype": "float32", "range": [0.0, 1.0]},
        "outputs": outputs,
        "includes_mano": include_mano,
        "redistributable": False,
        "checkpoint_sha256": checkpoint_sha256,
        "notes": [
            "Static batch size 1 with one left-hand slot and one right-hand slot.",
            "Invalid hand slots are zeroed and identified by the valid output.",
            "Review the ACR checkpoint license before sharing this artifact.",
            "If includes_mano is true, MANO redistribution restrictions also apply.",
        ],
    }


class AcrNeuralCore(nn.Module):
    def __init__(self, model: nn.Module, threshold: float) -> None:
        super().__init__()
        self.model = model
        self.threshold = threshold

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.model.backbone(image * 255.0)
        outputs = self.model.head_forward(features)
        decoded = decode_static_maps(
            outputs["l_params_maps"],
            outputs["r_params_maps"],
            outputs["l_center_map"],
            outputs["r_center_map"],
            outputs.get("l_prior_maps"),
            outputs.get("r_prior_maps"),
            threshold=self.threshold,
        )
        return decoded.params, decoded.scores, decoded.centers, decoded.valid


class AcrManoCore(nn.Module):
    def __init__(self, neural_core: AcrNeuralCore, left_mano: nn.Module, right_mano: nn.Module, rot6d_to_angular) -> None:
        super().__init__()
        self.neural_core = neural_core
        self.left_mano = left_mano
        self.right_mano = right_mano
        self.rot6d_to_angular = rot6d_to_angular

    def _mano(self, params: torch.Tensor, layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        global_rotation = self.rot6d_to_angular(params[:, 3:9])
        hand_pose = self.rot6d_to_angular(params[:, 9:99])
        pose = torch.cat((global_rotation, hand_pose), dim=1)
        vertices, joints, _ = layer(pose, th_betas=params[:, 99:109])
        return vertices, joints

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        params, scores, centers, valid = self.neural_core(image)
        left_vertices, left_joints = self._mano(params[:, 0], self.left_mano)
        right_vertices, right_joints = self._mano(params[:, 1], self.right_mano)
        vertices = torch.stack((left_vertices, right_vertices), dim=1)
        joints = torch.stack((left_joints, right_joints), dim=1)
        gate = valid.to(vertices.dtype).unsqueeze(-1).unsqueeze(-1)
        return params, scores, centers, valid, vertices * gate, joints * gate


def _load_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        payload = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")
    current = model.state_dict()
    loaded = 0
    for key, destination in current.items():
        source = payload.get(key)
        if source is None:
            source = payload.get(f"module.{key}")
        if source is not None and source.shape == destination.shape:
            destination.copy_(source)
            loaded += 1
    if loaded == 0:
        raise RuntimeError("No checkpoint tensors matched the ACR model.")
    missing = len(current) - loaded
    print(f"Loaded {loaded}/{len(current)} tensors ({missing} unmatched).")


def _construct_export_model(args: argparse.Namespace) -> nn.Module:
    # acr.config parses sys.argv at import time. Give it only known inference
    # arguments, then restore the exporter's command line.
    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--configs_yml", args.config,
        "--model_precision", "fp32",
        "--input_size", str(args.input_size),
        "--demo_mode", "image",
    ]
    try:
        from acr.model import ACR
        from acr.mps_realtime import patch_model_for_realtime
        from acr.utils import rot6D_to_angular
        from mano.manolayer import ManoLayer
        from acr.config import args as acr_args
    finally:
        sys.argv = original_argv

    model = ACR().eval()
    _load_checkpoint(model, args.checkpoint)
    patch_model_for_realtime(
        model,
        input_size=args.input_size,
        device=torch.device("cpu"),
        compute_camera_translation=False,
    )
    neural = AcrNeuralCore(model, threshold=args.threshold).eval()
    if not args.include_mano:
        return neural

    left = ManoLayer(
        ncomps=45,
        center_idx=acr_args().align_idx if acr_args().mano_mesh_root_align else None,
        side="left",
        mano_root=str(args.mano_root),
        use_pca=False,
        flat_hand_mean=False,
    ).eval()
    right = ManoLayer(
        ncomps=45,
        center_idx=acr_args().align_idx if acr_args().mano_mesh_root_align else None,
        side="right",
        mano_root=str(args.mano_root),
        use_pca=False,
        flat_hand_mean=False,
    ).eval()
    left.th_shapedirs[:, 0, :] *= -1
    return AcrManoCore(neural, left, right, rot6D_to_angular).eval()


def export_onnx(model: nn.Module, output: Path, input_size: int, include_mano: bool, opset: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, input_size, input_size, 3, dtype=torch.float32)
    output_names = ["params", "scores", "centers", "valid"]
    if include_mano:
        output_names.extend(("vertices", "joints"))
    common = dict(
        input_names=["image"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, (dummy,), str(output), dynamo=True, **common)
    except Exception as dynamo_error:
        print(f"Dynamo exporter failed ({dynamo_error}); retrying legacy exporter.", file=sys.stderr)
        torch.onnx.export(model, dummy, str(output), dynamo=False, **common)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoints/wild.pkl")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    parser.add_argument("--config", default="configs/demo.yml", help="ACR YAML configuration")
    parser.add_argument("--input-size", type=int, default=256, choices=(256, 288, 320, 352, 384, 416, 448, 480, 512))
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--include-mano", action="store_true", help="Embed locally licensed MANO layers and mesh outputs")
    parser.add_argument("--mano-root", type=Path, default=Path("mano"))
    parser.add_argument("--acknowledge-mano-license", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    validate_export_policy(
        include_mano=args.include_mano,
        acknowledge_mano_license=args.acknowledge_mano_license,
    )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.include_mano:
        for filename in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
            path = args.mano_root / filename
            if not path.is_file():
                raise FileNotFoundError(path)

    checkpoint_hash = sha256_file(args.checkpoint)
    model = _construct_export_model(args)
    export_onnx(model, args.output, args.input_size, args.include_mano, args.opset)
    manifest = build_export_manifest(
        output=args.output,
        input_size=args.input_size,
        include_mano=args.include_mano,
        checkpoint_sha256=checkpoint_hash,
    )
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {args.output} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"Manifest: {manifest_path}")
    print("Private artifact: review checkpoint and MANO licenses before redistribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
