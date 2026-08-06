#!/usr/bin/env python3
"""Create a MediaPipe-compatible skinned GLB from a neutral open-hand OBJ.

The converter intentionally has a narrow contract: an open left hand whose
fingers point along negative X and whose thumb points toward negative Y. It
writes a dependency-free glTF 2.0 binary with 21 named joints and top-four
linear-blend skin weights. The generated GLB is a private user asset and is not
intended to be committed to the source repository.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

BONE_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
BONE_PARENTS = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
]
DEFORM_BONES = np.asarray([0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], dtype=np.int32)

_COMPONENT_FLOAT = 5126
_COMPONENT_UNSIGNED_SHORT = 5123
_COMPONENT_UNSIGNED_INT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


@dataclass
class RigReport:
    output_path: Path
    vertex_count: int
    triangle_count: int
    bone_count: int
    joint_positions: np.ndarray
    joints: np.ndarray
    weights: np.ndarray
    bounds: np.ndarray


def _parse_face_index(token: str, vertex_count: int) -> int:
    raw = int(token.split("/", 1)[0])
    return raw - 1 if raw > 0 else vertex_count + raw


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    continue
                indices = [_parse_face_index(token, len(vertices)) for token in tokens]
                for offset in range(1, len(indices) - 1):
                    triangles.append((indices[0], indices[offset], indices[offset + 1]))
    if len(vertices) < 8 or not triangles:
        raise ValueError(f"OBJ does not contain a usable mesh: {path}")
    result_vertices = np.asarray(vertices, dtype=np.float64)
    result_faces = np.asarray(triangles, dtype=np.int64)
    if result_faces.min() < 0 or result_faces.max() >= len(result_vertices):
        raise ValueError("OBJ face index is outside the vertex array.")
    return result_vertices, result_faces


def simplify_voxel(vertices: np.ndarray, faces: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if not voxel_size > 0:
        raise ValueError("voxel_size must be positive.")
    origin = vertices.min(axis=0)
    cells = np.floor((vertices - origin) / voxel_size + 0.5).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    sums = np.zeros((count, 3), dtype=np.float64)
    np.add.at(sums, inverse, vertices)
    counts = np.bincount(inverse, minlength=count).astype(np.float64)
    clustered = sums / counts[:, None]

    remapped = inverse[faces]
    valid = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    remapped = remapped[valid]
    if len(remapped) == 0:
        raise ValueError("voxel simplification removed every triangle.")
    keys = np.sort(remapped, axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    remapped = remapped[np.sort(first)]

    used = np.unique(remapped)
    compact = np.full(len(clustered), -1, dtype=np.int64)
    compact[used] = np.arange(len(used), dtype=np.int64)
    return clustered[used], compact[remapped]


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    face_normals = np.cross(b - a, c - a)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    fallback = lengths < 1e-12
    normals[~fallback] /= lengths[~fallback, None]
    normals[fallback] = (0.0, 0.0, 1.0)
    return normals


def _median_z_near(vertices: np.ndarray, x: float, y: float, scale: float) -> float:
    radius = max(scale * 0.055, 0.25)
    distance2 = (vertices[:, 0] - x) ** 2 + (vertices[:, 1] - y) ** 2
    selected = vertices[distance2 <= radius * radius]
    if len(selected) < 12:
        nearest = np.argpartition(distance2, min(64, len(distance2) - 1))[: min(64, len(distance2))]
        selected = vertices[nearest]
    lower, upper = np.percentile(selected[:, 2], (10.0, 90.0))
    return float((lower + upper) * 0.5)


def _tip_from_band(vertices: np.ndarray, y_center: float, y_radius: float) -> np.ndarray:
    selected = vertices[np.abs(vertices[:, 1] - y_center) <= y_radius]
    if len(selected) < 20:
        selected = vertices
    threshold = np.percentile(selected[:, 0], 1.0)
    tip = selected[selected[:, 0] <= threshold]
    return np.median(tip, axis=0)


def place_joint_positions(vertices: np.ndarray) -> np.ndarray:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extent = maximum - minimum
    x0, y0, _ = minimum
    sx, sy, _ = extent
    scale = float(max(sx, sy))

    wrist_xy = np.asarray([x0 + sx * 0.89, y0 + sy * 0.59], dtype=np.float64)
    wrist = np.asarray([wrist_xy[0], wrist_xy[1], _median_z_near(vertices, *wrist_xy, scale)])
    positions = np.zeros((21, 3), dtype=np.float64)
    positions[0] = wrist

    finger_specs = [
        ((5, 6, 7, 8), 0.88, 0.53),
        ((9, 10, 11, 12), 0.71, 0.50),
        ((13, 14, 15, 16), 0.52, 0.49),
        ((17, 18, 19, 20), 0.34, 0.53),
    ]
    for indices, yn, base_xn in finger_specs:
        y = y0 + sy * yn
        tip = _tip_from_band(vertices, y, sy * 0.075)
        base_xy = np.asarray([x0 + sx * base_xn, y], dtype=np.float64)
        base = np.asarray([base_xy[0], base_xy[1], _median_z_near(vertices, *base_xy, scale)])
        for index, ratio in zip(indices, (0.0, 0.40, 0.72, 1.0), strict=True):
            point = base * (1.0 - ratio) + tip * ratio
            point[2] = _median_z_near(vertices, float(point[0]), float(point[1]), scale)
            positions[index] = point

    thumb_y = y0 + sy * 0.07
    thumb_tip = _tip_from_band(vertices, thumb_y, sy * 0.13)
    thumb_base_xy = np.asarray([x0 + sx * 0.70, y0 + sy * 0.35], dtype=np.float64)
    thumb_base = np.asarray([
        thumb_base_xy[0], thumb_base_xy[1], _median_z_near(vertices, *thumb_base_xy, scale)
    ])
    for index, ratio in zip((1, 2, 3, 4), (0.0, 0.38, 0.68, 1.0), strict=True):
        point = thumb_base * (1.0 - ratio) + thumb_tip * ratio
        point[2] = _median_z_near(vertices, float(point[0]), float(point[1]), scale)
        positions[index] = point

    if not np.all(np.isfinite(positions)):
        raise ValueError("joint placement produced non-finite coordinates.")
    return positions


def _segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-12:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip(((points - start) @ direction) / denominator, 0.0, 1.0)
    nearest = start + t[:, None] * direction
    return np.linalg.norm(points - nearest, axis=1)


def compute_skin_weights(vertices: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    sigma_finger = max(scale * 0.045, 0.3)
    sigma_root = max(scale * 0.12, 0.8)
    scores = np.zeros((len(vertices), len(DEFORM_BONES)), dtype=np.float64)

    mcp_indices = [1, 5, 9, 13, 17]
    root_distance = np.full(len(vertices), np.inf, dtype=np.float64)
    for mcp in mcp_indices:
        root_distance = np.minimum(root_distance, _segment_distance(vertices, positions[0], positions[mcp]))
    finger_base_x = float(np.mean(positions[mcp_indices, 0]))
    palm_extension = np.maximum(0.0, finger_base_x - vertices[:, 0])
    scores[:, 0] = np.exp(-(root_distance ** 2) / (2.0 * sigma_root ** 2)) * np.exp(
        -(palm_extension ** 2) / (2.0 * (sigma_root * 0.85) ** 2)
    )

    child_for = {
        1: 2, 2: 3, 3: 4,
        5: 6, 6: 7, 7: 8,
        9: 10, 10: 11, 11: 12,
        13: 14, 14: 15, 15: 16,
        17: 18, 18: 19, 19: 20,
    }
    for column, bone in enumerate(DEFORM_BONES[1:], start=1):
        child = child_for[int(bone)]
        distance = _segment_distance(vertices, positions[bone], positions[child])
        score = np.exp(-(distance ** 2) / (2.0 * sigma_finger ** 2))
        wrist_side = np.maximum(0.0, vertices[:, 0] - positions[bone, 0] - scale * 0.10)
        score *= np.exp(-(wrist_side ** 2) / (2.0 * (sigma_root * 0.7) ** 2))
        scores[:, column] = score

    top = np.argpartition(scores, -4, axis=1)[:, -4:]
    top_scores = np.take_along_axis(scores, top, axis=1)
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    total = top_scores.sum(axis=1)
    empty = total < 1e-12
    top_scores[empty] = (1.0, 0.0, 0.0, 0.0)
    top[empty] = 0
    total[empty] = 1.0
    weights = top_scores / total[:, None]
    joint_indices = DEFORM_BONES[top]
    return joint_indices.astype(np.uint16), weights.astype(np.float32)


def _inverse_translation_matrix(position: np.ndarray) -> list[float]:
    x, y, z = map(float, position)
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -x, -y, -z, 1.0,
    ]


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    remainder = len(data) % 4
    return data if remainder == 0 else data + fill * (4 - remainder)


def _write_glb(
    path: Path,
    vertices: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    weights: np.ndarray,
    positions: np.ndarray,
) -> None:
    binary = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []

    def add_view(data: bytes, *, target: int | None = None) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(data)
        view: dict[str, object] = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    def add_accessor(
        array: np.ndarray,
        *,
        component_type: int,
        type_name: str,
        target: int | None = None,
        include_bounds: bool = False,
    ) -> int:
        contiguous = np.ascontiguousarray(array)
        view = add_view(contiguous.tobytes(order="C"), target=target)
        accessor: dict[str, object] = {
            "bufferView": view,
            "byteOffset": 0,
            "componentType": component_type,
            "count": int(len(contiguous)),
            "type": type_name,
        }
        if include_bounds:
            accessor["min"] = np.min(contiguous, axis=0).astype(float).tolist()
            accessor["max"] = np.max(contiguous, axis=0).astype(float).tolist()
        accessors.append(accessor)
        return len(accessors) - 1

    positions_accessor = add_accessor(vertices.astype(np.float32), component_type=_COMPONENT_FLOAT, type_name="VEC3", target=_ARRAY_BUFFER, include_bounds=True)
    normals_accessor = add_accessor(normals.astype(np.float32), component_type=_COMPONENT_FLOAT, type_name="VEC3", target=_ARRAY_BUFFER)
    joints_accessor = add_accessor(joints.astype(np.uint16), component_type=_COMPONENT_UNSIGNED_SHORT, type_name="VEC4", target=_ARRAY_BUFFER)
    weights_accessor = add_accessor(weights.astype(np.float32), component_type=_COMPONENT_FLOAT, type_name="VEC4", target=_ARRAY_BUFFER)
    if len(vertices) <= 65535:
        index_array = faces.astype(np.uint16).reshape(-1)
        index_component = _COMPONENT_UNSIGNED_SHORT
    else:
        index_array = faces.astype(np.uint32).reshape(-1)
        index_component = _COMPONENT_UNSIGNED_INT
    indices_accessor = add_accessor(index_array, component_type=index_component, type_name="SCALAR", target=_ELEMENT_ARRAY_BUFFER)
    inverse_binds = np.asarray([_inverse_translation_matrix(position) for position in positions], dtype=np.float32)
    inverse_accessor = add_accessor(inverse_binds, component_type=_COMPONENT_FLOAT, type_name="MAT4")

    nodes: list[dict] = [
        {"name": "Armature", "children": [1, 2]},
        {"name": "HandMesh", "mesh": 0, "skin": 0},
    ]
    joint_node_indices = list(range(2, 23))
    for index, (name, parent) in enumerate(zip(BONE_NAMES, BONE_PARENTS, strict=True)):
        translation = positions[index] if parent < 0 else positions[index] - positions[parent]
        children = [
            joint_node_indices[child]
            for child, candidate_parent in enumerate(BONE_PARENTS)
            if candidate_parent == index
        ]
        node: dict[str, object] = {
            "name": name,
            "translation": np.asarray(translation, dtype=float).tolist(),
            "extras": {"mediapipeLandmark": index},
        }
        if children:
            node["children"] = children
        nodes.append(node)

    document = {
        "asset": {"version": "2.0", "generator": "ACR rig_hand_obj_to_glb"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{
            "name": "RiggedHand",
            "primitives": [{
                "attributes": {
                    "POSITION": positions_accessor,
                    "NORMAL": normals_accessor,
                    "JOINTS_0": joints_accessor,
                    "WEIGHTS_0": weights_accessor,
                },
                "indices": indices_accessor,
                "material": 0,
            }],
        }],
        "skins": [{
            "name": "MediaPipeHandSkin",
            "inverseBindMatrices": inverse_accessor,
            "skeleton": 2,
            "joints": joint_node_indices,
        }],
        "materials": [{
            "name": "HandSurface",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.72, 0.76, 0.82, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.48,
            },
        }],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "extras": {
            "handedness": "left",
            "boneNames": BONE_NAMES,
            "sourceAxis": {"fingers": "-X", "thumb": "-Y", "normal": "+Z"},
        },
    }
    json_chunk = _pad4(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad4(bytes(binary), b"\x00")
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    output = bytearray(struct.pack("<4sII", b"glTF", 2, total_length))
    output.extend(struct.pack("<I4s", len(json_chunk), b"JSON"))
    output.extend(json_chunk)
    output.extend(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
    output.extend(bin_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(output)
    temporary.replace(path)


def parse_glb(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError("GLB is truncated.")
    magic, version, total = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total != len(raw):
        raise ValueError("GLB header is invalid.")
    json_length, json_type = struct.unpack_from("<I4s", raw, 12)
    if json_type != b"JSON" or 20 + json_length > len(raw):
        raise ValueError("GLB JSON chunk is invalid.")
    return json.loads(raw[20:20 + json_length].decode("utf-8").rstrip(" \x00"))


def build_rigged_hand(obj_path: Path, output_path: Path, voxel_size: float) -> RigReport:
    vertices, faces = load_obj(obj_path)
    vertices, faces = simplify_voxel(vertices, faces, voxel_size)
    normals = compute_vertex_normals(vertices, faces)
    positions = place_joint_positions(vertices)
    joints, weights = compute_skin_weights(vertices, positions)
    _write_glb(output_path, vertices, normals, faces, joints, weights, positions)
    return RigReport(
        output_path=output_path,
        vertex_count=len(vertices),
        triangle_count=len(faces),
        bone_count=len(BONE_NAMES),
        joint_positions=positions,
        joints=joints,
        weights=weights,
        bounds=np.asarray([vertices.min(axis=0), vertices.max(axis=0)]),
    )


def _report_json(report: RigReport, voxel_size: float) -> dict[str, object]:
    return {
        "format": "acr-rigged-hand-report-v1",
        "output": report.output_path.name,
        "voxelSize": voxel_size,
        "vertices": report.vertex_count,
        "triangles": report.triangle_count,
        "bones": BONE_NAMES,
        "bounds": report.bounds.astype(float).tolist(),
        "jointPositions": report.joint_positions.astype(float).tolist(),
        "privateAsset": True,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Neutral open-hand OBJ")
    parser.add_argument("--output", type=Path, required=True, help="Output GLB")
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_rigged_hand(args.input, args.output, args.voxel_size)
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(_report_json(report, args.voxel_size), indent=2) + "\n", encoding="utf-8")
    print(f"GLB: {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")
    print(f"Vertices: {report.vertex_count}; triangles: {report.triangle_count}; bones: {report.bone_count}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
