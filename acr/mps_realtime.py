"""Runtime helpers for low-latency ACR inference on Apple Silicon.

The model was trained at 512 px, but its convolutional trunk can run at lower
multiples of 32 after the small spatial assumptions in the result head are
patched at runtime. This module keeps those changes isolated from the research
model definition so the original CUDA/512 path remains available.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import queue
import sys
import threading
import time
import types
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from acr.config import args
from acr.model import get_coord_maps
from acr.utils import (
    batch_orth_proj,
    convert_kp2d_from_input_to_orgimg,
    estimate_translation,
)


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass(frozen=True)
class RealtimeOptions:
    profile: str
    input_size: int
    render_size: int
    live_visualization: str
    inference_stride: int
    render_every: int
    mps_amp: bool
    show_fps: bool
    compute_camera_translation: bool


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def derive_spatial_sizes(input_size: int) -> tuple[int, int]:
    """Return HRNet feature and center-map sizes for an ACR input size."""
    if input_size < 256 or input_size > 512 or input_size % 32 != 0:
        raise ValueError("ACR realtime input size must be 256..512 and divisible by 32")
    return input_size // 4, input_size // 8


def build_realtime_options(
    config,
    device: torch.device,
    *,
    environ: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
) -> RealtimeOptions:
    """Resolve a conservative quality profile or an MPS webcam profile."""
    environ = os.environ if environ is None else environ
    argv = sys.argv[1:] if argv is None else list(argv)
    is_live_mps = device.type == "mps" and config.demo_mode == "webcam"
    profile = environ.get("ACR_MPS_PROFILE", "realtime" if is_live_mps else "quality").lower()
    if profile not in {"quality", "balanced", "realtime"}:
        raise ValueError("ACR_MPS_PROFILE must be quality, balanced, or realtime")

    defaults = {
        "quality": (int(config.input_size), int(config.render_size), "mesh", False),
        "balanced": (384, 256, "keypoints", True),
        "realtime": (256, 256, "keypoints", True),
    }
    input_size, render_size, visualization, mps_amp = defaults[profile]

    # An explicit existing CLI option remains authoritative. Environment
    # variables are the highest-priority override for the dedicated launcher.
    if "--input_size" in argv:
        input_size = int(config.input_size)
    input_size = _env_int(environ, "ACR_INPUT_SIZE", input_size)
    render_size = _env_int(environ, "ACR_RENDER_SIZE", render_size)
    derive_spatial_sizes(input_size)

    visualization = environ.get("ACR_LIVE_VISUALIZATION", visualization).lower()
    if visualization not in {"mesh", "keypoints", "none"}:
        raise ValueError("ACR_LIVE_VISUALIZATION must be mesh, keypoints, or none")

    target_fps = _env_int(environ, "ACR_TARGET_FPS", 30)
    default_stride = 2 if target_fps >= 60 and profile != "quality" else 1
    inference_stride = max(1, _env_int(environ, "ACR_INFERENCE_STRIDE", default_stride))
    render_every = max(1, _env_int(environ, "ACR_RENDER_EVERY", 2 if visualization == "mesh" else 1))
    mps_amp = device.type == "mps" and _env_bool(environ, "ACR_MPS_AMP", mps_amp)
    show_fps = _env_bool(environ, "ACR_SHOW_FPS", True)
    compute_camera_translation = (
        config.demo_mode != "webcam"
        or bool(config.save_dict_results)
        or visualization == "mesh"
    )

    return RealtimeOptions(
        profile=profile,
        input_size=input_size,
        render_size=render_size,
        live_visualization=visualization,
        inference_stride=inference_stride,
        render_every=render_every,
        mps_amp=mps_amp,
        show_fps=show_fps,
        compute_camera_translation=compute_camera_translation,
    )


def apply_realtime_options(config, options: RealtimeOptions) -> None:
    config.input_size = options.input_size
    config.render_size = options.render_size


def _dynamic_part_forward(self, x, gt_segm, pred_segm, l_params_maps, r_params_maps):
    """ACR part head with spatial repeats derived from the actual map size."""
    batch_size = len(x)
    part_attention = F.interpolate(
        pred_segm.clone().float(),
        scale_factor=(0.5, 0.5),
        mode="nearest",
    )[:, 1:, :, :]
    logits = pred_segm

    deconv_contact_features = self.contact_layers[1](x)
    deconv_shape_features = self.cam_shape_layers[1](deconv_contact_features)
    weighted_contact_features = self.Hadamard_product(
        deconv_contact_features,
        part_attention,
    ).unsqueeze(-1)
    weighted_shape_features = self.Hadamard_product(
        deconv_shape_features,
        part_attention,
    )

    l_weighted_contact_features = weighted_contact_features[:, :, 16:, :]
    r_weighted_contact_features = weighted_contact_features[:, :, :16, :]
    l_weighted_shape_features = torch.flatten(weighted_shape_features[:, :, 16:], start_dim=1)
    r_weighted_shape_features = torch.flatten(weighted_shape_features[:, :, :16], start_dim=1)

    l_contact_offsets = self.contact_layers[2](l_weighted_contact_features).squeeze(-1).transpose(2, 1).reshape(batch_size, 96)
    r_contact_offsets = self.contact_layers[3](r_weighted_contact_features).squeeze(-1).transpose(2, 1).reshape(batch_size, 96)
    l_shape_offsets = self.cam_shape_layers[2](l_weighted_shape_features)
    r_shape_offsets = self.cam_shape_layers[3](r_weighted_shape_features)

    map_height, map_width = l_params_maps.shape[-2:]
    l_pare_features = torch.cat((l_contact_offsets, l_shape_offsets), dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, map_height, map_width)
    r_pare_features = torch.cat((r_contact_offsets, r_shape_offsets), dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, map_height, map_width)
    l_params_pare = torch.cat((l_params_maps[:, :3].clone(), l_pare_features), dim=1)
    r_params_pare = torch.cat((r_params_maps[:, :3].clone(), r_pare_features), dim=1)

    l_params_maps = self.contact_layers[4](torch.cat((l_params_maps, l_params_pare), dim=1))
    r_params_maps = self.contact_layers[5](torch.cat((r_params_maps, r_params_pare), dim=1))
    return l_params_maps, r_params_maps, logits


def _install_projection(input_size: int, compute_camera_translation: bool) -> None:
    import acr.mano_wrapper as mano_wrapper_module

    pixel_scale = float(input_size) / 2.0
    focal_scale = float(input_size) / 512.0

    def dynamic_vertices_kp3d_projection(outputs, params_dict, meta_data=None, presp=False):
        del presp
        vertices, j3ds = outputs["verts"], outputs["j3d"]
        verts_camed = batch_orth_proj(vertices, params_dict["cam"], mode="3d", keep_dim=True)
        pj3d = batch_orth_proj(j3ds, params_dict["cam"], mode="2d")

        if compute_camera_translation:
            predicts_j3ds = j3ds[:, :24].contiguous().detach().cpu().numpy()
            predicts_pj2ds = (pj3d[:, :24, :2].detach().cpu().numpy() + 1.0) * pixel_scale
            cam_trans = estimate_translation(
                predicts_j3ds,
                predicts_pj2ds,
                focal_length=float(args().focal_length) * focal_scale,
                img_size=np.array([input_size, input_size]),
            ).to(vertices.device)
        else:
            # Keypoint display and live tracking do not consume the expensive
            # CPU solvePnP result. Preserve the output contract without forcing
            # an MPS-to-CPU synchronization for every detected hand.
            cam_trans = vertices.new_zeros((vertices.shape[0], 3))

        projected_outputs = {
            "verts_camed": verts_camed,
            "pj2d": pj3d[:, :, :2],
            "cam_trans": cam_trans,
        }
        if meta_data is not None:
            projected_outputs["pj2d_org"] = convert_kp2d_from_input_to_orgimg(
                projected_outputs["pj2d"],
                meta_data["offsets"],
            )
        return projected_outputs

    mano_wrapper_module.vertices_kp3d_projection = dynamic_vertices_kp3d_projection


def patch_model_for_realtime(
    model: torch.nn.Module,
    *,
    input_size: int,
    device: torch.device,
    compute_camera_translation: bool,
) -> int:
    """Patch fixed 512/64 spatial assumptions without changing weights."""
    feature_size, centermap_size = derive_spatial_sizes(input_size)
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.coordmaps = get_coord_maps(feature_size).to(device)
    base_model.part_forward = types.MethodType(_dynamic_part_forward, base_model)

    parser = base_model._result_parser
    parser.map_size = centermap_size
    parser.centermap_parser.size = centermap_size
    parser.centermap_parser.shrink_scale = float(input_size // centermap_size)
    args().centermap_size = centermap_size
    _install_projection(input_size, compute_camera_translation)
    return centermap_size


def extract_keypoint_overlay(outputs) -> tuple[np.ndarray, np.ndarray]:
    detected = outputs["detection_flag_cache"].bool()
    points = outputs["pj2d_org"][detected].detach().float().cpu().numpy()
    hand_types = outputs["output_hand_type"][detected].detach().cpu().numpy().astype(np.int32)
    return points, hand_types


def draw_keypoint_overlay(
    frame: np.ndarray,
    points: np.ndarray | None,
    hand_types: np.ndarray | None,
    *,
    display_fps: float = 0.0,
    inference_fps: float = 0.0,
    label: str = "",
    show_fps: bool = True,
) -> np.ndarray:
    canvas = frame.copy()
    if points is not None and hand_types is not None:
        for hand_points, hand_type in zip(points, hand_types):
            color = (255, 170, 80) if int(hand_type) == 0 else (80, 170, 255)
            xy = np.rint(hand_points[:, :2]).astype(np.int32)
            for start, end in HAND_EDGES:
                cv2.line(canvas, tuple(xy[start]), tuple(xy[end]), color, 2, cv2.LINE_AA)
            for point in xy:
                cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)

    if show_fps:
        text = f"display {display_fps:4.1f} FPS | inference {inference_fps:4.1f} FPS"
        if label:
            text += f" | {label}"
        cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


@dataclass(frozen=True)
class WorkerResult:
    generation: int
    payload: Any
    duration: float
    error: BaseException | None = None


class LatestFrameWorker:
    """Run one inference at a time and discard stale queued camera frames."""

    def __init__(self, infer: Callable[[np.ndarray], Any]):
        self._infer = infer
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._result_lock = threading.Lock()
        self._latest: WorkerResult | None = None
        self._generation = 0
        self._thread = threading.Thread(
            target=self._run,
            name="acr-mps-inference",
            daemon=True,
        )

    def start(self) -> "LatestFrameWorker":
        self._thread.start()
        return self

    def submit(self, frame: np.ndarray) -> None:
        if self._stop.is_set():
            return
        owned_frame = frame.copy()
        try:
            self._frames.put_nowait(owned_frame)
            return
        except queue.Full:
            pass

        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(owned_frame)
        except queue.Full:
            # The worker won the race and another newer frame is already queued.
            pass

    def latest(self) -> WorkerResult | None:
        with self._result_lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break

            started = time.perf_counter()
            try:
                payload = self._infer(frame)
                result = WorkerResult(
                    generation=self._generation + 1,
                    payload=payload,
                    duration=time.perf_counter() - started,
                )
            except BaseException as error:
                result = WorkerResult(
                    generation=self._generation + 1,
                    payload=None,
                    duration=time.perf_counter() - started,
                    error=error,
                )

            self._generation = result.generation
            with self._result_lock:
                self._latest = result


class ExponentialRate:
    def __init__(self, smoothing: float = 0.15):
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self.smoothing = smoothing
        self.value = 0.0

    def update_duration(self, seconds: float) -> float:
        if seconds <= 0.0:
            return self.value
        instantaneous = 1.0 / seconds
        if self.value == 0.0:
            self.value = instantaneous
        else:
            self.value += self.smoothing * (instantaneous - self.value)
        return self.value
