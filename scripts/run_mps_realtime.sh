#!/usr/bin/env bash
set -euo pipefail

# Fast defaults. Every value can be overridden by exporting it before launch.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTORCH_MPS_FAST_MATH="${PYTORCH_MPS_FAST_MATH:-1}"
export PYTORCH_MPS_PREFER_METAL="${PYTORCH_MPS_PREFER_METAL:-1}"
export ACR_MPS_PROFILE="${ACR_MPS_PROFILE:-realtime}"
export ACR_TARGET_FPS="${ACR_TARGET_FPS:-60}"
export ACR_LIVE_VISUALIZATION="${ACR_LIVE_VISUALIZATION:-keypoints}"
export ACR_SHOW_FPS="${ACR_SHOW_FPS:-1}"

CAMERA_ID="${ACR_CAMERA_ID:-0}"

exec python -m acr.main \
  --demo_mode webcam \
  --cam_id "${CAMERA_ID}" \
  "$@"
