# MPS realtime optimization design

## Goal

Optimize the ACR webcam path toward a smooth 30–60 FPS user experience on Apple Silicon while preserving the original CUDA and 512-pixel quality paths.

## Design

The MPS webcam default becomes a 256-pixel keypoint-display profile. A runtime patch removes fixed 128/64 spatial assumptions from coordinate maps, parameter-map expansion, center-map parsing, and projection scaling without changing model weights. A 384-pixel balanced profile and the original 512-pixel mesh profile remain selectable.

Inference runs in one background latest-frame worker, while OpenCV display remains on the foreground thread. The worker has a one-frame queue and replaces stale pending frames, preventing latency growth. The overlay reports display FPS and inference FPS separately.

MPS performance controls are enabled before PyTorch import. The model uses MPS autocast where supported and automatically retries in FP32 on failure. Keypoint mode avoids per-frame CPU solvePnP and skips conversion of all vertices and result fields to NumPy; mesh mode retains those operations because rendering consumes camera translation and vertices.

## Safety and quality boundaries

- Do not claim a measured 30–60 inference FPS without Apple hardware evidence.
- Preserve CUDA DataParallel and CUDA FP16 behavior.
- Preserve full 512-pixel mesh rendering through the quality profile.
- Make reduced input resolution explicit because it can lower fine spatial accuracy.
- Keep unsupported MPS operation fallback enabled, but expose inference FPS so CPU fallback regressions are visible.
