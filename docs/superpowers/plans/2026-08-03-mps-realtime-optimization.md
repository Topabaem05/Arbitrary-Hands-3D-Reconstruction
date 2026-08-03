# MPS Realtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a low-latency Apple MPS webcam path with separate display and inference FPS.

**Architecture:** Isolate profile resolution, dynamic-shape patches, lightweight visualization, and the latest-frame worker in `acr/mps_realtime.py`. Keep orchestration and existing image/video behavior in `acr/main.py`.

**Tech Stack:** Python, PyTorch MPS, OpenCV, unittest.

## Global Constraints

- Preserve CUDA and 512-pixel quality behavior.
- Never represent display FPS as independent reconstruction FPS.
- Keep actual Mac performance unverified until run on Apple hardware.

---

### Task 1: Define and test realtime runtime primitives

- [x] Add profile resolution and environment overrides.
- [x] Add dynamic spatial-size derivation and dynamic parameter-map expansion.
- [x] Add keypoint overlay and exponential FPS meters.
- [x] Add a one-slot latest-frame inference worker.
- [x] Verify the new tests fail when the runtime module is absent.

### Task 2: Integrate the optimized webcam path

- [x] Enable MPS fast-math, Metal preference, and fallback before importing PyTorch.
- [x] Apply 256/384/512 runtime patches after loading the original weights.
- [x] Add MPS autocast with FP32 retry.
- [x] Remove live result serialization and optional solvePnP from keypoint mode.
- [x] Move keypoint/none webcam inference to the worker while retaining foreground mesh rendering.

### Task 3: Validate and publish

- [x] Run 16 focused unit tests and Python compilation checks locally.
- [x] Add launcher and runtime documentation.
- [ ] Run the model with MANO/checkpoint on Apple Silicon and record device-specific inference/display FPS.
