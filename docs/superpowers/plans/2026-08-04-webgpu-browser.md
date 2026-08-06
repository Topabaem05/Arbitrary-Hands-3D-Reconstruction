# ACR WebGPU Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a browser-only hand-tracking application using ONNX Runtime Web/WebGPU, with a private local ACR conversion path.

**Architecture:** Keep browser inference and rendering in focused ES modules under `webgpu/src`. Use OpenCV's Apache-2.0 MediaPipe ONNX models for the public runtime, and export only the ACR neural core from user-owned weights through `tools/export_acr_webgpu.py`.

**Tech Stack:** JavaScript ES modules, Canvas 2D, ONNX Runtime Web 1.26.0, Python/PyTorch ONNX export, Node test runner, Vercel static hosting.

## Global Constraints

- Do not commit or deploy MANO model files or MANO-derived assets.
- Camera frames remain on device.
- WebGPU is preferred; WASM is the explicit fallback.
- Display FPS and inference FPS are separate metrics.
- Public model URLs are user-overridable and local ONNX loading is supported.
- Fixed detector input is 192×192 and fixed landmark input is 224×224.

---

### Task 1: Geometry and scheduling primitives

**Files:**
- Create: `webgpu/src/geometry.js`
- Create: `webgpu/src/palm.js`
- Create: `webgpu/src/scheduler.js`
- Test: `webgpu/tests/geometry.test.mjs`
- Test: `webgpu/tests/palm.test.mjs`
- Test: `webgpu/tests/scheduler.test.mjs`

- [x] Write tests for 2,016 palm anchors, detector decode, NMS, affine projection, and latest-frame replacement.
- [x] Run `node --test webgpu/tests/*.test.mjs` and verify the tests fail before implementation.
- [x] Implement the minimum geometry, palm post-processing, and scheduler code.
- [x] Re-run the Node tests and require zero failures.

### Task 2: WebGPU runtime and camera pipeline

**Files:**
- Create: `webgpu/src/runtime.js`
- Create: `webgpu/src/camera.js`
- Create: `webgpu/src/renderer.js`
- Create: `webgpu/src/demo.js`
- Create: `webgpu/app.js`
- Create: `webgpu/models/manifest.json`

- [x] Add WebGPU session creation with graph-capture retry and WASM fallback.
- [x] Add remote URL and local-file model loading.
- [x] Add detector/landmark output signature discovery.
- [x] Add rotated ROI tracking and periodic palm re-detection.
- [x] Add latest-frame inference, 2D overlay, 3D skeleton, and separate FPS meters.

### Task 3: Static product surface

**Files:**
- Create: `webgpu/index.html`
- Create: `webgpu/styles.css`
- Create: `webgpu/package.json`
- Create: `webgpu/vercel.json`
- Create: `webgpu/README.md`

- [x] Build camera, image-upload, demo, backend, model-source, and diagnostics controls.
- [x] Add responsive desktop/mobile layout and accessible focus states.
- [x] Configure immutable caching for versioned assets and no caching for HTML/manifest.
- [x] Validate the zero-build static project with a local HTTP server and Chromium demo-mode smoke test.

### Task 4: Private ACR ONNX export path

**Files:**
- Create: `tools/export_acr_webgpu.py`
- Create: `tests/test_acr_webgpu_export.py`
- Create: `docs/WEBGPU.md`

- [x] Write failing tests for fixed-shape static decoding and the MANO redistribution guard.
- [x] Implement ACR raw-head static decoding and ONNX export metadata.
- [x] Require explicit local license acknowledgement for any MANO-derived output.
- [x] Document the distinction between public MediaPipe mode and private ACR conversion.

### Task 5: CI, repository publication, and deployment

**Files:**
- Create: `.github/workflows/webgpu.yml`

- [x] Run Node, Python, syntax, and Chromium smoke checks.
- [x] Commit the implementation to `agent/webgpu-browser` and open a draft PR.
- [x] Deploy `webgpu/` to a Vercel preview when the connected deployment tool accepts the project.
- [x] Inspect deployment status and report the exact unverified hardware boundaries.
