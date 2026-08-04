# ACR WebGPU Browser Design

## Goal

Provide a Vercel-hostable browser application that performs private, on-device real-time hand tracking with ONNX Runtime Web's WebGPU execution provider, while preserving a local-only conversion path for the original ACR checkpoint.

## Architecture

The public web application uses Apache-2.0 OpenCV MediaPipe palm and hand-landmark ONNX models. A fixed-shape detector runs periodically and a 224×224 landmark model tracks the most recent hand ROI between detections. Inference runs through WebGPU when available and falls back to WebAssembly; camera frames never leave the browser.

The original ACR/MANO path is separate. `tools/export_acr_webgpu.py` converts the user's own ACR checkpoint into a fixed 256×256 ONNX neural-core model with static left/right outputs. The repository does not redistribute MANO files, a MANO-derived browser bundle, or a converted ACR+MANO model. Users may load locally generated ACR artifacts through the browser's local-model controls.

## Browser data flow

1. The camera supplies the latest video frame.
2. The palm detector receives an aspect-preserving 192×192 RGB tensor.
3. SSD anchors, sigmoid scoring, and NMS select at most two palms.
4. Rotated square ROIs are cropped into 224×224 landmark inputs.
5. Landmark outputs are projected back to camera coordinates and update the next-frame ROI.
6. A foreground canvas draws the camera overlay; a second canvas renders a perspective 3D skeleton.
7. Display FPS and completed inference FPS are reported separately.

## Performance policy

- Use static input dimensions and request WebGPU graph capture; retry without graph capture if session creation fails.
- Run at most one inference job at a time and replace stale queued frames.
- Run palm detection every eight completed frames or after tracking confidence is lost.
- Prefer requestVideoFrameCallback when supported.
- Default to one hand on constrained devices; permit two hands through a UI control.
- Never label display refresh rate as model inference throughput.

## Deployment

`webgpu/` is a zero-build static project. Vercel serves HTML, CSS, JavaScript, and model manifest files. ONNX model files are fetched directly from the OpenCV Hugging Face repositories, with user-overridable URLs and local file loading. Camera access requires HTTPS, which Vercel provides.

## Failure handling

- If WebGPU is unavailable, show the WASM fallback state rather than failing silently.
- If remote model loading fails, expose local `.onnx` file inputs.
- If the camera is denied, retain image-upload/demo mode.
- If a model output signature differs, stop with the discovered input/output names in the diagnostic panel.
- If ACR conversion lacks checkpoint or MANO license acknowledgement, fail closed with an actionable message.

## Validation

- Node tests cover anchor generation, SSD decode, IoU/NMS, ROI transforms, scheduler replacement, and FPS measurement.
- Python tests cover ACR static output decoding and exporter license gates without requiring proprietary assets.
- Chromium smoke tests cover responsive layout, demo animation, camera-control state, backend diagnostics, and no console errors in demo mode.
- The deployed build remains a preview until real WebGPU inference is measured on a supported browser.
