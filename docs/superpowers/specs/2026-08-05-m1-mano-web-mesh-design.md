# M1 MANO Web Mesh Design

## Objective

Replace the browser's joint-and-pipe visualization with a filled, shaded hand surface derived from the same MANO topology used by the original ACR repository, while keeping a black background and the small lower-right camera picture-in-picture. The performance target is a 60 Hz display and mesh-animation loop on a base Apple M1 Mac in current Chrome, with fresh landmark inference measured separately and never misreported as 60 neural-network inferences per second.

## Licensing boundary

MANO data is not committed, embedded in the public deployment, or uploaded to Vercel. A local Python converter reads the user's licensed `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` and writes a private browser bundle. The browser imports that bundle locally, validates it, and stores it in IndexedDB. Camera frames and MANO data remain on the device.

## Architecture

1. The existing MediaPipe palm and landmark ONNX models continue to run with ONNX Runtime Web, preferring ordinary WebGPU and falling back to WASM.
2. A local MANO bundle contains neutral vertices, triangle faces, skinning weights, and neutral 21-joint positions for both hands.
3. Browser-side deformation maps the 16 MANO skinning joints to MediaPipe's 21 landmarks, constructs per-joint rigid transforms from rest and detected bone frames, and applies linear blend skinning to all 778 vertices.
4. A dedicated WebGL2 renderer uploads the deformed vertex buffer and draws the indexed triangle mesh with computed normals, simple directional lighting, depth testing, and a pure black clear color.
5. New inference results update a target landmark state. The render loop runs independently through `requestAnimationFrame`, smooths toward the target at 60 Hz, recomputes the small 778-vertex mesh, and renders it. Stale camera frames are still discarded rather than queued.

## Browser flow

- First visit without a saved bundle: the full-screen prompt requests a private `mano-browser-bundle.json` file.
- After validation: the bundle is saved in IndexedDB and the prompt changes to camera start.
- Later visits: the bundle is loaded automatically from IndexedDB.
- `Escape` stops the camera. A hidden keyboard action (`Shift+Delete`) clears the locally stored bundle and returns to bundle selection.
- The only persistent visible surfaces are the full-screen hand mesh and the lower-right camera PIP.

## Performance contract

- Target device: base Apple M1 Mac, current Chrome.
- Display and mesh animation target: 60 Hz where the display permits.
- Fresh landmark inference target: at least 30 FPS, aspirational 45–60 FPS.
- The application must not label repeated/interpolated frames as fresh inference.
- Palm detection runs periodically or after tracking loss; landmark inference operates on the latest frame only.
- WebGL buffers are allocated once and updated with `bufferSubData`; no per-frame shader/program/buffer allocation is permitted.
- Mesh deformation uses typed arrays and preselected non-zero skinning influences.
- Background is exactly `#000000`.

## Failure behavior

- Invalid or wrong-version MANO bundle: reject before IndexedDB storage and show a concise retry prompt.
- IndexedDB unavailable: keep the validated bundle in memory for the current session.
- WebGL2 unavailable: show an explicit unsupported-browser prompt; do not silently fall back to the old skeleton.
- WebGPU unavailable: ONNX Runtime may use WASM, while mesh rendering remains WebGL2.
- Camera or model failure: stop the camera, preserve the private bundle, and allow retry.

## Verification

- Python tests validate converter policy, bundle schema, array dimensions, face bounds, weight normalization, and private-artifact metadata.
- JavaScript tests validate schema rejection, MANO-to-MediaPipe mapping, rigid-frame construction, skinning identity, deformation continuity, IndexedDB adapter behavior through an injectable store, and 60 Hz interpolation math.
- Static layout tests enforce black full-screen mesh canvas plus lower-right camera PIP and reject the previous skeleton renderer.
- GitHub Actions runs all Python/JavaScript tests and syntax checks.
- Vercel production is checked for HTTP 200 on the page and required static assets. Actual camera permission, WebGPU/WASM inference, WebGL2 rendering, and M1 performance remain target-device measurements.