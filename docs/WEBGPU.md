# Browser WebGPU inference and private MANO mesh

The browser path keeps model inference, MANO data, and camera pixels on the user's device. The public Vercel deployment contains no MANO pickle, MANO-derived bundle, ACR checkpoint, or private ONNX artifact.

## Runtime architecture

```text
camera frame
  -> 192x192 MediaPipe palm detector (periodic / tracking loss)
  -> rotated 224x224 MediaPipe landmark model
  -> latest 21-point target state
  -> 60 Hz landmark interpolation
  -> private MANO 778-vertex linear blend skinning
  -> persistent WebGL2 indexed triangle rendering
```

ONNX Runtime Web prefers ordinary WebGPU execution and falls back to WebAssembly. Graph capture remains disabled because the current input path supplies CPU-backed JavaScript tensors and the model may assign shape operations to CPU. The latest-frame scheduler discards stale work instead of accumulating latency.

The render loop is independent of the inference loop. A 60 Hz mesh animation does not imply 60 fresh neural-network results per second.

## Create the private MANO bundle

Use your locally licensed MANO files:

```bash
python tools/convert_mano_browser_bundle.py \
  --mano-root mano \
  --output mano-browser-bundle.json \
  --acknowledge-mano-license
```

The converter writes `mano-browser-bundle-v1` with, for each hand:

- 778 neutral vertices
- MANO triangle faces
- 16-joint skinning weights
- 16 rest skin joints
- 21 rest landmarks
- skin-joint parent indices

The file is marked non-redistributable and is ignored by Git. Do not upload it to Vercel or publish it unless the applicable MANO agreement explicitly permits redistribution.

## Browser flow

1. Open the deployed site or local server.
2. On first use, select `mano-browser-bundle.json`.
3. The browser validates all dimensions, finite values, face bounds, and weight normalization before saving it to IndexedDB.
4. Press **카메라 시작**.
5. The full-screen canvas displays the filled hand surface; the lower-right PIP shows the camera and landmark overlay.

Keyboard controls:

- `Escape`: stop the camera.
- `Shift+Delete`: remove the bundle from IndexedDB and return to local file selection.

The default is one hand for the base-M1 performance target. Add `?hands=2` to the URL to enable two hands.

## M1 performance target

Target environment: base Apple M1 Mac and a current Chrome build.

- Display / mesh animation: 60 Hz target where the display permits.
- Fresh landmark inference: minimum target 30 FPS, aspirational 45-60 FPS.
- Camera preview: submitted through a one-slot latest-frame queue.
- Palm detector interval: every 12 completed frames unless tracking is lost.
- WebGL allocations: shader programs and buffers are created once; vertices and normals update through `bufferSubData`.
- Mesh size: 778 vertices per hand, with indexed triangle rendering and CPU normal recomputation.
- Background: exactly `#000000`.

Record display FPS, fresh inference FPS, p50/p95 latency, Chrome version, macOS version, camera format, and thermal conditions before claiming a sustained M1 result. The PR remains draft until this hardware measurement is recorded.

## Private full ACR export

`tools/export_acr_webgpu.py` remains available for a private fixed-shape ACR ONNX export. Its neural core predicts static left/right ACR parameter slots; optional `--include-mano` embeds locally licensed MANO tensors and must remain private unless redistribution is permitted.

```bash
python tools/export_acr_webgpu.py \
  --checkpoint checkpoints/wild.pkl \
  --output private_models/acr-mano-256.onnx \
  --input-size 256 \
  --include-mano \
  --mano-root mano \
  --acknowledge-mano-license
```

The public real-time page currently uses the lighter landmark-driven MANO deformation path to maximize base-M1 responsiveness. It uses the original MANO topology but is not equivalent to running the complete HRNet-based ACR network on every frame.
