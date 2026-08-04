# Browser WebGPU hand tracking

The repository now has two deliberately separate browser paths.

## 1. Public Vercel application

`webgpu/` is a zero-build static application that runs OpenCV's Apache-2.0 MediaPipe palm and hand-landmark ONNX models through ONNX Runtime Web.

- Preferred execution provider: WebGPU
- Fallback execution provider: WebAssembly
- Palm detector input: `1 × 192 × 192 × 3` NHWC float32
- Landmark input: `1 × 224 × 224 × 3` NHWC float32
- Tracking: periodic palm detection plus landmark-based rotated ROI updates
- Output: 21 screen-space points, 21 world-space points when supplied by the model, handedness, confidence, 2D overlay, and an interactive 3D skeleton

The browser never uploads camera pixels. Model files are fetched only when camera or image inference begins. The model URL fields can be replaced with local `.onnx` files.

### Local launch

```bash
python3 -m http.server 4173 --directory webgpu
```

Open:

```text
http://localhost:4173/
```

The dependency-free product demo is available at:

```text
http://localhost:4173/?demo=1
```

### Vercel

Set the Vercel project's **Root Directory** to `webgpu`. There is no build command and no output directory. `webgpu/vercel.json` adds camera permissions, cross-origin isolation headers, and cache policy.

The app reports display FPS and completed inference FPS independently. A 60 Hz preview does not imply 60 new neural-network results per second.

## 2. Private ACR conversion

The original ACR network is not directly browser-compatible because its runtime parser uses dynamic Python control flow, tensor-length-dependent branches, CUDA-specific calls, CPU `solvePnP`, and MANO post-processing. `tools/export_acr_webgpu.py` replaces that parser with a fixed contract:

```text
input image       [1, H, H, 3] float32 RGB, range 0..1
params            [1, 2, 109]  left/right ACR parameters
scores            [1, 2]
centers           [1, 2, 2]
valid             [1, 2]
```

The output always contains one left slot and one right slot. Invalid slots are zeroed. This eliminates dynamic output lengths and is suitable for ONNX Runtime Web's static WebGPU path.

### Neural core only

Install a recent PyTorch ONNX toolchain in the existing ACR environment:

```bash
pip install onnx onnxscript
```

Export at 256 pixels:

```bash
python tools/export_acr_webgpu.py \
  --checkpoint checkpoints/wild.pkl \
  --output private_models/acr-core-256.onnx \
  --input-size 256
```

The command writes:

```text
private_models/acr-core-256.onnx
private_models/acr-core-256.onnx.json
```

### Local full ACR + MANO model

A full export embeds MANO tensors and adds static vertex/joint outputs:

```bash
python tools/export_acr_webgpu.py \
  --checkpoint checkpoints/wild.pkl \
  --output private_models/acr-mano-256.onnx \
  --input-size 256 \
  --include-mano \
  --mano-root mano \
  --acknowledge-mano-license
```

This mode is intentionally fail-closed without the acknowledgement flag. The generated ONNX contains MANO-derived data and must remain private unless the user's MANO agreement explicitly permits redistribution. The repository, GitHub branch, and Vercel deployment do not contain MANO files or generated ACR/MANO artifacts.

## Model source and licenses

The public application loads:

- `opencv/palm_detection_mediapipe`
- `opencv/handpose_estimation_mediapipe`

Both model repositories identify their files as Apache-2.0. ONNX Runtime is MIT-licensed. ACR source code is Apache-2.0, but checkpoint and MANO rights must be checked separately before publishing converted artifacts.

## Browser requirements

Use a current Chrome or Edge build for WebGPU. On macOS, browser WebGPU maps to Metal through the browser implementation. Browsers without WebGPU use ONNX Runtime Web's WASM execution provider. Camera access requires a secure context (`https://` or localhost).

## Performance controls

- One hand is the default for minimum latency.
- Two-hand mode runs the landmark model once per ROI.
- Palm detection runs every eighth completed frame unless tracking is lost.
- Static shapes enable the runtime to attempt WebGPU graph capture.
- The latest-frame scheduler drops stale pending work rather than accumulating delay.
- Model compilation and first inference are excluded from steady-state FPS interpretation.

Actual throughput depends on browser version, GPU, thermal state, model operator placement, and camera resolution. Measure the application's **Inference** metric on the target machine before claiming a fixed 30 or 60 inference FPS.
