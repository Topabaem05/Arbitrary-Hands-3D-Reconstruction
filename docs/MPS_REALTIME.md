# Apple Silicon MPS realtime mode

The realtime path is designed to keep the webcam window responsive at 30–60 display FPS while reporting the model's separate inference FPS. It does not claim that every Mac can reconstruct a fresh 3D hand mesh 60 times per second.

## Recommended launch

Use an Apple Silicon Mac with a recent MPS-enabled PyTorch build and place the MANO files and checkpoint as described in the main README.

```bash
./scripts/run_mps_realtime.sh
```

Press `q` or Escape to exit.

## Profiles

| Profile | Input | Live display | Intended use |
|---|---:|---|---|
| `realtime` | 256 | projected keypoints | maximum throughput |
| `balanced` | 384 | projected keypoints | better spatial accuracy |
| `quality` | configured value, normally 512 | full mesh | original-quality path |

Examples:

```bash
# Default 60 Hz display target; inference runs continuously on the newest frame.
ACR_TARGET_FPS=60 ./scripts/run_mps_realtime.sh

# Higher reconstruction quality at lower throughput.
ACR_MPS_PROFILE=balanced ACR_TARGET_FPS=30 ./scripts/run_mps_realtime.sh

# Full mesh rendering. This is normally slower than the keypoint display path.
ACR_MPS_PROFILE=quality ACR_LIVE_VISUALIZATION=mesh ./scripts/run_mps_realtime.sh
```

## What the optimization changes

- Reduces the default MPS webcam input from 512 to 256 pixels. The convolutional trunk and result head are patched for dynamic 256/384/512 spatial sizes without changing learned weights.
- Uses MPS mixed-precision autocast when available. A runtime error disables it and retries in FP32.
- Enables MPS fast math, Metal matmul preference, and unsupported-operator CPU fallback before importing PyTorch.
- Runs webcam inference on a single latest-frame worker. Frames that become stale while inference is active are discarded instead of building latency.
- Keeps OpenCV display and keyboard handling on the foreground thread.
- In keypoint mode, skips CPU `solvePnP`, full vertex transfer, and result-dictionary serialization on every webcam frame.
- Uses `torch.inference_mode()` for the live inference path.

The network still predicts MANO pose, shape, vertices, and joints. The fast live view draws projected joints because per-frame offscreen mesh rendering is a separate CPU/OpenGL bottleneck.

## Runtime controls

| Variable | Default | Meaning |
|---|---:|---|
| `ACR_MPS_PROFILE` | `realtime` | `realtime`, `balanced`, or `quality` |
| `ACR_TARGET_FPS` | `60` in the launcher | Uses a two-to-one display/inference submission stride at 60 |
| `ACR_INPUT_SIZE` | profile value | 256–512, divisible by 32 |
| `ACR_LIVE_VISUALIZATION` | `keypoints` | `keypoints`, `mesh`, or `none` |
| `ACR_INFERENCE_STRIDE` | automatic | Submit every Nth camera frame |
| `ACR_RENDER_EVERY` | `2` for mesh | Render a full mesh every N inferences |
| `ACR_MPS_AMP` | `1` | Enable MPS FP16 autocast |
| `ACR_SHOW_FPS` | `1` | Draw display and inference FPS |
| `PYTORCH_MPS_FAST_MATH` | `1` | Enable faster, less strict MPS math |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Run unsupported MPS operators on CPU |

An explicit existing `--input_size` command-line option overrides the profile size. Environment variables override both.

## Interpreting FPS

`display FPS` measures webcam/UI refresh. `inference FPS` measures complete model, MANO, projection, and the small keypoint transfer. With a 60 Hz target, the UI can stay near the camera refresh rate while inference updates at a lower rate. This avoids presenting repeated screen refreshes as 60 independent reconstructions.

The attainable rate depends on the Apple chip, camera format, PyTorch version, active CPU fallbacks, thermal state, and whether full mesh rendering is enabled. Actual Apple hardware benchmarking remains required before treating 30 or 60 inference FPS as verified.
