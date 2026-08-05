# Browser WebGPU inference and private rigged hand mesh

The public page performs palm/landmark inference in the browser and drives a locally imported, bone-rigged GLB through Three.js GPU skinning. Camera pixels and the private hand asset remain on the user's device. The Vercel deployment contains no uploaded OBJ, generated GLB, MANO pickle, MANO-derived bundle, ACR checkpoint, or private ONNX artifact.

## Runtime architecture

```text
camera frame
  -> 192x192 MediaPipe palm detector (periodic / tracking loss)
  -> rotated 224x224 MediaPipe landmark model
  -> latest 21-point target state
  -> time-based landmark interpolation
  -> 21-bone hand retargeting
  -> Three.js SkinnedMesh GPU skinning
  -> 60 Hz requestAnimationFrame rendering
```

ONNX Runtime Web prefers ordinary WebGPU execution and falls back to WebAssembly. Graph capture remains disabled because the current input path supplies CPU-backed JavaScript tensors and the model may assign shape operations to CPU. The latest-frame scheduler replaces stale pending work instead of accumulating latency.

The render loop is independent of the inference loop. A 60 Hz mesh animation does not imply 60 fresh neural-network results per second.

## Create a rigged GLB from a neutral OBJ

The converter is intentionally specialized for the supplied open left-hand asset: fingers point toward negative X, the thumb points toward negative Y, and the palm normal is positive Z.

LOD0, intended for the default M1 path:

```bash
python tools/rig_hand_obj_to_glb.py \
  --input 12683_hand_v1_FINAL.obj \
  --output hand_rigged_lod0.glb \
  --voxel-size 1.0
```

LOD1, intended for lower render cost:

```bash
python tools/rig_hand_obj_to_glb.py \
  --input 12683_hand_v1_FINAL.obj \
  --output hand_rigged_lod1.glb \
  --voxel-size 1.5
```

The generated GLB contains:

- an indexed triangle surface
- vertex normals
- `JOINTS_0` and normalized four-component `WEIGHTS_0`
- a glTF skin with inverse bind matrices
- 21 MediaPipe-compatible bone nodes
- stable bone names from `wrist` through `pinky_tip`

Generated GLBs and source OBJ files are ignored by Git and must not be uploaded to Vercel unless the asset license explicitly permits redistribution.

## Browser flow

1. Open the deployed site or local server.
2. On first use, select the generated `.glb` file.
3. The browser verifies the GLB 2.0 header, parses it with `GLTFLoader`, requires a `SkinnedMesh`, validates four-component skin attributes, and checks all 21 bone names.
4. The validated bytes are saved to IndexedDB under the selected LOD key.
5. Press **카메라 시작**.
6. The full-screen canvas displays the filled skinned hand; the lower-right PIP shows the camera and landmark overlay.

Keyboard controls:

- `Escape`: stop the camera.
- `Shift+Delete`: remove the selected LOD asset from IndexedDB and return to local file selection.

Query parameters:

- default: LOD0, one hand, hidden bones
- `?lod=1`: use the separately stored LOD1 GLB
- `?hands=2`: enable two rig instances
- `?debugBones=1`: display `THREE.SkeletonHelper`
- parameters may be combined, for example `?lod=1&hands=2&debugBones=1`

## Bone contract

```text
wrist
thumb_cmc -> thumb_mcp -> thumb_ip -> thumb_tip
index_mcp -> index_pip -> index_dip -> index_tip
middle_mcp -> middle_pip -> middle_dip -> middle_tip
ring_mcp -> ring_pip -> ring_dip -> ring_tip
pinky_mcp -> pinky_pip -> pinky_dip -> pinky_tip
```

The source GLB is a left-hand rig. Right-hand results use a mirrored rig instance while retaining the same stable bone contract.

## Base-M1 performance target

Target environment: base Apple M1 Mac and a current Chrome build.

- Display / bone animation: 60 Hz target where the display permits.
- Fresh landmark inference: measured separately.
- Default mode: one hand and LOD0.
- Lower-cost mode: `?lod=1`.
- Camera preview: submitted through a one-slot latest-frame queue.
- Palm detector interval: every 12 completed frames unless tracking is lost.
- Rendering: one persistent Three.js renderer, scene, material, skeleton, and GPU-skinned geometry.
- Per-frame geometry rebuild or CPU vertex-normal recomputation: none.
- Background: exactly `#000000`.

Every five seconds the console reports:

```text
ACR M1 rig performance {
  displayFps,
  freshInferenceFps,
  captureMs,
  schedulerMs,
  inferenceMs,
  detectorMs,
  landmarkMs,
  renderMs
}
```

Record sustained display FPS, fresh inference FPS, p50/p95 latency, Chrome version, macOS version, camera format, and thermal conditions before claiming a fixed M1 result. The PR remains draft until that hardware measurement is recorded.

## Private MANO and ACR utilities

The previous private MANO bundle converter and fixed-shape ACR ONNX exporter remain available for research workflows, but the production page now uses the generic rigged-GLB path. Neither MANO data nor generated private ACR artifacts are included in the public deployment.
