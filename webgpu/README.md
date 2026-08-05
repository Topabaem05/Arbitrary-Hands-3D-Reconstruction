# ACR Hand Lab — private rigged hand web runtime

The public page runs palm/landmark inference locally with ONNX Runtime Web and renders a filled bone-rigged GLB through Three.js GPU skinning. The source OBJ and generated GLB are **not** included in the repository or Vercel deployment.

## 1. Generate the private hand assets

Default LOD0:

```bash
python tools/rig_hand_obj_to_glb.py \
  --input 12683_hand_v1_FINAL.obj \
  --output hand_rigged_lod0.glb \
  --voxel-size 1.0
```

Lower-cost LOD1:

```bash
python tools/rig_hand_obj_to_glb.py \
  --input 12683_hand_v1_FINAL.obj \
  --output hand_rigged_lod1.glb \
  --voxel-size 1.5
```

Each GLB contains a 21-bone MediaPipe-compatible skeleton and maximum four normalized skin influences per vertex. Generated files are ignored by Git.

## 2. Run locally

```bash
python3 -m http.server 4173 --directory webgpu
```

Open `http://localhost:4173/`. On the first visit:

1. Select `hand_rigged_lod0.glb`.
2. The browser validates the rig and stores the GLB in IndexedDB.
3. Press **카메라 시작**.

Controls and query parameters:

- `Escape`: stop the camera.
- `Shift+Delete`: clear the locally stored asset for the active LOD.
- `?lod=1`: select the separate LOD1 storage slot.
- `?hands=2`: enable two hands.
- `?debugBones=1`: show the bone helper.

## Performance contract

- Target: base Apple M1 Mac, current Chrome.
- Bone/display loop: 60 Hz target through `requestAnimationFrame`.
- Fresh inference: measured separately; interpolation never counts as a new inference.
- GPU skinning: persistent `SkinnedMesh`; no per-frame geometry rebuilding.
- Background: pure black.
- Camera: small lower-right picture-in-picture.

See `../docs/WEBGPU.md` for the full architecture, asset contract, and measurement protocol.
