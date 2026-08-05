# Rigged Hand GLB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the uploaded neutral hand OBJ into M1-ready 21-bone GLB assets and add a private local-GLB browser runtime that drives those bones from MediaPipe landmarks.

**Architecture:** A Python asset tool performs mesh normalization, voxel-cluster LOD generation, skeleton placement, four-influence skin weighting, and dependency-free GLB 2.0 serialization. The browser stores the selected GLB in IndexedDB, loads it with Three.js, validates the bone contract, retargets MediaPipe landmarks to the skeleton, and renders the filled hand mesh in a 60 Hz `requestAnimationFrame` loop.

**Tech Stack:** Python 3.11, NumPy, trimesh for local asset preparation, glTF 2.0/GLB, JavaScript ES modules, Three.js, IndexedDB, ONNX Runtime Web, Node test runner, Vercel static hosting.

## Global Constraints

- Target device: base Apple M1 Mac in current Chrome.
- Main background: exactly `#000000`.
- Main viewport: filled hand surface, not joints or pipes.
- Camera: small PIP fixed to lower-right.
- Bone names: exact 21-name MediaPipe contract from the approved design.
- Vertex influences: maximum four, normalized.
- LOD0: 14,000–18,000 triangles; LOD1: 5,000–8,000 triangles.
- Uploaded OBJ and generated GLB files must not be committed or deployed.
- Display/render target is 60 Hz; fresh inference FPS remains separately reported.

---

### Task 1: Dependency-free rigged GLB asset generator

**Files:**
- Create: `tools/rig_hand_obj_to_glb.py`
- Create: `tests/test_rig_hand_obj_to_glb.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: OBJ path, output path, voxel size, handedness.
- Produces: `build_rigged_hand(obj_path: Path, output_path: Path, voxel_size: float) -> RigReport` and a GLB containing `POSITION`, `NORMAL`, `JOINTS_0`, `WEIGHTS_0`, indexed triangles, 21 joint nodes, one skin, and inverse-bind matrices.

- [ ] Write tests for the exact 21 bone names and hierarchy, finite joint locations, maximum four normalized weights, valid GLB header/chunks, bounded indices, and triangle budgets.
- [ ] Run `python -m unittest tests/test_rig_hand_obj_to_glb.py -v` and confirm the module is missing.
- [ ] Implement OBJ loading, voxel-cluster simplification, centerline-based joint placement, segment-distance weights, normals, inverse bind matrices, and GLB serialization.
- [ ] Generate LOD0 at voxel size near `1.0` and LOD1 near `1.5`; assert target triangle ranges.
- [ ] Add `*.rigged-hand*.glb`, uploaded OBJ extraction folders, and asset reports to `.gitignore`.
- [ ] Run targeted tests and Python compilation.
- [ ] Commit the generator and tests.

### Task 2: Local GLB validation and IndexedDB storage

**Files:**
- Create: `webgpu/src/rigged-asset-store.js`
- Create: `webgpu/tests/rigged-asset-store.test.mjs`

**Interfaces:**
- Produces: `RiggedAssetStore`, `createMemoryRiggedAssetStore()`, `readGlbFile(file)`, `validateGlbHeader(buffer)`.
- Store value: `{ name: string, bytes: ArrayBuffer, importedAt: number }`.

- [ ] Write failing tests for GLB magic/version/length validation and save/load/clear behavior.
- [ ] Run the targeted Node test and confirm missing-module failure.
- [ ] Implement validation, ArrayBuffer cloning, IndexedDB persistence, and injectable memory storage.
- [ ] Run targeted and complete JavaScript tests.
- [ ] Commit storage support.

### Task 3: Three.js rig loader and bone contract

**Files:**
- Create: `webgpu/src/rigged-hand-renderer.js`
- Create: `webgpu/tests/rigged-hand-renderer.test.mjs`
- Modify: `webgpu/index.html`

**Interfaces:**
- Produces: `REQUIRED_HAND_BONES`, `validateRiggedHand(root)`, and `RiggedHandRenderer` with `load(arrayBuffer)`, `setHands(hands, now)`, `render(now)`, `reset()`, and `dispose()`.
- Uses import map aliases `three` and `three/addons/`.

- [ ] Write source-contract tests requiring Three.js imports, `GLTFLoader.parse`, a `SkinnedMesh`, exact required bone names, black renderer clear color, persistent scene objects, and optional `THREE.SkeletonHelper` only under `debugBones`.
- [ ] Run tests and confirm the renderer is absent.
- [ ] Implement renderer, camera, lighting, material override, left-rig cloning, right-hand Y reflection, resize handling, drag rotation, and cleanup.
- [ ] Add import map entries pinned to one Three.js version.
- [ ] Run JavaScript tests and syntax checks.
- [ ] Commit the renderer.

### Task 4: MediaPipe-to-bone retargeting

**Files:**
- Create: `webgpu/src/hand-retarget.js`
- Create: `webgpu/tests/hand-retarget.test.mjs`

**Interfaces:**
- Produces: `MEDIAPIPE_BONE_INDEX`, `HandLandmarkSmoother`, `applyHandPose(rig, hand, now)`.
- Consumes: 21 MediaPipe `worldPoints` or screen-space points and a validated rig.

- [ ] Write failing tests for index mapping, palm-frame construction, rest-direction identity, 90-degree finger rotation, handedness reflection, and elapsed-time interpolation.
- [ ] Run targeted tests and confirm missing-module failure.
- [ ] Implement root translation/scale/orientation and parent-local bone quaternion solving with stable degenerate-vector fallbacks.
- [ ] Run targeted and complete JavaScript tests.
- [ ] Commit retargeting.

### Task 5: Replace MANO bundle flow with rigged GLB flow

**Files:**
- Modify: `webgpu/app.js`
- Modify: `webgpu/index.html`
- Modify: `webgpu/styles.css`
- Modify: `webgpu/tests/app-mano-flow.test.mjs`
- Modify: `webgpu/tests/layout.test.mjs`
- Create: `webgpu/tests/app-rigged-flow.test.mjs`

**Interfaces:**
- First visit imports `.glb`; later visits load IndexedDB automatically.
- `requestAnimationFrame` owns bone update and render cadence.
- Camera callbacks submit only latest inference work.

- [ ] Write failing tests requiring `.glb` input, `RiggedAssetStore`, `RiggedHandRenderer`, no `ManoBundleStore`, no CPU mesh deformation path, exact black background, lower-right PIP, `?lod=1`, `?hands=2`, `?debugBones=1`, Escape stop, and Shift+Delete asset reset.
- [ ] Run tests and confirm current MANO flow fails them.
- [ ] Integrate GLB import/storage, renderer load, retarget update, camera flow, and existing performance reporting.
- [ ] Keep concise recovery states for invalid GLB, missing bones, WebGL2, camera, and inference errors.
- [ ] Run all JavaScript tests and syntax checks.
- [ ] Commit app integration.

### Task 6: CI, documentation, asset generation, and deployment

**Files:**
- Modify: `.github/workflows/webgpu.yml`
- Modify: `docs/WEBGPU.md`
- Modify: `webgpu/README.md`

**Interfaces:**
- Documents asset-generation commands, private asset boundary, browser import, debug query parameters, and M1 measurement protocol.

- [ ] Add Python rig-generator tests and compilation to CI.
- [ ] Run all Python and JavaScript tests, syntax checks, JSON checks, and static smoke checks.
- [ ] Generate `hand-rigged-lod0.glb` and `hand-rigged-lod1.glb` from the uploaded OBJ and validate reports.
- [ ] Confirm generated GLBs and source OBJ are untracked.
- [ ] Push the branch and verify GitHub Actions success.
- [ ] Deploy Vercel Production without private GLB files and verify root, manifest, favicon, and pinned module responses.
- [ ] Keep the PR draft until a base M1 Chrome run records sustained display and inference measurements.