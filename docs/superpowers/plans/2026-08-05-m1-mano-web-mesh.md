# M1 MANO Web Mesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a filled MANO-topology hand mesh at a 60 Hz display loop on a base M1 Mac while keeping MANO assets private and retaining the minimal black UI with a lower-right camera PIP.

**Architecture:** Convert locally licensed MANO pickle files into a private JSON bundle containing neutral mesh, faces, weights, and rest joints. Load and persist that bundle in the browser, drive a typed-array linear-blend-skinning implementation from MediaPipe landmarks, and render the dynamically deformed 778-vertex mesh with a persistent WebGL2 pipeline independent of the ONNX inference cadence.

**Tech Stack:** Python 3.11, PyTorch/MANO loader, JavaScript ES modules, IndexedDB, ONNX Runtime Web, WebGL2, Node test runner, Vercel static hosting.

## Global Constraints

- Target device is a base Apple M1 Mac in current Chrome.
- Display and mesh-animation target is 60 Hz; fresh inference FPS remains separately measured.
- Background must be exactly `#000000`.
- MANO pickle files, converted bundle contents, and generated MANO ONNX artifacts must never be committed or uploaded to Vercel.
- Public runtime must not silently fall back to the old joint-and-pipe renderer.
- Existing ordinary WebGPU first, WASM fallback inference behavior remains intact.

---

### Task 1: Private MANO browser-bundle converter

**Files:**
- Create: `tools/convert_mano_browser_bundle.py`
- Create: `tests/test_mano_browser_bundle.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: local `MANO_LEFT.pkl`, `MANO_RIGHT.pkl`, repository `mano.manolayer.ManoLayer`.
- Produces: `convert_mano_bundle(mano_root: Path, output: Path, acknowledge_license: bool) -> dict` and schema `mano-browser-bundle-v1`.

- [ ] Write tests that reject conversion without `--acknowledge-mano-license`, validate 778 vertices, bounded triangle indices, 16 normalized weights per vertex, and 21 rest joints per hand.
- [ ] Run `python -m unittest tests/test_mano_browser_bundle.py -v` and verify failure because the converter is absent.
- [ ] Implement the converter with float rounding, private metadata, and atomic output replacement.
- [ ] Add generated bundle names to `.gitignore`.
- [ ] Run the converter tests and existing exporter tests.
- [ ] Commit the converter and tests.

### Task 2: Browser bundle validation and persistence

**Files:**
- Create: `webgpu/src/mano-bundle.js`
- Create: `webgpu/tests/mano-bundle.test.mjs`

**Interfaces:**
- Produces: `validateManoBundle(value)`, `loadManoBundleFile(file)`, `ManoBundleStore`, and `createMemoryBundleStore()`.
- Bundle returns typed arrays for left/right vertices, faces, weights, rest joints, and joint parents.

- [ ] Write failing tests for version, dimensions, finite values, face bounds, normalized weights, and injectable memory-store save/load/clear behavior.
- [ ] Run the targeted Node test and verify missing-module failure.
- [ ] Implement validation, typed-array conversion, IndexedDB storage, and memory fallback.
- [ ] Run targeted and complete JavaScript tests.
- [ ] Commit bundle loading and persistence.

### Task 3: Landmark-driven MANO deformation

**Files:**
- Create: `webgpu/src/mano-deformer.js`
- Create: `webgpu/tests/mano-deformer.test.mjs`

**Interfaces:**
- Consumes: validated hand data and MediaPipe `worldPoints`/`points`.
- Produces: `ManoHandDeformer`, `MANO_TO_MEDIAPIPE`, and `LandmarkInterpolator`.

- [ ] Write failing tests for the 16-joint mapping, rest-pose identity deformation, rigid translation, normalized non-zero influence packing, and interpolation convergence independent of inference cadence.
- [ ] Run the targeted test and verify missing-module failure.
- [ ] Implement stable palm/finger frames, per-joint rigid transforms, four-influence packed skinning, typed-array output, and time-based smoothing.
- [ ] Run targeted and complete JavaScript tests.
- [ ] Commit deformation math.

### Task 4: Persistent WebGL2 mesh renderer

**Files:**
- Create: `webgpu/src/mesh-renderer.js`
- Create: `webgpu/tests/mesh-renderer.test.mjs`
- Modify: `webgpu/src/renderer.js`

**Interfaces:**
- Produces: `HandMeshRenderer(canvas)` with `setBundle(bundle)`, `updateHands(hands, now)`, `render(now)`, and `dispose()`.
- Renderer accepts up to two deformed hand vertex arrays and indexed faces.

- [ ] Write source-contract tests that require WebGL2, one-time buffers/program creation, `bufferSubData`, indexed triangle drawing, depth testing, and black clear color.
- [ ] Run the tests and verify failure because the renderer does not exist.
- [ ] Implement shaders, normals, camera controls, resize handling, persistent buffers, and two-hand draw calls.
- [ ] Remove/export no old `Skeleton3DRenderer` path from the production app.
- [ ] Run all JavaScript tests and syntax checks.
- [ ] Commit the renderer.

### Task 5: Minimal first-run bundle import and 60 Hz app loop

**Files:**
- Modify: `webgpu/index.html`
- Modify: `webgpu/styles.css`
- Modify: `webgpu/app.js`
- Modify: `webgpu/tests/layout.test.mjs`
- Create: `webgpu/tests/app-mano-flow.test.mjs`

**Interfaces:**
- First-run prompt selects `.json`; saved bundle enables camera start.
- `requestAnimationFrame` owns mesh render cadence; camera video callbacks only submit latest inference work.

- [ ] Write failing layout/flow tests for black background, hidden file input, bundle-first prompt, no skeleton renderer, lower-right PIP, rAF render loop, and `Shift+Delete` local-bundle reset.
- [ ] Run tests and verify expected failures.
- [ ] Integrate `ManoBundleStore`, `HandMeshRenderer`, bundle selection, camera flow, and independent render loop.
- [ ] Preserve concise errors for invalid bundle, WebGL2 failure, camera failure, and inference failure.
- [ ] Run all JavaScript tests and syntax checks.
- [ ] Commit the app integration.

### Task 6: Performance and deployment verification

**Files:**
- Modify: `webgpu/README.md`
- Modify: `docs/WEBGPU.md`
- Modify: `.github/workflows/webgpu.yml`

**Interfaces:**
- Documents converter command, local-only licensing boundary, M1 measurement protocol, and display-vs-inference FPS distinction.

- [ ] Add CI execution of converter tests and all new JavaScript tests.
- [ ] Run Python tests, Node tests, Python compilation, JavaScript syntax checks, and JSON validation.
- [ ] Confirm no MANO bundle or pickle is tracked.
- [ ] Push the branch, verify GitHub Actions success, deploy Vercel Production, and verify page/static asset HTTP responses.
- [ ] Keep the PR draft until a base M1 Chrome run records display FPS, inference FPS, latency, and thermal conditions.
