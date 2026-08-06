# Public Rigged Hand Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically load the approved LOD0/LOD1 rigged hand GLBs from Vercel on first visit.

**Architecture:** Preserve IndexedDB as an optional local override, then fetch and validate a same-origin bundled GLB when no override exists. Publish only the two approved assets and expose their integrity metadata in the manifest.

**Tech Stack:** JavaScript ES modules, IndexedDB, Three.js GLTFLoader, static GLB 2.0, Vercel.

## Global Constraints

- Do not run CI for this change.
- Use a commit message containing `[skip ci]`.
- Publish only `hand_rigged_v3_lod0.glb` and `hand_rigged_v3_lod1.glb`.
- Keep the source OBJ, MANO files, conversion reports, and other generated assets private.
- Preserve local-file fallback and camera permission gating.

---

### Task 1: Bundled asset fallback

**Files:**
- Modify: `webgpu/src/rigged-asset-store.js`
- Modify: `webgpu/tests/rigged-asset-store.test.mjs`

- [x] Add deterministic LOD URL resolution from the existing storage key.
- [x] Fetch, validate, and label bundled GLBs when IndexedDB has no override.
- [x] Verify local overrides take precedence and HTTP/invalid-GLB failures reject.

### Task 2: Publish assets and cache contract

**Files:**
- Create: `webgpu/assets/hand_rigged_v3_lod0.glb`
- Create: `webgpu/assets/hand_rigged_v3_lod1.glb`
- Modify: `webgpu/models/manifest.json`
- Modify: `webgpu/vercel.json`
- Modify: `.gitignore`
- Modify: `.github/workflows/webgpu.yml`

- [x] Add the two approved binaries.
- [x] Record SHA-256, triangle, vertex, and bone metadata.
- [x] Use immutable GLB caching and no-store page/runtime caching.
- [x] Permit only the two exact public GLBs in future private-asset checks.

### Task 3: Build marker and direct verification

**Files:**
- Modify: `webgpu/index.html`

- [x] Advance the build marker to `rigged-v4-public-assets`.
- [x] Pin the application module to the asset-publication commit.
- [x] Run local unit tests, GLB structure checks, and static HTTP checks without CI.
- [x] Deploy Production and verify root, manifest, LOD0, and LOD1 HTTP responses.
