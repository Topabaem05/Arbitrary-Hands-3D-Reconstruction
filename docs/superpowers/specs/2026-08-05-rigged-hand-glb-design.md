# Uploaded Hand GLB Rigging Design

## Goal

Convert the uploaded high-resolution neutral-pose hand OBJ into a browser-ready, bone-rigged GLB and replace the current MANO-specific surface path with a generic local rigged-GLB path that follows MediaPipe hand landmarks at a 60 Hz render cadence on a base Apple M1 Mac in current Chrome.

## Source Asset Facts

- Source archive: `hand_v1_L1...zip`
- Source mesh: `12683_hand_v1_FINAL.obj`
- Source topology after OBJ triangulation: approximately 51,000 vertices and 99,700 triangles.
- Source asset contains no armature, joints, skin weights, usable material, or texture.
- Rest pose is a flat open left hand, fingers directed toward negative X and thumb toward negative Y.

## Selected Architecture

1. A local Python converter reads the OBJ, normalizes it, produces two voxel-cluster LODs, generates a 21-joint MediaPipe-compatible hand skeleton, computes four-influence skin weights, and writes a standards-compliant GLB 2.0 file.
2. The GLB contains a skinned left-hand mesh. The browser mirrors the rig for right-hand tracking without modifying the source file.
3. The browser loads the user-selected GLB through Three.js `GLTFLoader`, validates required bone names, and stores the private file in IndexedDB.
4. Each inference result updates target landmark positions. A separate `requestAnimationFrame` loop interpolates targets and updates bone quaternions, root scale, root orientation, and root translation before rendering.
5. The main viewport remains a filled hand surface on exact black `#000000`; the only other visible element is the lower-right camera PIP. Bones are hidden by default and shown only with `?debugBones=1`.

## Bone Contract

The asset uses 21 stable node names matching MediaPipe landmark semantics:

- `wrist`
- `thumb_cmc`, `thumb_mcp`, `thumb_ip`, `thumb_tip`
- `index_mcp`, `index_pip`, `index_dip`, `index_tip`
- `middle_mcp`, `middle_pip`, `middle_dip`, `middle_tip`
- `ring_mcp`, `ring_pip`, `ring_dip`, `ring_tip`
- `pinky_mcp`, `pinky_pip`, `pinky_dip`, `pinky_tip`

The skin has at most four normalized influences per vertex. The GLB preserves the original visible hand shape while reducing triangle count for M1 browser delivery.

## LOD and Performance Contract

- LOD0 target: 14,000–18,000 triangles.
- LOD1 target: 5,000–8,000 triangles.
- Default browser asset: LOD0.
- `?lod=1` selects LOD1.
- Default tracking: one hand; `?hands=2` explicitly enables two.
- Render cadence target: 60 Hz.
- Fresh inference FPS is measured separately and must not be presented as 60 FPS unless measured.
- Three.js renderer, geometry, material, skeleton, and shader programs are created once.
- Per-frame work updates bone transforms and renders; it does not rebuild geometry.

## Private Asset Boundary

The uploaded OBJ and generated rigged GLB are treated as private assets:

- They are not committed to GitHub.
- They are not uploaded to Vercel.
- The browser imports the GLB from a local file and persists it in IndexedDB.
- Generated asset names are ignored by Git.
- The generated GLB is returned to the user as a downloadable artifact from this session.

## Error Handling

The browser stops before camera inference when:

- WebGL2 is unavailable.
- The GLB cannot be parsed.
- A required bone is missing.
- The GLB does not contain a `SkinnedMesh`.
- Skin attributes or inverse bind matrices are invalid.

Camera, model-loading, and inference errors remain concise and recoverable through the start overlay.

## Verification

- Python tests verify skeleton names, hierarchy, normalized top-four weights, GLB chunk validity, triangle budgets, and finite data.
- JavaScript tests verify local GLB persistence, required-bone validation, MediaPipe-to-bone mapping, independent render cadence, exact black background, lower-right PIP, and optional debug bone display.
- CI runs all existing tests plus the new rigging and browser-rig tests.
- The production deployment remains asset-free and serves only the generic loader/runtime.
- Base-M1 performance remains a measurement gate; the browser console reports display FPS, fresh inference FPS, capture time, inference time, and render time.