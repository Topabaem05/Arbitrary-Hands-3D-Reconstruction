# Public Rigged Hand Assets Design

## Goal

Make the approved LOD0 and LOD1 rigged hand GLBs public Vercel static assets so a first-time visitor sees the neutral filled hand immediately without selecting a local file.

## Runtime contract

1. A locally saved IndexedDB GLB remains an optional user override.
2. When no override exists, `RiggedAssetStore.load()` fetches the LOD-specific same-origin asset:
   - `/assets/hand_rigged_v3_lod0.glb`
   - `/assets/hand_rigged_v3_lod1.glb`
3. The GLB header is validated before Three.js parsing.
4. Failure of the bundled asset falls back to the existing local file selector.
5. Camera permission remains a separate explicit user action.

## Publication boundary

The user explicitly approved public publication of these two generated GLBs. The source OBJ, MANO data, reports, and any other generated variants remain excluded. The public files are immutable and identified in the model manifest by SHA-256, geometry counts, and bone count.

## Deployment

The page/runtime uses `no-store`; fingerprinted GLBs use a one-year immutable cache. The build marker advances to `rigged-v4-public-assets` so stale HTML is readily identifiable.
