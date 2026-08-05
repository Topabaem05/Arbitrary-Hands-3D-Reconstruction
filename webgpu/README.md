# ACR Hand Lab — private MANO web mesh

The public page runs palm/landmark inference locally with ONNX Runtime Web and renders a filled MANO-topology surface with WebGL2. The MANO data is **not** included in the repository or Vercel deployment.

## 1. Create your private MANO browser bundle

Place your licensed files at `mano/MANO_LEFT.pkl` and `mano/MANO_RIGHT.pkl`, then run:

```bash
python tools/convert_mano_browser_bundle.py \
  --mano-root mano \
  --output mano-browser-bundle.json \
  --acknowledge-mano-license
```

The generated JSON contains MANO-derived vertices, faces, skinning weights, and rest joints. Keep it private. It is ignored by Git.

## 2. Run locally

```bash
python3 -m http.server 4173 --directory webgpu
```

Open `http://localhost:4173/`. On the first visit:

1. Select `mano-browser-bundle.json`.
2. The browser validates it and stores it in IndexedDB.
3. Press **카메라 시작**.

`Shift+Delete` clears the locally stored bundle. `Escape` stops the camera. Add `?hands=2` to enable two-hand rendering; the default one-hand mode is the M1 performance path.

## Performance contract

- Target: base Apple M1 Mac, current Chrome.
- Mesh/display loop: 60 Hz target through `requestAnimationFrame`.
- Fresh inference: measured separately; interpolation never counts as a new inference.
- Background: pure black.
- Camera: small lower-right picture-in-picture.

See `../docs/WEBGPU.md` for architecture, licensing, and measurement details.
