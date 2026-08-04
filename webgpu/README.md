# ACR Hand Lab WebGPU

A zero-build browser application for real-time hand tracking with ONNX Runtime Web.

```bash
python3 -m http.server 4173 --directory webgpu
```

Open `http://localhost:4173/?demo=1` for the dependency-free synthetic UI demo. Open the normal root URL and press **Start camera** to load the ONNX models and start WebGPU inference.

The public runtime uses OpenCV's Apache-2.0 MediaPipe ONNX models. It does not include MANO files or a MANO-derived ACR model. See `../docs/WEBGPU.md` for local ACR conversion.
