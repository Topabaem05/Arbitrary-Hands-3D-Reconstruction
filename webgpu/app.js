import { CameraController, loadImageFile, scheduleVideoFrames } from './src/camera.js';
import { createDemoFrame, syntheticHands } from './src/demo.js';
import { OverlayRenderer, Skeleton3DRenderer } from './src/renderer.js';
import { WebGpuHandTracker, loadModelManifest } from './src/runtime.js';
import { ExponentialRate, LatestFrameScheduler } from './src/scheduler.js';

const elements = {
  video: document.querySelector('#cameraVideo'),
  overlay: document.querySelector('#overlayCanvas'),
  skeleton: document.querySelector('#skeletonCanvas'),
  start: document.querySelector('#startCamera'),
  stop: document.querySelector('#stopCamera'),
  demo: document.querySelector('#startDemo'),
  image: document.querySelector('#imageInput'),
  backend: document.querySelector('#backendSelect'),
  maxHands: document.querySelector('#maxHands'),
  mirror: document.querySelector('#mirrorCamera'),
  detectorUrl: document.querySelector('#detectorUrl'),
  landmarkUrl: document.querySelector('#landmarkUrl'),
  detectorFile: document.querySelector('#detectorFile'),
  landmarkFile: document.querySelector('#landmarkFile'),
  reload: document.querySelector('#reloadModels'),
  statusDot: document.querySelector('#statusDot'),
  status: document.querySelector('#statusText'),
  backendValue: document.querySelector('#backendValue'),
  inferenceValue: document.querySelector('#inferenceValue'),
  displayValue: document.querySelector('#displayValue'),
  latencyValue: document.querySelector('#latencyValue'),
  diagnostics: document.querySelector('#diagnostics'),
  privacy: document.querySelector('#privacyState'),
  mode: document.querySelector('#modeValue'),
};

const overlayRenderer = new OverlayRenderer(elements.overlay, { mirrored: elements.mirror.checked });
const skeletonRenderer = new Skeleton3DRenderer(elements.skeleton);
const camera = new CameraController(elements.video);
const displayRate = new ExponentialRate(0.12);
const inferenceRate = new ExponentialRate(0.2);
let manifest = null;
let tracker = null;
let scheduler = null;
let cancelVideoLoop = null;
let lastDisplayTime = performance.now();
let latestResult = null;
let staticSource = null;
let demoFrame = createDemoFrame();
let demoAnimation = 0;
let mode = 'idle';
let loadingPromise = null;
let snapshotPending = false;

function setStatus(message, level = 'idle') {
  elements.status.textContent = message;
  elements.statusDot.dataset.level = level;
}

function setMode(nextMode) {
  mode = nextMode;
  elements.mode.textContent = nextMode;
  elements.stop.disabled = nextMode === 'idle';
}

function appendDiagnostic(message) {
  const timestamp = new Date().toLocaleTimeString();
  elements.diagnostics.textContent = `[${timestamp}] ${message}\n${elements.diagnostics.textContent}`.trim();
}

function selectedModelSources() {
  return {
    detectorSource: elements.detectorFile.files[0] || elements.detectorUrl.value.trim(),
    landmarkSource: elements.landmarkFile.files[0] || elements.landmarkUrl.value.trim(),
  };
}

function diagnosticsText(diagnostics) {
  return diagnostics.sessions.map((session) => {
    if (session.error) return `${session.label}: ${session.backend} failed — ${session.error}`;
    return `${session.label}: ${session.backend}; inputs=${session.inputs.join(',')}; outputs=${session.outputs.join(',')}`;
  }).join('\n');
}

async function ensureManifest() {
  if (manifest) return manifest;
  manifest = await loadModelManifest();
  elements.detectorUrl.value = manifest.models.palm.url;
  elements.landmarkUrl.value = manifest.models.landmark.url;
  return manifest;
}

async function loadTracker({ force = false } = {}) {
  if (tracker && !force) return tracker;
  if (loadingPromise && !force) return loadingPromise;
  loadingPromise = (async () => {
    setStatus('Loading ONNX models…', 'working');
    await ensureManifest();
    const nextTracker = new WebGpuHandTracker({
      manifest,
      backend: elements.backend.value,
      maxHands: Number(elements.maxHands.value),
      detectorInterval: 8,
      scoreThreshold: 0.55,
    });
    const diagnostics = await nextTracker.load(selectedModelSources());
    tracker = nextTracker;
    scheduler = new LatestFrameScheduler(
      async (bitmap) => tracker.infer(bitmap),
      { dispose: (bitmap) => bitmap?.close?.() },
    );
    elements.backendValue.textContent = diagnostics.backend + (diagnostics.graphCapture ? ' + capture' : '');
    elements.privacy.textContent = 'on-device';
    appendDiagnostic(diagnosticsText(diagnostics));
    setStatus(`Models ready on ${diagnostics.backend.toUpperCase()}`, 'ready');
    return tracker;
  })();
  try {
    return await loadingPromise;
  } catch (error) {
    tracker = null;
    scheduler = null;
    const message = error instanceof Error ? error.message : String(error);
    appendDiagnostic(message);
    setStatus('Model initialization failed', 'error');
    throw error;
  } finally {
    loadingPromise = null;
  }
}

function statsFromResult(result = latestResult) {
  const duration = result?.timing?.totalMilliseconds || 0;
  elements.inferenceValue.textContent = `${inferenceRate.value.toFixed(1)} FPS`;
  elements.displayValue.textContent = `${displayRate.value.toFixed(1)} FPS`;
  elements.latencyValue.textContent = duration > 0 ? `${duration.toFixed(1)} ms` : '—';
  return {
    displayFps: displayRate.value,
    inferenceFps: inferenceRate.value,
    backend: result?.backend || elements.backendValue.textContent,
  };
}

function renderSource(source, hands, now = performance.now()) {
  const elapsed = Math.max(1, now - lastDisplayTime) / 1000;
  displayRate.updateDuration(elapsed);
  lastDisplayTime = now;
  const stats = statsFromResult();
  overlayRenderer.draw(source, hands, stats);
  skeletonRenderer.draw(hands);
}

async function submitCurrentVideoFrame() {
  if (!tracker || !scheduler || snapshotPending || scheduler.pending !== undefined) return;
  if (!(elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA)) return;
  snapshotPending = true;
  try {
    const bitmap = await createImageBitmap(elements.video);
    scheduler.submit(bitmap);
  } catch (error) {
    appendDiagnostic(`Camera snapshot failed: ${error instanceof Error ? error.message : error}`);
  } finally {
    snapshotPending = false;
  }
}

function updateLatestInference() {
  const scheduled = scheduler?.latest;
  if (!scheduled || scheduled.generation === latestResult?.generation) return;
  if (scheduled.error) {
    appendDiagnostic(`Inference failed: ${scheduled.error.message || scheduled.error}`);
    setStatus('Inference error', 'error');
    return;
  }
  latestResult = { ...scheduled.value, generation: scheduled.generation };
  inferenceRate.updateDuration(scheduled.duration);
}

function startCameraRenderLoop() {
  cancelVideoLoop?.();
  cancelVideoLoop = scheduleVideoFrames(elements.video, (now) => {
    if (mode !== 'camera') return;
    updateLatestInference();
    renderSource(elements.video, latestResult?.hands || [], now);
    void submitCurrentVideoFrame();
  });
}

async function startCamera() {
  await stopActiveMode();
  setStatus('Requesting camera permission…', 'working');
  try {
    await Promise.all([camera.start(), loadTracker()]);
    tracker.reset();
    latestResult = null;
    setMode('camera');
    setStatus('Camera tracking active', 'ready');
    startCameraRenderLoop();
  } catch (error) {
    appendDiagnostic(error instanceof Error ? error.message : String(error));
    setStatus('Camera could not start', 'error');
  }
}

async function startDemo() {
  await stopActiveMode();
  setMode('demo');
  setStatus('Synthetic demo mode', 'ready');
  elements.backendValue.textContent = tracker ? elements.backendValue.textContent : 'demo';
  const loop = (now) => {
    if (mode !== 'demo') return;
    const hands = syntheticHands(now, demoFrame.width, demoFrame.height, Number(elements.maxHands.value));
    renderSource(demoFrame, hands, now);
    demoAnimation = requestAnimationFrame(loop);
  };
  demoAnimation = requestAnimationFrame(loop);
}

async function runImage(file) {
  await stopActiveMode();
  setStatus('Loading image and models…', 'working');
  try {
    const [bitmap] = await Promise.all([loadImageFile(file), loadTracker()]);
    staticSource?.close?.();
    staticSource = bitmap;
    tracker.reset();
    const started = performance.now();
    const result = await tracker.infer(bitmap);
    inferenceRate.updateDuration((performance.now() - started) / 1000);
    latestResult = result;
    setMode('image');
    renderSource(bitmap, result.hands);
    setStatus(`Image processed: ${result.hands.length} hand${result.hands.length === 1 ? '' : 's'}`, 'ready');
  } catch (error) {
    appendDiagnostic(error instanceof Error ? error.message : String(error));
    setStatus('Image inference failed', 'error');
  }
}

async function stopActiveMode() {
  cancelVideoLoop?.();
  cancelVideoLoop = null;
  cancelAnimationFrame(demoAnimation);
  demoAnimation = 0;
  await camera.stop();
  latestResult = null;
  setMode('idle');
  overlayRenderer.draw(demoFrame, [], statsFromResult(null));
  skeletonRenderer.draw([]);
}

async function reloadModels() {
  setStatus('Reloading model sessions…', 'working');
  await scheduler?.stop();
  scheduler = null;
  tracker = null;
  latestResult = null;
  try {
    await loadTracker({ force: true });
  } catch {
    // Detailed diagnostics are already exposed.
  }
}

elements.start.addEventListener('click', () => void startCamera());
elements.stop.addEventListener('click', () => void stopActiveMode());
elements.demo.addEventListener('click', () => void startDemo());
elements.image.addEventListener('change', () => {
  const [file] = elements.image.files;
  if (file) void runImage(file);
});
elements.reload.addEventListener('click', () => void reloadModels());
elements.maxHands.addEventListener('change', () => tracker?.setMaxHands(elements.maxHands.value));
elements.mirror.addEventListener('change', () => overlayRenderer.setMirrored(elements.mirror.checked));
elements.backend.addEventListener('change', () => {
  elements.backendValue.textContent = 'reload required';
  setStatus('Backend changed; reload models', 'idle');
});
window.addEventListener('beforeunload', () => {
  void camera.stop();
  void scheduler?.stop();
  staticSource?.close?.();
});

(async function initialize() {
  overlayRenderer.draw(demoFrame, [], { displayFps: 0, inferenceFps: 0, backend: 'not loaded' });
  skeletonRenderer.draw([]);
  try {
    await ensureManifest();
    appendDiagnostic(`Manifest ${manifest.version} loaded. Models are fetched only after camera/image inference starts.`);
  } catch (error) {
    appendDiagnostic(error instanceof Error ? error.message : String(error));
    setStatus('Manifest load failed', 'error');
  }
  if (new URLSearchParams(location.search).get('demo') === '1') void startDemo();
})();
