import { CameraController, captureVideoFrame, scheduleVideoFrames } from './src/camera.js';
import { loadManoBundleFile, ManoBundleStore } from './src/mano-bundle.js';
import { HandMeshRenderer } from './src/mesh-renderer.js';
import { RuntimePerformance } from './src/performance.js';
import { OverlayRenderer } from './src/renderer.js';
import { WebGpuHandTracker, loadModelManifest } from './src/runtime.js';
import { LatestFrameScheduler } from './src/scheduler.js';

const elements = {
  video: document.querySelector('#cameraVideo'),
  overlay: document.querySelector('#overlayCanvas'),
  mesh: document.querySelector('#meshCanvas'),
  pip: document.querySelector('#cameraPip'),
  start: document.querySelector('#startCamera'),
  startLabel: document.querySelector('#startLabel'),
  startDetail: document.querySelector('#startDetail'),
  manoBundleInput: document.querySelector('#manoBundleInput'),
  liveStatus: document.querySelector('#liveStatus'),
};

const overlayRenderer = new OverlayRenderer(elements.overlay, { mirrored: true });
const camera = new CameraController(elements.video);
const bundleStore = new ManoBundleStore();
const maxHands = new URLSearchParams(location.search).get('hands') === '2' ? 2 : 1;
const performanceMetrics = new RuntimePerformance({ windowMs: 5000 });

let meshRenderer = null;
let manoBundle = null;
let manifest = null;
let tracker = null;
let scheduler = null;
let loadingPromise = null;
let cancelCaptureLoop = null;
let latestResult = null;
let latestGeneration = 0;
let snapshotPending = false;
let mode = 'initializing';
let renderRequest = 0;

function showPrompt(title, detail, level = 'idle') {
  elements.startLabel.textContent = title;
  elements.startDetail.textContent = detail;
  elements.start.dataset.level = level;
  elements.start.disabled = level === 'working';
  elements.start.setAttribute('aria-busy', level === 'working' ? 'true' : 'false');
  elements.start.classList.remove('is-hidden');
  elements.liveStatus.textContent = `${title}. ${detail}`;
}

function hidePrompt(message) {
  elements.start.disabled = false;
  elements.start.setAttribute('aria-busy', 'false');
  elements.start.classList.add('is-hidden');
  elements.liveStatus.textContent = message;
}

function bundlePrompt() {
  showPrompt('MANO 모델 불러오기', '로컬 변환한 mano-browser-bundle.json을 선택하세요');
}

function cameraPrompt(detail = 'WebGPU 추론과 MANO 메시가 이 브라우저 안에서만 실행됩니다') {
  showPrompt('카메라 시작', detail);
}

function errorMessage(error) {
  if (error?.name === 'NotAllowedError') return '브라우저 카메라 권한을 허용한 뒤 다시 누르세요';
  if (error?.name === 'NotFoundError') return '사용할 수 있는 카메라를 찾지 못했습니다';
  return error instanceof Error ? error.message : '카메라 또는 모델을 시작하지 못했습니다';
}

async function ensureManifest() {
  if (!manifest) manifest = await loadModelManifest();
  return manifest;
}

async function loadTracker() {
  if (tracker) return tracker;
  if (loadingPromise) return loadingPromise;
  loadingPromise = (async () => {
    const modelManifest = await ensureManifest();
    const nextTracker = new WebGpuHandTracker({
      manifest: modelManifest,
      backend: 'auto',
      maxHands,
      detectorInterval: 12,
      scoreThreshold: 0.55,
    });
    const diagnostics = await nextTracker.load();
    tracker = nextTracker;
    scheduler = new LatestFrameScheduler(
      async (frame) => tracker.infer(frame),
      { dispose: (frame) => frame?.close?.() },
    );
    console.info('ACR Hand Lab runtime', diagnostics);
    return tracker;
  })();
  try {
    return await loadingPromise;
  } finally {
    loadingPromise = null;
  }
}

function updateLatestInference(now) {
  const scheduled = scheduler?.latest;
  if (!scheduled || scheduled.generation === latestGeneration) return;
  latestGeneration = scheduled.generation;
  if (scheduled.error) {
    console.error('WebGPU hand inference failed', scheduled.error);
    void stopCamera({ title: '다시 시도', detail: '실시간 손 추론 중 오류가 발생했습니다', level: 'error' });
    return;
  }
  latestResult = scheduled.value;
  performanceMetrics.recordInference(latestResult?.timing, scheduled.duration * 1000);
  meshRenderer?.updateHands(latestResult?.hands || [], now);
}

function renderLoop(now) {
  const renderStarted = performance.now();
  updateLatestInference(now);
  meshRenderer?.render(now);
  if (mode === 'camera' && elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    overlayRenderer.draw(elements.video, latestResult?.hands || []);
  }
  performanceMetrics.recordRender(performance.now() - renderStarted);
  const report = performanceMetrics.sample(now);
  if (report) console.info('ACR M1 performance', report);
  renderRequest = requestAnimationFrame(renderLoop);
}

async function submitCurrentVideoFrame() {
  if (!tracker || !scheduler || snapshotPending) return;
  if (!(elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA)) return;
  snapshotPending = true;
  try {
    const captureStarted = performance.now();
    const frame = await captureVideoFrame(elements.video);
    performanceMetrics.recordCapture(performance.now() - captureStarted);
    scheduler.submit(frame);
  } catch (error) {
    console.error('Camera snapshot failed', error);
  } finally {
    snapshotPending = false;
  }
}

function startCaptureLoop() {
  cancelCaptureLoop?.();
  cancelCaptureLoop = scheduleVideoFrames(elements.video, () => {
    if (mode === 'camera') void submitCurrentVideoFrame();
  });
}

async function startCamera() {
  if (!manoBundle) {
    elements.manoBundleInput.click();
    return;
  }
  if (mode !== 'idle') return;
  mode = 'loading';
  showPrompt('카메라 연결 중', 'WebGPU 모델을 불러오고 있습니다', 'working');
  try {
    await Promise.all([camera.start(), loadTracker()]);
    tracker.reset();
    meshRenderer.resetTracking();
    latestResult = null;
    latestGeneration = scheduler?.latest?.generation || 0;
    elements.pip.hidden = false;
    mode = 'camera';
    hidePrompt('카메라와 MANO 메시 실행 중');
    startCaptureLoop();
  } catch (error) {
    console.error('Unable to start camera experience', error);
    await camera.stop();
    elements.pip.hidden = true;
    mode = 'idle';
    showPrompt('다시 시도', errorMessage(error), 'error');
  }
}

async function stopCamera({ title = '카메라 시작', detail, level = 'idle' } = {}) {
  cancelCaptureLoop?.();
  cancelCaptureLoop = null;
  await camera.stop();
  elements.pip.hidden = true;
  latestResult = null;
  meshRenderer?.resetTracking();
  mode = manoBundle ? 'idle' : 'bundle';
  if (manoBundle) showPrompt(title, detail || 'WebGPU 추론과 MANO 메시가 이 브라우저 안에서만 실행됩니다', level);
  else bundlePrompt();
}

async function importBundle(file) {
  mode = 'loading';
  showPrompt('MANO 모델 확인 중', '파일 구조와 메시 데이터를 검사하고 있습니다', 'working');
  try {
    const bundle = await loadManoBundleFile(file);
    await bundleStore.save(bundle);
    manoBundle = bundle;
    meshRenderer.setBundle(bundle);
    mode = 'idle';
    cameraPrompt('MANO 모델이 이 브라우저에 저장되었습니다');
  } catch (error) {
    console.error('MANO bundle import failed', error);
    mode = 'bundle';
    showPrompt('다시 선택', error instanceof Error ? error.message : 'MANO 번들을 읽지 못했습니다', 'error');
  } finally {
    elements.manoBundleInput.value = '';
  }
}

async function clearPrivateBundle() {
  await stopCamera();
  await bundleStore.clear();
  manoBundle = null;
  meshRenderer?.setBundle(null);
  mode = 'bundle';
  bundlePrompt();
}

elements.start.addEventListener('click', () => void startCamera());
elements.manoBundleInput.addEventListener('change', () => {
  const [file] = elements.manoBundleInput.files;
  if (file) void importBundle(file);
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && mode === 'camera') void stopCamera();
  if (event.getModifierState('Shift') && event.key === 'Delete') void clearPrivateBundle();
});
window.addEventListener('beforeunload', () => {
  cancelCaptureLoop?.();
  cancelAnimationFrame(renderRequest);
  void camera.stop();
  void scheduler?.stop();
  meshRenderer?.dispose();
});

(async function initialize() {
  try {
    meshRenderer = new HandMeshRenderer(elements.mesh);
  } catch (error) {
    mode = 'unsupported';
    showPrompt('WebGL2 필요', error instanceof Error ? error.message : '이 브라우저는 손 표면 렌더링을 지원하지 않습니다', 'error');
    return;
  }
  renderRequest = requestAnimationFrame(renderLoop);
  try {
    manoBundle = await bundleStore.load();
    if (manoBundle) {
      meshRenderer.setBundle(manoBundle);
      mode = 'idle';
      cameraPrompt('저장된 MANO 모델을 사용합니다');
    } else {
      mode = 'bundle';
      bundlePrompt();
    }
    await ensureManifest();
  } catch (error) {
    console.error('Initialization failed', error);
    mode = manoBundle ? 'idle' : 'bundle';
    showPrompt('다시 시도', error instanceof Error ? error.message : '초기화에 실패했습니다', 'error');
  }
})();
