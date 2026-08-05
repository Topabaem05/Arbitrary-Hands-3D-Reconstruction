import { CameraController, captureVideoFrame, scheduleVideoFrames } from './src/camera.js';
import { RiggedAssetStore, readGlbFile } from './src/rigged-asset-store.js';
import { RiggedHandRenderer } from './src/rigged-hand-renderer.js';
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
  riggedHandInput: document.querySelector('#riggedHandInput'),
  liveStatus: document.querySelector('#liveStatus'),
};

const params = new URLSearchParams(location.search);
const maxHands = params.get('hands') === '2' ? 2 : 1;
const lod = params.get('lod') === '1' ? 1 : 0;
const debugBones = params.get('debugBones') === '1';
const overlayRenderer = new OverlayRenderer(elements.overlay, { mirrored: true });
const camera = new CameraController(elements.video);
const assetStore = new RiggedAssetStore({ key: `rigged-hand-glb-v1:lod${lod}` });
const performanceMetrics = new RuntimePerformance({ windowMs: 5000 });

let rigRenderer = null;
let riggedAsset = null;
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

function assetPrompt(detail = `LOD${lod} 리깅 GLB를 선택하세요`) {
  showPrompt('리깅 손 모델 불러오기', detail);
}

function cameraPrompt(
  detail = 'WebGPU 추론과 GPU bone skinning이 이 브라우저 안에서만 실행됩니다',
) {
  showPrompt('카메라 시작', detail);
}

function errorMessage(error) {
  if (error?.name === 'NotAllowedError') {
    return '브라우저 카메라 권한을 허용한 뒤 다시 누르세요';
  }
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
    void stopCamera({
      title: '다시 시도',
      detail: '실시간 손 추론 중 오류가 발생했습니다',
      level: 'error',
    });
    return;
  }
  latestResult = scheduled.value;
  performanceMetrics.recordInference(latestResult?.timing, scheduled.duration * 1000);
  rigRenderer?.setHands(latestResult?.hands || [], now);
}

function renderLoop(now) {
  const started = performance.now();
  updateLatestInference(now);
  rigRenderer?.render(now);
  if (mode === 'camera' && elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    overlayRenderer.draw(elements.video, latestResult?.hands || [], { showStats: false });
  }
  performanceMetrics.recordRender(performance.now() - started);
  const report = performanceMetrics.sample(now);
  if (report) {
    console.info('ACR M1 rig performance', { lod, maxHands, debugBones, ...report });
  }
  renderRequest = requestAnimationFrame(renderLoop);
}

async function submitCurrentVideoFrame() {
  if (!tracker || !scheduler || snapshotPending) return;
  if (!(elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA)) return;
  snapshotPending = true;
  try {
    const started = performance.now();
    const frame = await captureVideoFrame(elements.video);
    performanceMetrics.recordCapture(performance.now() - started);
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
  if (!riggedAsset) {
    elements.riggedHandInput.click();
    return;
  }
  if (mode !== 'idle') return;
  mode = 'loading';
  showPrompt('카메라 연결 중', 'WebGPU 모델을 불러오고 있습니다', 'working');
  try {
    await Promise.all([camera.start(), loadTracker()]);
    tracker.reset();
    rigRenderer.reset();
    latestResult = null;
    latestGeneration = scheduler?.latest?.generation || 0;
    elements.pip.hidden = false;
    mode = 'camera';
    hidePrompt('카메라와 리깅 손 모델 실행 중');
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
  rigRenderer?.reset();
  mode = riggedAsset ? 'idle' : 'asset';
  if (riggedAsset) {
    showPrompt(
      title,
      detail || 'WebGPU 추론과 GPU bone skinning이 이 브라우저 안에서만 실행됩니다',
      level,
    );
  } else {
    assetPrompt();
  }
}

async function importAsset(file) {
  mode = 'loading';
  showPrompt('리깅 모델 확인 중', 'GLB와 21개 bone 구조를 검사하고 있습니다', 'working');
  try {
    const candidate = await readGlbFile(file);
    const metadata = await rigRenderer.load(candidate.bytes.slice(0));
    riggedAsset = await assetStore.save(candidate);
    mode = 'idle';
    cameraPrompt(
      `${riggedAsset.name} · ${Math.round(metadata.triangleCount).toLocaleString()} triangles · ${metadata.boneCount} bones`,
    );
  } catch (error) {
    console.error('Rigged GLB import failed', error);
    mode = 'asset';
    showPrompt(
      '다시 선택',
      error instanceof Error ? error.message : '리깅 GLB를 읽지 못했습니다',
      'error',
    );
  } finally {
    elements.riggedHandInput.value = '';
  }
}

async function clearPrivateAsset() {
  await stopCamera();
  await assetStore.clear();
  riggedAsset = null;
  rigRenderer?.reset();
  mode = 'asset';
  assetPrompt('저장된 로컬 리깅 모델을 삭제했습니다');
}

elements.start.addEventListener('click', () => void startCamera());
elements.riggedHandInput.addEventListener('change', () => {
  const [file] = elements.riggedHandInput.files;
  if (file) void importAsset(file);
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && mode === 'camera') void stopCamera();
  if (event.getModifierState('Shift') && event.key === 'Delete') {
    void clearPrivateAsset();
  }
});
window.addEventListener('beforeunload', () => {
  cancelCaptureLoop?.();
  cancelAnimationFrame(renderRequest);
  void camera.stop();
  void scheduler?.stop();
  rigRenderer?.dispose();
});

(async function initialize() {
  try {
    rigRenderer = new RiggedHandRenderer(elements.mesh, { maxHands, debugBones });
  } catch (error) {
    mode = 'unsupported';
    showPrompt(
      'WebGL2 필요',
      error instanceof Error ? error.message : '이 브라우저는 리깅 손 렌더링을 지원하지 않습니다',
      'error',
    );
    return;
  }
  renderRequest = requestAnimationFrame(renderLoop);
  try {
    riggedAsset = await assetStore.load();
    if (riggedAsset) {
      const metadata = await rigRenderer.load(riggedAsset.bytes.slice(0));
      mode = 'idle';
      cameraPrompt(
        `저장된 LOD${lod} 모델 · ${Math.round(metadata.triangleCount).toLocaleString()} triangles`,
      );
    } else {
      mode = 'asset';
      assetPrompt();
    }
    await ensureManifest();
  } catch (error) {
    console.error('Initialization failed', error);
    await assetStore.clear().catch(() => {});
    riggedAsset = null;
    mode = 'asset';
    showPrompt(
      '다시 선택',
      error instanceof Error ? error.message : '초기화에 실패했습니다',
      'error',
    );
  }
})();
