import { CameraController, scheduleVideoFrames } from './src/camera.js';
import { OverlayRenderer, Skeleton3DRenderer } from './src/renderer.js';
import { WebGpuHandTracker, loadModelManifest } from './src/runtime.js';
import { LatestFrameScheduler } from './src/scheduler.js';

const elements = {
  video: document.querySelector('#cameraVideo'),
  overlay: document.querySelector('#overlayCanvas'),
  skeleton: document.querySelector('#skeletonCanvas'),
  pip: document.querySelector('#cameraPip'),
  start: document.querySelector('#startCamera'),
  startLabel: document.querySelector('#startLabel'),
  startDetail: document.querySelector('#startDetail'),
  liveStatus: document.querySelector('#liveStatus'),
};

const overlayRenderer = new OverlayRenderer(elements.overlay, { mirrored: true });
const skeletonRenderer = new Skeleton3DRenderer(elements.skeleton);
const camera = new CameraController(elements.video);

let manifest = null;
let tracker = null;
let scheduler = null;
let loadingPromise = null;
let cancelVideoLoop = null;
let latestResult = null;
let latestGeneration = 0;
let snapshotPending = false;
let mode = 'idle';

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

function cameraErrorMessage(error) {
  if (error?.name === 'NotAllowedError') {
    return '브라우저의 카메라 권한을 허용한 뒤 다시 누르세요';
  }
  if (error?.name === 'NotFoundError') {
    return '사용할 수 있는 카메라를 찾지 못했습니다';
  }
  return '카메라 또는 WebGPU 모델을 시작하지 못했습니다';
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
      maxHands: 2,
      detectorInterval: 8,
      scoreThreshold: 0.55,
    });
    const diagnostics = await nextTracker.load();
    tracker = nextTracker;
    scheduler = new LatestFrameScheduler(
      async (bitmap) => tracker.infer(bitmap),
      { dispose: (bitmap) => bitmap?.close?.() },
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

function updateLatestInference() {
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
}

function renderCurrentFrame() {
  const hands = latestResult?.hands || [];
  overlayRenderer.draw(elements.video, hands, { showStats: false });
  skeletonRenderer.draw(hands);
}

async function submitCurrentVideoFrame() {
  if (!tracker || !scheduler || snapshotPending) return;
  if (scheduler.pending !== undefined) return;
  if (!(elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA)) return;

  snapshotPending = true;
  try {
    scheduler.submit(await createImageBitmap(elements.video));
  } catch (error) {
    console.error('Camera snapshot failed', error);
  } finally {
    snapshotPending = false;
  }
}

function startCameraRenderLoop() {
  cancelVideoLoop?.();
  cancelVideoLoop = scheduleVideoFrames(elements.video, () => {
    if (mode !== 'camera') return;
    updateLatestInference();
    renderCurrentFrame();
    void submitCurrentVideoFrame();
  });
}

async function startCamera() {
  if (mode !== 'idle') return;
  mode = 'loading';
  showPrompt('카메라 연결 중', 'WebGPU 모델을 불러오고 있습니다', 'working');

  try {
    const cameraPromise = camera.start();
    const trackerPromise = loadTracker();
    await Promise.all([cameraPromise, trackerPromise]);

    tracker.reset();
    latestResult = null;
    latestGeneration = scheduler?.latest?.generation || 0;
    elements.pip.hidden = false;
    mode = 'camera';
    document.body.dataset.state = 'running';
    hidePrompt('카메라 추론 실행 중');
    startCameraRenderLoop();
  } catch (error) {
    console.error('Unable to start camera experience', error);
    await camera.stop();
    elements.pip.hidden = true;
    mode = 'idle';
    delete document.body.dataset.state;
    showPrompt('다시 시도', cameraErrorMessage(error), 'error');
  }
}

async function stopCamera({
  title = '카메라 시작',
  detail = 'WebGPU로 이 브라우저 안에서만 처리됩니다',
  level = 'idle',
} = {}) {
  cancelVideoLoop?.();
  cancelVideoLoop = null;
  await camera.stop();
  elements.pip.hidden = true;
  latestResult = null;
  mode = 'idle';
  delete document.body.dataset.state;
  skeletonRenderer.draw([]);
  showPrompt(title, detail, level);
}

elements.start.addEventListener('click', () => void startCamera());
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && mode === 'camera') void stopCamera();
});
window.addEventListener('beforeunload', () => {
  cancelVideoLoop?.();
  void camera.stop();
  void scheduler?.stop();
});

(async function initialize() {
  skeletonRenderer.draw([]);
  try {
    await ensureManifest();
    elements.liveStatus.textContent = '카메라 시작 준비 완료';
  } catch (error) {
    console.error('Model manifest failed to load', error);
    showPrompt('다시 시도', '모델 정보를 불러오지 못했습니다', 'error');
  }
})();
