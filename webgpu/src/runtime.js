import { computeRoiFromLandmarks, computeRoiFromPalm, createRoiCrop, projectLandmarksFromRoi } from './geometry.js';
import { decodePalmDetections, generatePalmAnchors, identifyPalmOutputs } from './palm.js';

const PALM_SIZE = 192;
const LANDMARK_SIZE = 224;
const ORT_VERSION = '1.26.0';
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let ortLoadPromise = null;

export function ensureOrtRuntime(url = `${ORT_DIST}ort.webgpu.min.js`) {
  if (globalThis.ort) return Promise.resolve(globalThis.ort);
  if (ortLoadPromise) return ortLoadPromise;

  ortLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.crossOrigin = 'anonymous';
    script.async = true;
    script.addEventListener('load', () => {
      if (globalThis.ort) resolve(globalThis.ort);
      else reject(new Error('ONNX Runtime script loaded without exposing globalThis.ort.'));
    }, { once: true });
    script.addEventListener(
      'error',
      () => reject(new Error(`Failed to load ONNX Runtime Web from ${url}.`)),
      { once: true },
    );
    document.head.append(script);
  });

  return ortLoadPromise;
}

function requireOrt() {
  if (!globalThis.ort) throw new Error('ONNX Runtime Web is not initialized.');
  return globalThis.ort;
}

function sourceDimensions(source) {
  const width = source.videoWidth || source.naturalWidth || source.width;
  const height = source.videoHeight || source.naturalHeight || source.height;
  if (!(width > 0 && height > 0)) {
    throw new Error('The camera or image has no readable dimensions.');
  }
  return { width, height };
}

function makeCanvas(width, height) {
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(width, height)
    : document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function imageDataToNhwcFloat(imageData) {
  const output = new Float32Array(imageData.width * imageData.height * 3);
  const pixels = imageData.data;
  let outputIndex = 0;

  for (let pixelIndex = 0; pixelIndex < pixels.length; pixelIndex += 4) {
    output[outputIndex] = pixels[pixelIndex] / 255;
    output[outputIndex + 1] = pixels[pixelIndex + 1] / 255;
    output[outputIndex + 2] = pixels[pixelIndex + 2] / 255;
    outputIndex += 3;
  }

  return output;
}

export function prepareLetterboxInput(source, size = PALM_SIZE, canvas = null) {
  const { width, height } = sourceDimensions(source);
  const output = canvas ?? makeCanvas(size, size);
  output.width = size;
  output.height = size;

  const context = output.getContext('2d', { willReadFrequently: true, alpha: false });
  context.fillStyle = '#000';
  context.fillRect(0, 0, size, size);

  const scale = Math.min(size / width, size / height);
  const drawWidth = width * scale;
  const drawHeight = height * scale;
  const padX = (size - drawWidth) / 2;
  const padY = (size - drawHeight) / 2;
  context.drawImage(source, padX, padY, drawWidth, drawHeight);

  return {
    data: imageDataToNhwcFloat(context.getImageData(0, 0, size, size)),
    transform: {
      inputSize: size,
      sourceWidth: width,
      sourceHeight: height,
      scale,
      padX,
      padY,
    },
    canvas: output,
  };
}

export function prepareRoiInput(source, roi, size = LANDMARK_SIZE, canvas = null) {
  const output = createRoiCrop(source, roi, size, canvas);
  const context = output.getContext('2d', { willReadFrequently: true });
  return {
    data: imageDataToNhwcFloat(context.getImageData(0, 0, size, size)),
    canvas: output,
  };
}

async function modelBytes(source) {
  const isFile = typeof File !== 'undefined' && source instanceof File;
  const isBlob = typeof Blob !== 'undefined' && source instanceof Blob;
  if (isFile || isBlob) return new Uint8Array(await source.arrayBuffer());
  return source;
}

export function buildExecutionAttempts({ backend = 'auto', hasWebGpu = false } = {}) {
  const attempts = [];

  if (hasWebGpu && backend !== 'wasm') {
    attempts.push({ executionProviders: ['webgpu'], graphCapture: false });
  }
  if (backend !== 'webgpu') {
    attempts.push({ executionProviders: ['wasm'], graphCapture: false });
  }
  if (attempts.length === 0) {
    throw new Error('WebGPU was required, but this browser does not expose navigator.gpu.');
  }

  return attempts;
}

export class OrtSessionFactory {
  constructor({ backend = 'auto', wasmPaths = ORT_DIST } = {}) {
    this.backend = backend;
    this.wasmPaths = wasmPaths;
    this.sessionDiagnostics = [];
  }

  async create(source, label) {
    const ort = await ensureOrtRuntime();

    // Shape-related CPU assignments are expected for these models. Keep the
    // browser console focused on actionable runtime errors.
    ort.env.logLevel = 'error';
    ort.env.wasm.wasmPaths = this.wasmPaths;
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
    ort.env.wasm.proxy = false;
    if (ort.env.webgpu) ort.env.webgpu.powerPreference = 'high-performance';

    const resolvedSource = await modelBytes(source);
    const attempts = buildExecutionAttempts({
      backend: this.backend,
      hasWebGpu: Boolean(navigator.gpu),
    });

    let lastError = null;
    for (const attempt of attempts) {
      const executionProvider = attempt.executionProviders[0];
      try {
        const session = await ort.InferenceSession.create(resolvedSource, {
          executionProviders: attempt.executionProviders,
          graphOptimizationLevel: 'all',
          enableGraphCapture: false,
        });
        const diagnostic = {
          label,
          backend: executionProvider,
          inputs: session.inputNames,
          outputs: session.outputNames,
        };
        this.sessionDiagnostics.push(diagnostic);
        return {
          session,
          executionProvider,
          graphCapture: false,
          diagnostic,
        };
      } catch (error) {
        lastError = error;
        this.sessionDiagnostics.push({
          label,
          backend: executionProvider,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    throw new Error(
      `${label} failed to initialize: ${lastError instanceof Error ? lastError.message : lastError}`,
    );
  }
}

function identifyLandmarkOutputs(outputs, outputNames = Object.keys(outputs)) {
  const ordered = outputNames
    .map((name) => [name, outputs[name]])
    .filter(([, tensor]) => tensor);
  const vectors = ordered.filter(([, tensor]) => tensor.data.length === 63);
  const scalars = ordered.filter(([, tensor]) => tensor.data.length === 1);

  if (vectors.length < 1 || scalars.length < 1) {
    const signature = ordered
      .map(([name, tensor]) => `${name}:${tensor.dims.join('x')}`)
      .join(', ');
    throw new Error(`Unexpected landmark output signature (${signature}).`);
  }

  const screen = vectors.find(([name]) => !name.toLowerCase().includes('world')) ?? vectors[0];
  const world = vectors.find(([name]) => name.toLowerCase().includes('world'))
    ?? vectors.find((entry) => entry !== screen)
    ?? null;
  const score = scalars.find(([name]) => /score|conf|presence|flag/i.test(name)) ?? scalars[0];
  const handedness = scalars.find(
    ([name]) => /hand|left|right|lr/i.test(name) && name !== score[0],
  ) ?? scalars.find((entry) => entry !== score) ?? null;

  return {
    landmarks: screen[1],
    worldLandmarks: world?.[1] ?? null,
    score: score[1].data[0],
    handedness: handedness?.[1].data[0] ?? 0.5,
  };
}

function cloneWorldLandmarks(tensor) {
  if (!tensor) return null;
  const result = [];
  for (let index = 0; index < tensor.data.length; index += 3) {
    result.push({
      x: tensor.data[index],
      y: tensor.data[index + 1],
      z: tensor.data[index + 2],
    });
  }
  return result;
}

export class WebGpuHandTracker {
  constructor({
    manifest,
    backend = 'auto',
    maxHands = 1,
    detectorInterval = 8,
    scoreThreshold = 0.55,
  } = {}) {
    if (!manifest) throw new TypeError('A model manifest is required.');

    this.manifest = manifest;
    this.maxHands = maxHands;
    this.detectorInterval = detectorInterval;
    this.scoreThreshold = scoreThreshold;
    this.anchors = generatePalmAnchors();
    this.factory = new OrtSessionFactory({ backend });
    this.detector = null;
    this.landmark = null;
    this.tracks = [];
    this.frameIndex = 0;
    this.detectorCanvas = makeCanvas(PALM_SIZE, PALM_SIZE);
    this.landmarkCanvas = makeCanvas(LANDMARK_SIZE, LANDMARK_SIZE);
  }

  async load({ detectorSource = null, landmarkSource = null } = {}) {
    const detectorModel = detectorSource ?? this.manifest.models.palm.url;
    const landmarkModel = landmarkSource ?? this.manifest.models.landmark.url;
    const [detector, landmark] = await Promise.all([
      this.factory.create(detectorModel, 'palm detector'),
      this.factory.create(landmarkModel, 'hand landmark'),
    ]);

    this.detector = detector;
    this.landmark = landmark;
    return this.diagnostics();
  }

  reset() {
    this.tracks = [];
    this.frameIndex = 0;
  }

  setMaxHands(maxHands) {
    this.maxHands = Math.max(1, Math.min(2, Number(maxHands) || 1));
    this.tracks = this.tracks.slice(0, this.maxHands);
  }

  diagnostics() {
    return {
      backend: this.detector?.executionProvider === 'webgpu'
        && this.landmark?.executionProvider === 'webgpu'
        ? 'webgpu'
        : 'wasm',
      graphCapture: false,
      sessions: this.factory.sessionDiagnostics,
    };
  }

  async detect(source) {
    const ort = requireOrt();
    const prepared = prepareLetterboxInput(source, PALM_SIZE, this.detectorCanvas);
    const inputName = this.detector.session.inputNames[0];
    const input = new ort.Tensor('float32', prepared.data, [1, PALM_SIZE, PALM_SIZE, 3]);
    const outputs = await this.detector.session.run({ [inputName]: input });
    const identified = identifyPalmOutputs(outputs);

    return decodePalmDetections(identified.boxes.data, identified.scores.data, {
      anchors: this.anchors,
      ...prepared.transform,
      scoreThreshold: this.scoreThreshold,
      maxHands: this.maxHands,
    });
  }

  async landmarksForRoi(source, roi) {
    const ort = requireOrt();
    const prepared = prepareRoiInput(source, roi, LANDMARK_SIZE, this.landmarkCanvas);
    const inputName = this.landmark.session.inputNames[0];
    const input = new ort.Tensor('float32', prepared.data, [1, LANDMARK_SIZE, LANDMARK_SIZE, 3]);
    const outputs = await this.landmark.session.run({ [inputName]: input });
    const decoded = identifyLandmarkOutputs(outputs, this.landmark.session.outputNames);

    if (decoded.score < this.scoreThreshold) return null;

    return {
      score: decoded.score,
      handednessScore: decoded.handedness,
      handedness: decoded.handedness >= 0.5 ? 'right' : 'left',
      points: projectLandmarksFromRoi(decoded.landmarks.data, roi, LANDMARK_SIZE),
      worldPoints: cloneWorldLandmarks(decoded.worldLandmarks),
    };
  }

  async infer(source) {
    if (!this.detector || !this.landmark) throw new Error('Models are not loaded.');

    const { width, height } = sourceDimensions(source);
    const started = performance.now();
    const shouldDetect = this.tracks.length === 0
      || this.frameIndex % this.detectorInterval === 0;
    let detectorMilliseconds = 0;

    if (shouldDetect) {
      const detectorStarted = performance.now();
      const detections = await this.detect(source);
      detectorMilliseconds = performance.now() - detectorStarted;
      if (detections.length > 0) {
        this.tracks = detections.map((detection) => ({
          roi: computeRoiFromPalm(detection, width, height),
          detection,
        }));
      }
    }

    const hands = [];
    const nextTracks = [];
    const landmarkStarted = performance.now();
    for (const track of this.tracks.slice(0, this.maxHands)) {
      const hand = await this.landmarksForRoi(source, track.roi);
      if (!hand) continue;
      const roi = computeRoiFromLandmarks(hand.points, width, height);
      hands.push({ ...hand, roi });
      nextTracks.push({ roi });
    }

    const landmarkMilliseconds = performance.now() - landmarkStarted;
    this.tracks = nextTracks;
    this.frameIndex += 1;

    return {
      hands,
      width,
      height,
      backend: this.diagnostics().backend,
      graphCapture: false,
      timing: {
        totalMilliseconds: performance.now() - started,
        detectorMilliseconds,
        landmarkMilliseconds,
      },
    };
  }
}

export async function loadModelManifest(url = './models/manifest.json') {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Model manifest request failed with HTTP ${response.status}.`);
  }
  return response.json();
}
