import test from 'node:test';
import assert from 'node:assert/strict';
import {
  decodePalmDetections,
  generatePalmAnchors,
  nonMaxSuppression,
} from '../src/palm.js';


test('generates the 2,016 anchors expected by the 192px MediaPipe palm model', () => {
  const anchors = generatePalmAnchors();
  assert.equal(anchors.length, 2016);
  assert.deepEqual(anchors[0], [1 / 48, 1 / 48]);
  assert.deepEqual(anchors[1], [1 / 48, 1 / 48]);
  assert.deepEqual(anchors[1152], [1 / 24, 1 / 24]);
});


test('decodes detector deltas through letterbox coordinates', () => {
  const anchors = generatePalmAnchors();
  const boxes = new Float32Array(2016 * 18);
  const scores = new Float32Array(2016).fill(-100);
  const index = 0;
  const offset = index * 18;
  boxes[offset] = 0;
  boxes[offset + 1] = 0;
  boxes[offset + 2] = 48;
  boxes[offset + 3] = 48;
  for (let k = 0; k < 7; k += 1) {
    boxes[offset + 4 + k * 2] = 0;
    boxes[offset + 5 + k * 2] = -12 * k;
  }
  scores[index] = 10;

  const detections = decodePalmDetections(boxes, scores, {
    anchors,
    inputSize: 192,
    sourceWidth: 640,
    sourceHeight: 480,
    scale: 0.3,
    padX: 0,
    padY: 24,
    scoreThreshold: 0.5,
    maxHands: 1,
  });

  assert.equal(detections.length, 1);
  const detection = detections[0];
  assert.ok(detection.score > 0.99);
  assert.ok(Number.isFinite(detection.box.x1));
  assert.equal(detection.keypoints.length, 7);
});


test('non maximum suppression keeps the strongest overlapping box', () => {
  const candidates = [
    { score: 0.9, box: { x1: 0, y1: 0, x2: 100, y2: 100 } },
    { score: 0.8, box: { x1: 10, y1: 10, x2: 90, y2: 90 } },
    { score: 0.7, box: { x1: 200, y1: 200, x2: 260, y2: 260 } },
  ];
  const kept = nonMaxSuppression(candidates, 0.3, 2);
  assert.deepEqual(kept.map((item) => item.score), [0.9, 0.7]);
});
