import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeRoiFromLandmarks,
  computeRoiFromPalm,
  projectLandmarksFromRoi,
} from '../src/geometry.js';


test('computes an enlarged rotated ROI from palm keypoints', () => {
  const detection = {
    box: { x1: 100, y1: 120, x2: 180, y2: 200 },
    keypoints: [
      { x: 140, y: 190 },
      { x: 120, y: 150 },
      { x: 140, y: 140 },
      { x: 150, y: 145 },
      { x: 160, y: 150 },
      { x: 130, y: 165 },
      { x: 150, y: 165 },
    ],
  };
  const roi = computeRoiFromPalm(detection, 320, 240);
  assert.ok(roi.size > 150);
  assert.ok(roi.cy < 180);
  assert.ok(Number.isFinite(roi.rotation));
});


test('projects landmark coordinates from crop space back to source space', () => {
  const roi = { cx: 100, cy: 80, size: 224, rotation: 0 };
  const points = projectLandmarksFromRoi(
    new Float32Array([112, 112, 0, 224, 112, 5]),
    roi,
    224,
  );
  assert.deepEqual(points[0], { x: 100, y: 80, z: 0 });
  assert.deepEqual(points[1], { x: 212, y: 80, z: 5 });
});


test('updates tracking ROI from 21 projected landmarks', () => {
  const points = Array.from({ length: 21 }, (_, index) => ({
    x: 80 + index * 2,
    y: 120 - index * 2,
    z: index,
  }));
  points[0] = { x: 100, y: 180, z: 0 };
  points[9] = { x: 100, y: 110, z: 0 };
  const roi = computeRoiFromLandmarks(points, 320, 240);
  assert.ok(roi.size > 50);
  assert.ok(roi.cx >= 0 && roi.cx <= 320);
  assert.ok(roi.cy >= 0 && roi.cy <= 240);
});
