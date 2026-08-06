import test from 'node:test';
import assert from 'node:assert/strict';

import {
  LANDMARK_CHILDREN,
  MANO_TO_MEDIAPIPE,
  LandmarkInterpolator,
  ManoHandDeformer,
} from '../src/mano-deformer.js';

const restLandmarks = new Float32Array([
  0, 0, 0,
  0.30, 0.10, 0, 0.45, 0.15, 0, 0.55, 0.20, 0, 0.65, 0.25, 0,
  0.20, 0.40, 0, 0.22, 0.65, 0, 0.23, 0.85, 0, 0.24, 1.00, 0,
  0.00, 0.45, 0, 0.00, 0.72, 0, 0.00, 0.95, 0, 0.00, 1.15, 0,
  -0.18, 0.40, 0, -0.20, 0.65, 0, -0.21, 0.85, 0, -0.22, 1.00, 0,
  -0.32, 0.32, 0, -0.38, 0.52, 0, -0.42, 0.68, 0, -0.45, 0.80, 0,
]);

const restSkinJoints = new Float32Array(16 * 3);
for (let joint = 0; joint < 16; joint += 1) {
  const landmark = MANO_TO_MEDIAPIPE[joint];
  restSkinJoints.set(restLandmarks.slice(landmark * 3, landmark * 3 + 3), joint * 3);
}

function makeHand() {
  const vertices = new Float32Array([
    0, 0, 0,
    0.20, 0.40, 0,
    0, 0.45, 0,
    0.30, 0.10, 0,
  ]);
  const influenceJoints = new Uint8Array([
    0, 0, 0, 0,
    1, 0, 0, 0,
    4, 0, 0, 0,
    13, 0, 0, 0,
  ]);
  const influenceWeights = new Float32Array([
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
  ]);
  return { vertices, restSkinJoints, restLandmarks, influenceJoints, influenceWeights };
}

function shifted(points, x, y, z) {
  const output = new Float32Array(points.length);
  for (let index = 0; index < points.length; index += 3) {
    output[index] = points[index] + x;
    output[index + 1] = points[index + 1] + y;
    output[index + 2] = points[index + 2] + z;
  }
  return output;
}

function closeArray(actual, expected, epsilon = 1e-4) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(Math.abs(actual[index] - expected[index]) < epsilon, `${index}: ${actual[index]} vs ${expected[index]}`);
  }
}

test('maps 16 MANO skin joints onto MediaPipe landmarks and tips', () => {
  assert.deepEqual(Array.from(MANO_TO_MEDIAPIPE), [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]);
  assert.deepEqual(Array.from(LANDMARK_CHILDREN), [9, 6, 7, 8, 10, 11, 12, 18, 19, 20, 14, 15, 16, 2, 3, 4]);
});

test('rest landmarks preserve the neutral mesh', () => {
  const hand = makeHand();
  const deformer = new ManoHandDeformer(hand);
  closeArray(deformer.deform(restLandmarks), hand.vertices);
});

test('a rigid landmark translation translates the skinned mesh', () => {
  const hand = makeHand();
  const deformer = new ManoHandDeformer(hand);
  const output = deformer.deform(shifted(restLandmarks, 1, -2, 0.5));
  closeArray(output, shifted(hand.vertices, 1, -2, 0.5));
});

test('landmark interpolator converges by elapsed time rather than inference count', () => {
  const interpolator = new LandmarkInterpolator({ halfLifeMs: 24 });
  interpolator.setTarget(restLandmarks, 0);
  closeArray(interpolator.sample(0), restLandmarks);
  const target = shifted(restLandmarks, 1, 0, 0);
  interpolator.setTarget(target, 8);
  const early = interpolator.sample(16);
  assert.ok(early[0] > 0 && early[0] < 1);
  const late = interpolator.sample(160);
  assert.ok(late[0] > 0.98);
});
