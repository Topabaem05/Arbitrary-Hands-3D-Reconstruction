import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BONE_NAMES,
  HandLandmarkSmoother,
  quatRotateVector,
  solveRigPose,
} from '../src/hand-retarget.js';

function makeRest() {
  const points = Array.from({ length: 21 }, () => [0, 0, 0]);
  points[0] = [0, 0, 0];
  points[1] = [0.4, -0.4, 0];
  points[2] = [0.8, -0.7, 0];
  points[3] = [1.2, -0.9, 0];
  points[4] = [1.6, -1.0, 0];
  points[5] = [1, 0.7, 0];
  points[6] = [2, 0.7, 0];
  points[7] = [3, 0.7, 0];
  points[8] = [4, 0.7, 0];
  points[9] = [1, 0.2, 0];
  points[10] = [2.2, 0.2, 0];
  points[11] = [3.4, 0.2, 0];
  points[12] = [4.6, 0.2, 0];
  points[13] = [1, -0.3, 0];
  points[14] = [2.1, -0.3, 0];
  points[15] = [3.2, -0.3, 0];
  points[16] = [4.3, -0.3, 0];
  points[17] = [0.9, -0.8, 0];
  points[18] = [1.9, -0.8, 0];
  points[19] = [2.9, -0.8, 0];
  points[20] = [3.9, -0.8, 0];
  return points.flat();
}

function near(actual, expected, epsilon = 1e-5) {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} not near ${expected}`);
}

test('bone contract matches the 21 MediaPipe landmark names', () => {
  assert.equal(BONE_NAMES.length, 21);
  assert.equal(BONE_NAMES[0], 'wrist');
  assert.equal(BONE_NAMES[8], 'index_tip');
  assert.equal(BONE_NAMES[20], 'pinky_tip');
});

test('rest landmarks produce identity root and local rotations', () => {
  const rest = makeRest();
  const pose = solveRigPose(rest, rest, 'left');
  for (const value of pose.rootQuaternion.slice(0, 3)) near(value, 0);
  near(pose.rootQuaternion[3], 1);
  for (let bone = 0; bone < 21; bone += 1) {
    const offset = bone * 4;
    for (let axis = 0; axis < 3; axis += 1) {
      near(pose.boneQuaternions[offset + axis], 0);
    }
    near(pose.boneQuaternions[offset + 3], 1);
  }
});

test('index MCP rotates its rest child direction toward the target child', () => {
  const rest = makeRest();
  const target = rest.slice();
  target[6 * 3] = 1;
  target[6 * 3 + 1] = 1.7;
  target[7 * 3] = 1;
  target[7 * 3 + 1] = 2.7;
  target[8 * 3] = 1;
  target[8 * 3 + 1] = 3.7;
  const pose = solveRigPose(rest, target, 'left');
  const quaternion = Array.from(pose.boneQuaternions.slice(5 * 4, 5 * 4 + 4));
  const rotated = quatRotateVector(quaternion, [1, 0, 0]);
  near(rotated[0], 0, 1e-4);
  near(rotated[1], 1, 1e-4);
});

test('right-hand canonicalization mirrors the palm across axis', () => {
  const rest = makeRest();
  const mirrored = rest.slice();
  for (let index = 0; index < 21; index += 1) mirrored[index * 3 + 1] *= -1;
  const pose = solveRigPose(rest, mirrored, 'right');
  assert.equal(pose.mirrorY, true);
  const quaternion = Array.from(pose.boneQuaternions.slice(5 * 4, 5 * 4 + 4));
  near(quaternion[0], 0);
  near(quaternion[1], 0);
  near(quaternion[2], 0);
  near(quaternion[3], 1);
});

test('landmark smoother converges by elapsed time', () => {
  const smoother = new HandLandmarkSmoother({ halfLifeMs: 100 });
  const a = new Float32Array(63);
  const b = new Float32Array(63).fill(10);
  smoother.setTarget(a, 0);
  smoother.sample(0);
  smoother.setTarget(b, 0);
  const halfway = smoother.sample(100);
  near(halfway[0], 5);
});
