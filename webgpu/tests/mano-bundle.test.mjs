import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MANO_BUNDLE_FORMAT,
  createMemoryBundleStore,
  validateManoBundle,
} from '../src/mano-bundle.js';

function hand(side) {
  const vertices = Array.from({ length: 778 }, (_, index) => [index / 778, (index % 31) / 31, 0]);
  const faces = [[0, 1, 2], [775, 776, 777]];
  const weights = Array.from({ length: 778 }, (_, index) => {
    const row = Array(16).fill(0);
    row[index % 16] = 0.75;
    row[(index + 1) % 16] = 0.25;
    return row;
  });
  const restSkinJoints = Array.from({ length: 16 }, (_, index) => [index / 16, index / 32, 0]);
  const restLandmarks = Array.from({ length: 21 }, (_, index) => [index / 21, index / 42, 0]);
  return { side, vertices, faces, weights, restSkinJoints, restLandmarks };
}

function bundle() {
  return {
    format: MANO_BUNDLE_FORMAT,
    private: true,
    redistributable: false,
    vertexCount: 778,
    faceCount: 2,
    skinJointCount: 16,
    landmarkCount: 21,
    jointParents: [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14],
    hands: { left: hand('left'), right: hand('right') },
  };
}

test('validates and packs a private MANO browser bundle', () => {
  const parsed = validateManoBundle(bundle());
  assert.equal(parsed.format, MANO_BUNDLE_FORMAT);
  assert.ok(parsed.hands.left.vertices instanceof Float32Array);
  assert.ok(parsed.hands.left.faces instanceof Uint16Array);
  assert.ok(parsed.hands.left.influenceJoints instanceof Uint8Array);
  assert.ok(parsed.hands.left.influenceWeights instanceof Float32Array);
  assert.equal(parsed.hands.left.vertices.length, 778 * 3);
  assert.equal(parsed.hands.left.influenceWeights.length, 778 * 4);
  for (let vertex = 0; vertex < 16; vertex += 1) {
    const base = vertex * 4;
    const total = parsed.hands.left.influenceWeights.slice(base, base + 4).reduce((sum, value) => sum + value, 0);
    assert.ok(Math.abs(total - 1) < 1e-6);
  }
});

test('rejects invalid format, face bounds, and weight normalization', () => {
  const invalidVersion = bundle();
  invalidVersion.format = 'wrong';
  assert.throws(() => validateManoBundle(invalidVersion), /format/);

  const invalidFace = bundle();
  invalidFace.hands.left.faces[0][0] = 778;
  assert.throws(() => validateManoBundle(invalidFace), /face/i);

  const invalidWeight = bundle();
  invalidWeight.hands.left.weights[0] = Array(16).fill(0);
  assert.throws(() => validateManoBundle(invalidWeight), /weight/i);
});

test('memory bundle store saves, loads, and clears validated data', async () => {
  const store = createMemoryBundleStore();
  assert.equal(await store.load(), null);
  const saved = await store.save(bundle());
  assert.equal(saved.format, MANO_BUNDLE_FORMAT);
  const loaded = await store.load();
  assert.equal(loaded.hands.right.side, 'right');
  await store.clear();
  assert.equal(await store.load(), null);
});
