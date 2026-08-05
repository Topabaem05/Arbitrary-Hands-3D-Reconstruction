import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const read = (name) => readFile(new URL(`../${name}`, import.meta.url), 'utf8');

test('renderer loads a SkinnedMesh through GLTFLoader and validates all bones', async () => {
  const source = await read('src/rigged-hand-renderer.js');
  assert.match(source, /from ['"]three['"]/);
  assert.match(source, /GLTFLoader/);
  assert.match(source, /SkeletonUtils/);
  assert.match(source, /loader\.parse/);
  assert.match(source, /isSkinnedMesh/);
  assert.match(source, /BONE_NAMES/);
  assert.match(source, /Missing required hand bones/);
});

test('renderer is persistent, black, GPU-skinned, and debug bones are optional', async () => {
  const source = await read('src/rigged-hand-renderer.js');
  assert.match(source, /WebGLRenderer/);
  assert.match(source, /setClearColor\(0x000000/);
  assert.match(source, /SkeletonHelper/);
  assert.match(source, /debugBones/);
  assert.doesNotMatch(source, /bufferSubData/);
  assert.doesNotMatch(source, /computeVertexNormals\(\)/);
});
