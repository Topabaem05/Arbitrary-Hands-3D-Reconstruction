import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { computeVertexNormals } from '../src/mesh-renderer.js';

test('computes reusable smooth normals for an indexed triangle', () => {
  const vertices = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const faces = new Uint16Array([0, 1, 2]);
  const target = new Float32Array(9);
  assert.equal(computeVertexNormals(vertices, faces, target), target);
  for (let vertex = 0; vertex < 3; vertex += 1) {
    assert.ok(Math.abs(target[vertex * 3]) < 1e-6);
    assert.ok(Math.abs(target[vertex * 3 + 1]) < 1e-6);
    assert.ok(Math.abs(target[vertex * 3 + 2] - 1) < 1e-6);
  }
});

test('renderer uses a persistent WebGL2 indexed mesh pipeline on a black clear color', async () => {
  const source = await readFile(new URL('../src/mesh-renderer.js', import.meta.url), 'utf8');
  assert.match(source, /getContext\(['"]webgl2['"]/);
  assert.match(source, /powerPreference:\s*['"]high-performance['"]/);
  assert.match(source, /clearColor\(0,\s*0,\s*0,\s*1\)/);
  assert.match(source, /enable\(gl\.DEPTH_TEST\)/);
  assert.match(source, /bufferSubData\(/);
  assert.match(source, /drawElements\(gl\.TRIANGLES/);
  assert.doesNotMatch(source, /createBuffer\([^)]*\)[\s\S]*requestAnimationFrame/);
});
