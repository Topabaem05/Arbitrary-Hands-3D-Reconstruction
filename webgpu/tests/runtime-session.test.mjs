import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { buildExecutionAttempts, resolveDefaultManifestUrl } from '../src/runtime.js';

const readWebGpuFile = (name) => readFile(new URL(`../${name}`, import.meta.url), 'utf8');

test('ordinary WebGPU is attempted without graph capture before WASM fallback', () => {
  const attempts = buildExecutionAttempts({ backend: 'auto', hasWebGpu: true });

  assert.deepEqual(attempts, [
    { executionProviders: ['webgpu'], graphCapture: false },
    { executionProviders: ['wasm'], graphCapture: false },
  ]);
  assert.equal(attempts.some((attempt) => attempt.enableGraphCapture === true), false);
});

test('required WebGPU fails early when navigator.gpu is unavailable', () => {
  assert.throws(
    () => buildExecutionAttempts({ backend: 'webgpu', hasWebGpu: false }),
    /does not expose navigator\.gpu/,
  );
});

test('the default manifest URL follows the runtime module instead of the page origin', () => {
  const runtimeUrl = 'https://cdn.jsdelivr.net/gh/Topabaem05/Arbitrary-Hands-3D-Reconstruction@deadbeef/webgpu/src/runtime.js';

  assert.equal(
    resolveDefaultManifestUrl(runtimeUrl).href,
    'https://cdn.jsdelivr.net/gh/Topabaem05/Arbitrary-Hands-3D-Reconstruction@deadbeef/webgpu/models/manifest.json',
  );
});

test('the page declares an inline favicon and suppresses non-actionable ORT warnings', async () => {
  const [html, runtime] = await Promise.all([
    readWebGpuFile('index.html'),
    readWebGpuFile('src/runtime.js'),
  ]);

  assert.match(html, /<link\s+rel="icon"\s+href="data:,">/);
  assert.match(runtime, /ort\.env\.logLevel\s*=\s*['"]error['"]/);
});
