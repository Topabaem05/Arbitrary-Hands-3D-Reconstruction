import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const read = (name) => readFile(new URL(`../${name}`, import.meta.url), 'utf8');

test('first-run flow imports and persists a private MANO bundle before camera start', async () => {
  const [html, app] = await Promise.all([read('index.html'), read('app.js')]);
  assert.match(html, /id="manoBundleInput"[^>]*type="file"[^>]*hidden/);
  assert.match(app, /ManoBundleStore/);
  assert.match(app, /loadManoBundleFile/);
  assert.match(app, /bundleStore\.save/);
  assert.match(app, /meshRenderer\.setBundle/);
  assert.match(app, /manoBundleInput/);
  assert.match(app, /Shift/);
  assert.match(app, /Delete/);
  assert.match(app, /bundleStore\.clear/);
});

test('mesh rendering owns an independent requestAnimationFrame loop and one-hand M1 default', async () => {
  const [app, camera] = await Promise.all([read('app.js'), read('src/camera.js')]);
  assert.match(app, /requestAnimationFrame\(renderLoop\)/);
  assert.match(app, /new HandMeshRenderer/);
  assert.match(app, /detectorInterval:\s*12/);
  assert.match(app, /get\(['"]hands['"]\)\s*===\s*['"]2['"]/);
  assert.match(camera, /VideoFrame/);
  assert.match(camera, /createImageBitmap/);
  assert.doesNotMatch(app, /Skeleton3DRenderer/);
});

test('the visual surface is pure black with only mesh canvas and lower-right camera PIP', async () => {
  const [html, css, renderer] = await Promise.all([read('index.html'), read('styles.css'), read('src/renderer.js')]);
  assert.match(css, /background:\s*#000(?:000)?/);
  assert.match(css, /\.cam-pip[\s\S]*position:\s*fixed[\s\S]*right:[\s\S]*bottom:/);
  assert.match(html, /id="meshCanvas"/);
  assert.doesNotMatch(html, /id="skeletonCanvas"/);
  assert.doesNotMatch(renderer, /class\s+Skeleton3DRenderer/);
});
