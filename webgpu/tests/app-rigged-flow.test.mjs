import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const read = (name) => readFile(new URL(`../${name}`, import.meta.url), 'utf8');

test('first-run flow imports and persists a private rigged GLB before camera start', async () => {
  const [html, app] = await Promise.all([read('index.html'), read('app.js')]);
  assert.match(html, /id="riggedHandInput"[^>]*accept="\.glb/);
  assert.match(app, /RiggedAssetStore/);
  assert.match(app, /readGlbFile/);
  assert.match(app, /assetStore\.save/);
  assert.match(app, /rigRenderer\.load/);
  assert.doesNotMatch(app, /ManoBundleStore|loadManoBundleFile|HandMeshRenderer/);
});

test('rendering uses an independent animation loop and query-gated M1 controls', async () => {
  const app = await read('app.js');
  assert.match(app, /requestAnimationFrame\(renderLoop\)/);
  assert.match(app, /new RiggedHandRenderer/);
  assert.match(app, /detectorInterval:\s*12/);
  assert.match(app, /get\(['"]hands['"]\)\s*===\s*['"]2['"]/);
  assert.match(app, /get\(['"]lod['"]\)\s*===\s*['"]1['"]/);
  assert.match(app, /get\(['"]debugBones['"]\)\s*===\s*['"]1['"]/);
  assert.match(app, /Shift/);
  assert.match(app, /Delete/);
});

test('page pins Three.js imports and keeps the private asset out of deployment', async () => {
  const html = await read('index.html');
  assert.match(html, /type="importmap"/);
  assert.match(html, /three@0\.185\.1\/build\/three\.module\.js/);
  assert.match(html, /three@0\.185\.1\/examples\/jsm\//);
  assert.doesNotMatch(html, /\.glb[^<]*src=/);
});

test('app previews the loaded rig and hides the rest pose only while live tracking starts', async () => {
  const app = await read('app.js');
  assert.match(app, /rigRenderer\.showRestPose\(\)/);
  assert.match(app, /rigRenderer\.reset\(\{\s*showRestPose:\s*false\s*\}\)/);
  assert.match(app, /ACR_RIG_BUILD/);
});
