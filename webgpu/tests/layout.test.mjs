import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const readWebGpuFile = (name) => readFile(new URL(`../${name}`, import.meta.url), 'utf8');

test('browser surface is a black fullscreen MANO mesh with camera picture-in-picture', async () => {
  const [html, css, app] = await Promise.all([
    readWebGpuFile('index.html'),
    readWebGpuFile('styles.css'),
    readWebGpuFile('app.js'),
  ]);
  assert.match(html, /id="meshCanvas"/);
  assert.match(html, /class="cam-pip"/);
  assert.match(html, /id="cameraVideo"/);
  assert.match(html, /id="overlayCanvas"/);
  assert.match(html, /id="manoBundleInput"/);
  assert.doesNotMatch(html, /skeletonCanvas/);
  for (const removedSurface of ['topbar', 'control-panel', 'metric-strip', 'stage-head', 'viewport-grid']) {
    assert.doesNotMatch(html, new RegExp(`class="[^"]*${removedSurface}`));
  }
  assert.match(css, /background:\s*#000000/);
  assert.match(css, /\.cam-pip\s*\{[^}]*position:\s*fixed[^}]*right:[^}]*bottom:/s);
  assert.match(css, /#meshCanvas\s*\{[^}]*width:\s*100%[^}]*height:\s*100%/s);
  assert.match(app, /HandMeshRenderer/);
  assert.doesNotMatch(app, /Skeleton3DRenderer/);
});
