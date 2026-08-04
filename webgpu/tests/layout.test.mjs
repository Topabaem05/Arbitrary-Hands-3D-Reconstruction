import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const readWebGpuFile = (name) => readFile(new URL(`../${name}`, import.meta.url), 'utf8');

test('browser surface is reduced to fullscreen hand model and camera picture-in-picture', async () => {
  const [html, css, app] = await Promise.all([
    readWebGpuFile('index.html'),
    readWebGpuFile('styles.css'),
    readWebGpuFile('app.js'),
  ]);

  assert.match(html, /id="skeletonCanvas"/);
  assert.match(html, /class="cam-pip"/);
  assert.match(html, /id="cameraVideo"/);
  assert.match(html, /id="overlayCanvas"/);
  assert.match(html, /id="startCamera"/);
  assert.match(html, /class="start-overlay"/);

  for (const removedSurface of ['topbar', 'control-panel', 'metric-strip', 'stage-head', 'viewport-grid']) {
    assert.doesNotMatch(html, new RegExp(`class="[^"]*${removedSurface}`));
  }

  assert.match(css, /\.hand-stage\s*\{[^}]*position:\s*fixed[^}]*inset:\s*0/s);
  assert.match(css, /\.cam-pip\s*\{[^}]*position:\s*fixed[^}]*right:/s);
  assert.match(css, /\.cam-pip\s*\{[^}]*bottom:/s);
  assert.match(css, /#skeletonCanvas\s*\{[^}]*width:\s*100%[^}]*height:\s*100%/s);

  for (const removedSelector of ['#backendSelect', '#maxHands', '#diagnostics', '#inferenceValue', '#displayValue']) {
    assert.doesNotMatch(app, new RegExp(removedSelector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  }
});
