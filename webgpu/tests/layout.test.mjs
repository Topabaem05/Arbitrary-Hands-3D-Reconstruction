import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const read = (name) => readFile(new URL(`../${name}`, import.meta.url), 'utf8');

test('browser surface is a black fullscreen rigged hand with lower-right camera PIP', async () => {
  const [html, css, app] = await Promise.all([read('index.html'), read('styles.css'), read('app.js')]);
  assert.match(html, /id="meshCanvas"/);
  assert.match(html, /id="riggedHandInput"/);
  assert.match(html, /class="cam-pip"/);
  assert.doesNotMatch(html, /manoBundleInput|skeletonCanvas/);
  assert.match(css, /background:\s*#000000/);
  assert.match(css, /\.cam-pip\s*\{[^}]*position:\s*fixed[^}]*right:[^}]*bottom:/s);
  assert.match(app, /RiggedHandRenderer/);
  assert.match(html, /name="acr-build"/);
  assert.match(html, /rigged-v3/);
  assert.doesNotMatch(app, /Skeleton3DRenderer|HandMeshRenderer|ManoBundleStore/);
});
