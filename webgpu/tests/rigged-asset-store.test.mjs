import test from 'node:test';
import assert from 'node:assert/strict';

import {
  RiggedAssetStore,
  createMemoryRiggedAssetStore,
  validateGlbHeader,
} from '../src/rigged-asset-store.js';

function glbBytes() {
  const json = new TextEncoder().encode('{"asset":{"version":"2.0"}}   ');
  const total = 12 + 8 + json.length;
  const buffer = new ArrayBuffer(total);
  const view = new DataView(buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, json.length, true);
  view.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(buffer, 20).set(json);
  return buffer;
}

test('validates the GLB 2.0 header and declared byte length', () => {
  const bytes = glbBytes();
  const metadata = validateGlbHeader(bytes);
  assert.equal(metadata.version, 2);
  assert.equal(metadata.byteLength, bytes.byteLength);
  const invalid = bytes.slice(0);
  new DataView(invalid).setUint32(8, bytes.byteLength + 4, true);
  assert.throws(() => validateGlbHeader(invalid), /declared length/i);
});

test('memory asset store clones bytes on save and load', async () => {
  const store = createMemoryRiggedAssetStore({ key: 'lod0' });
  const bytes = glbBytes();
  await store.save({ name: 'hand.glb', bytes, importedAt: 1 });
  new Uint8Array(bytes)[20] = 0;
  const loaded = await store.load();
  assert.equal(loaded.name, 'hand.glb');
  assert.equal(new Uint8Array(loaded.bytes)[20], 123);
  await store.clear();
  assert.equal(await store.load(), null);
});

test('store rejects non-GLB assets', async () => {
  const store = new RiggedAssetStore({
    adapter: { async get() { return null; }, async set() {}, async delete() {} },
  });
  await assert.rejects(
    () => store.save({ name: 'bad.glb', bytes: new ArrayBuffer(24), importedAt: 1 }),
    /GLB magic/i,
  );
});
