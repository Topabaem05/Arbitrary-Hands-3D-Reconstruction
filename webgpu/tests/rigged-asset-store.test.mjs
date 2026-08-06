import test from 'node:test';
import assert from 'node:assert/strict';

import {
  RiggedAssetStore,
  createMemoryRiggedAssetStore,
  fetchRiggedAsset,
  resolveBundledAssetUrl,
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

test('derives the public Vercel asset URL from the LOD storage key', () => {
  assert.equal(
    resolveBundledAssetUrl({ key: 'rigged-hand-glb-v1:lod0', origin: 'https://app.example' }),
    'https://app.example/assets/hand_rigged_v3_lod0.glb',
  );
  assert.equal(
    resolveBundledAssetUrl({ key: 'rigged-hand-glb-v1:lod1', origin: 'https://app.example' }),
    'https://app.example/assets/hand_rigged_v3_lod1.glb',
  );
});

test('loads the bundled GLB when no local override exists', async () => {
  const bytes = glbBytes();
  const requests = [];
  const store = createMemoryRiggedAssetStore({
    key: 'lod0',
    fallbackUrl: 'https://app.example/assets/hand_rigged_v3_lod0.glb',
    fetchImpl: async (url, options) => {
      requests.push({ url, options });
      return { ok: true, status: 200, async arrayBuffer() { return bytes; } };
    },
  });
  const loaded = await store.load();
  assert.equal(loaded.source, 'bundled');
  assert.equal(loaded.name, 'hand_rigged_v3_lod0.glb');
  assert.equal(requests.length, 1);
  assert.equal(requests[0].options.cache, 'force-cache');
});

test('a saved local GLB overrides the bundled default', async () => {
  const bytes = glbBytes();
  let fetchCount = 0;
  const store = createMemoryRiggedAssetStore({
    key: 'lod0',
    fallbackUrl: '/assets/default.glb',
    fetchImpl: async () => {
      fetchCount += 1;
      return { ok: true, status: 200, async arrayBuffer() { return bytes; } };
    },
  });
  await store.save({ name: 'custom.glb', bytes, importedAt: 1 });
  const loaded = await store.load();
  assert.equal(loaded.source, 'local');
  assert.equal(loaded.name, 'custom.glb');
  assert.equal(fetchCount, 0);
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

test('bundled asset fetch rejects HTTP and invalid GLB responses', async () => {
  await assert.rejects(
    () => fetchRiggedAsset('/missing.glb', {
      fetchImpl: async () => ({ ok: false, status: 404 }),
    }),
    /HTTP 404/,
  );
  await assert.rejects(
    () => fetchRiggedAsset('/bad.glb', {
      fetchImpl: async () => ({
        ok: true,
        status: 200,
        async arrayBuffer() { return new ArrayBuffer(24); },
      }),
    }),
    /GLB magic/i,
  );
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
