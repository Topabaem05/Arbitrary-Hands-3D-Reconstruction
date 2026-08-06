const DEFAULT_DB_NAME = 'acr-hand-lab';
const DEFAULT_STORE_NAME = 'private-assets';
const DEFAULT_KEY = 'rigged-hand-glb-v2';
const GLB_MAGIC = 0x46546c67;
const GLB_JSON_CHUNK = 0x4e4f534a;
const MAX_GLB_BYTES = 32 * 1024 * 1024;

function asArrayBuffer(value) {
  if (value instanceof ArrayBuffer) return value;
  if (ArrayBuffer.isView(value)) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw new TypeError('Rigged hand bytes must be an ArrayBuffer or typed array.');
}

function cloneBuffer(value) {
  return asArrayBuffer(value).slice(0);
}

export function validateGlbHeader(value) {
  const buffer = asArrayBuffer(value);
  if (buffer.byteLength < 20) throw new RangeError('GLB is truncated.');
  if (buffer.byteLength > MAX_GLB_BYTES) throw new RangeError('GLB is unexpectedly large.');
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== GLB_MAGIC) throw new TypeError('GLB magic is invalid.');
  const version = view.getUint32(4, true);
  if (version !== 2) throw new RangeError('Only GLB version 2 is supported.');
  const declaredLength = view.getUint32(8, true);
  if (declaredLength !== buffer.byteLength) {
    throw new RangeError('GLB declared length does not match the file.');
  }
  const jsonLength = view.getUint32(12, true);
  if (view.getUint32(16, true) !== GLB_JSON_CHUNK) {
    throw new TypeError('GLB first chunk must be JSON.');
  }
  if (jsonLength <= 0 || 20 + jsonLength > buffer.byteLength) {
    throw new RangeError('GLB JSON chunk length is invalid.');
  }
  return { version, byteLength: buffer.byteLength, jsonLength };
}

export async function readGlbFile(file) {
  if (!file || typeof file.arrayBuffer !== 'function') {
    throw new TypeError('Select a rigged hand .glb file.');
  }
  if (file.size > MAX_GLB_BYTES) throw new RangeError('Rigged hand GLB is unexpectedly large.');
  const bytes = await file.arrayBuffer();
  validateGlbHeader(bytes);
  return {
    name: String(file.name || 'rigged-hand.glb'),
    bytes: cloneBuffer(bytes),
    importedAt: Date.now(),
    source: 'local',
  };
}

function assetNameFromUrl(url) {
  try {
    const pathname = new URL(url, 'https://local.invalid/').pathname;
    return decodeURIComponent(pathname.split('/').filter(Boolean).at(-1) || 'rigged-hand.glb');
  } catch {
    return 'rigged-hand.glb';
  }
}

export function resolveBundledAssetUrl({ key = DEFAULT_KEY, origin = globalThis.location?.origin } = {}) {
  if (!origin) return null;
  const lod = /lod1(?:$|:)/.test(String(key)) ? 1 : 0;
  return new URL(`/assets/hand_rigged_v3_lod${lod}.glb`, origin).href;
}

export async function fetchRiggedAsset(
  url,
  { fetchImpl = globalThis.fetch, cache = 'force-cache' } = {},
) {
  if (typeof fetchImpl !== 'function') throw new TypeError('A fetch implementation is required.');
  const response = await fetchImpl(url, { cache });
  if (!response?.ok) {
    throw new Error(`Bundled rigged hand request failed with HTTP ${response?.status ?? 'unknown'}.`);
  }
  const bytes = await response.arrayBuffer();
  validateGlbHeader(bytes);
  return {
    name: assetNameFromUrl(url),
    bytes: cloneBuffer(bytes),
    importedAt: Date.now(),
    source: 'bundled',
    url: String(url),
  };
}

function validateAsset(value, source = 'local') {
  if (!value || typeof value !== 'object') {
    throw new TypeError('Rigged hand asset must be an object.');
  }
  const bytes = cloneBuffer(value.bytes);
  validateGlbHeader(bytes);
  const name = String(value.name || 'rigged-hand.glb');
  const importedAt = Number(value.importedAt);
  return {
    name,
    bytes,
    importedAt: Number.isFinite(importedAt) ? importedAt : Date.now(),
    source: value.source === 'bundled' ? 'bundled' : source,
    ...(value.url ? { url: String(value.url) } : {}),
  };
}

function createMemoryAdapter() {
  const values = new Map();
  return {
    async get(key) {
      const value = values.get(key);
      return value ? { ...value, bytes: cloneBuffer(value.bytes) } : null;
    },
    async set(key, value) {
      values.set(key, { ...value, bytes: cloneBuffer(value.bytes) });
    },
    async delete(key) { values.delete(key); },
  };
}

function openDatabase(dbName, storeName) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, 1);
    request.addEventListener('upgradeneeded', () => {
      if (!request.result.objectStoreNames.contains(storeName)) {
        request.result.createObjectStore(storeName);
      }
    });
    request.addEventListener('success', () => resolve(request.result), { once: true });
    request.addEventListener(
      'error',
      () => reject(request.error ?? new Error('IndexedDB open failed.')),
      { once: true },
    );
  });
}

function createIndexedDbAdapter(dbName = DEFAULT_DB_NAME, storeName = DEFAULT_STORE_NAME) {
  return {
    async get(key) {
      const db = await openDatabase(dbName, storeName);
      try {
        return await new Promise((resolve, reject) => {
          const transaction = db.transaction(storeName, 'readonly');
          const request = transaction.objectStore(storeName).get(key);
          request.addEventListener('success', () => resolve(request.result ?? null), { once: true });
          request.addEventListener('error', () => reject(request.error), { once: true });
        });
      } finally {
        db.close();
      }
    },
    async set(key, value) {
      const db = await openDatabase(dbName, storeName);
      try {
        await new Promise((resolve, reject) => {
          const transaction = db.transaction(storeName, 'readwrite');
          transaction.objectStore(storeName).put(value, key);
          transaction.addEventListener('complete', resolve, { once: true });
          transaction.addEventListener('error', () => reject(transaction.error), { once: true });
        });
      } finally {
        db.close();
      }
    },
    async delete(key) {
      const db = await openDatabase(dbName, storeName);
      try {
        await new Promise((resolve, reject) => {
          const transaction = db.transaction(storeName, 'readwrite');
          transaction.objectStore(storeName).delete(key);
          transaction.addEventListener('complete', resolve, { once: true });
          transaction.addEventListener('error', () => reject(transaction.error), { once: true });
        });
      } finally {
        db.close();
      }
    },
  };
}

export class RiggedAssetStore {
  constructor({
    adapter = null,
    key = DEFAULT_KEY,
    fallbackUrl = null,
    fetchImpl = globalThis.fetch,
  } = {}) {
    this.adapter = adapter
      ?? (typeof indexedDB === 'undefined' ? createMemoryAdapter() : createIndexedDbAdapter());
    this.key = key;
    this.fallbackUrl = fallbackUrl
      ? String(fallbackUrl)
      : resolveBundledAssetUrl({ key });
    this.fetchImpl = fetchImpl;
  }

  async load() {
    const value = await this.adapter.get(this.key);
    if (value != null) return validateAsset(value, 'local');
    if (!this.fallbackUrl) return null;
    return fetchRiggedAsset(this.fallbackUrl, { fetchImpl: this.fetchImpl });
  }

  async save(value) {
    const asset = validateAsset({ ...value, source: 'local' }, 'local');
    await this.adapter.set(this.key, asset);
    return { ...asset, bytes: cloneBuffer(asset.bytes) };
  }

  async clear() {
    await this.adapter.delete(this.key);
  }
}

export function createMemoryRiggedAssetStore({
  key = DEFAULT_KEY,
  fallbackUrl = null,
  fetchImpl = globalThis.fetch,
} = {}) {
  return new RiggedAssetStore({
    adapter: createMemoryAdapter(),
    key,
    fallbackUrl,
    fetchImpl,
  });
}
