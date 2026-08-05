export const MANO_BUNDLE_FORMAT = 'mano-browser-bundle-v1';
export const MANO_VERTEX_COUNT = 778;
export const MANO_SKIN_JOINT_COUNT = 16;
export const MANO_LANDMARK_COUNT = 21;
export const MANO_INFLUENCE_COUNT = 4;

const DEFAULT_DB_NAME = 'acr-hand-lab';
const DEFAULT_STORE_NAME = 'private-assets';
const DEFAULT_KEY = 'mano-browser-bundle-v1';

function assertObject(value, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object.`);
  }
}

function flattenNumeric(value, rows, columns, name, TypedArray = Float32Array) {
  const expected = rows * columns;
  let flat;
  if (ArrayBuffer.isView(value)) {
    flat = Array.from(value);
  } else if (Array.isArray(value)) {
    if (value.length === expected && !Array.isArray(value[0])) {
      flat = value.slice();
    } else {
      if (value.length !== rows) {
        throw new RangeError(`${name} must have ${rows} rows.`);
      }
      flat = [];
      for (let rowIndex = 0; rowIndex < value.length; rowIndex += 1) {
        const row = value[rowIndex];
        if (!Array.isArray(row) && !ArrayBuffer.isView(row)) {
          throw new TypeError(`${name}[${rowIndex}] must be an array.`);
        }
        if (row.length !== columns) {
          throw new RangeError(`${name}[${rowIndex}] must have ${columns} values.`);
        }
        flat.push(...row);
      }
    }
  } else {
    throw new TypeError(`${name} must be an array or typed array.`);
  }
  if (flat.length !== expected) {
    throw new RangeError(`${name} must contain ${expected} values.`);
  }
  for (let index = 0; index < flat.length; index += 1) {
    if (!Number.isFinite(Number(flat[index]))) {
      throw new TypeError(`${name}[${index}] must be finite.`);
    }
  }
  return new TypedArray(flat);
}

function validateFaces(value, faceCount, name) {
  const faces = flattenNumeric(value, faceCount, 3, name, Uint16Array);
  for (let index = 0; index < faces.length; index += 1) {
    const vertex = faces[index];
    if (!Number.isInteger(vertex) || vertex < 0 || vertex >= MANO_VERTEX_COUNT) {
      throw new RangeError(`${name} contains an out-of-range face index.`);
    }
  }
  return faces;
}

function packInfluences(weights, name) {
  const joints = new Uint8Array(MANO_VERTEX_COUNT * MANO_INFLUENCE_COUNT);
  const packed = new Float32Array(MANO_VERTEX_COUNT * MANO_INFLUENCE_COUNT);
  for (let vertex = 0; vertex < MANO_VERTEX_COUNT; vertex += 1) {
    const rowOffset = vertex * MANO_SKIN_JOINT_COUNT;
    const ranked = [];
    let total = 0;
    for (let joint = 0; joint < MANO_SKIN_JOINT_COUNT; joint += 1) {
      const weight = weights[rowOffset + joint];
      if (weight < 0 || !Number.isFinite(weight)) {
        throw new RangeError(`${name}[${vertex}] contains an invalid weight.`);
      }
      total += weight;
      if (weight > 0) ranked.push([joint, weight]);
    }
    if (!(total > 0)) {
      throw new RangeError(`${name}[${vertex}] has no positive skinning weight.`);
    }
    ranked.sort((a, b) => b[1] - a[1]);
    const selected = ranked.slice(0, MANO_INFLUENCE_COUNT);
    const selectedTotal = selected.reduce((sum, entry) => sum + entry[1], 0);
    const packedOffset = vertex * MANO_INFLUENCE_COUNT;
    for (let influence = 0; influence < MANO_INFLUENCE_COUNT; influence += 1) {
      const entry = selected[influence] ?? selected[0] ?? [0, 1];
      joints[packedOffset + influence] = entry[0];
      packed[packedOffset + influence] = influence < selected.length ? entry[1] / selectedTotal : 0;
    }
  }
  return { influenceJoints: joints, influenceWeights: packed };
}

function validateHand(raw, side, faceCount) {
  assertObject(raw, `hands.${side}`);
  if (raw.side !== side) throw new TypeError(`hands.${side}.side must equal ${side}.`);
  const vertices = flattenNumeric(raw.vertices, MANO_VERTEX_COUNT, 3, `hands.${side}.vertices`);
  const faces = validateFaces(raw.faces, faceCount, `hands.${side}.faces`);
  const weights = flattenNumeric(
    raw.weights,
    MANO_VERTEX_COUNT,
    MANO_SKIN_JOINT_COUNT,
    `hands.${side}.weights`,
  );
  const restSkinJoints = flattenNumeric(
    raw.restSkinJoints,
    MANO_SKIN_JOINT_COUNT,
    3,
    `hands.${side}.restSkinJoints`,
  );
  const restLandmarks = flattenNumeric(
    raw.restLandmarks,
    MANO_LANDMARK_COUNT,
    3,
    `hands.${side}.restLandmarks`,
  );
  return {
    side,
    vertices,
    faces,
    weights,
    restSkinJoints,
    restLandmarks,
    ...packInfluences(weights, `hands.${side}.weights`),
  };
}

export function validateManoBundle(raw) {
  assertObject(raw, 'MANO bundle');
  if (raw.format !== MANO_BUNDLE_FORMAT) {
    throw new TypeError(`MANO bundle format must be ${MANO_BUNDLE_FORMAT}.`);
  }
  if (raw.private !== true || raw.redistributable !== false) {
    throw new TypeError('MANO bundle must be marked private and non-redistributable.');
  }
  if (raw.vertexCount !== MANO_VERTEX_COUNT) throw new RangeError('Unexpected MANO vertexCount.');
  if (raw.skinJointCount !== MANO_SKIN_JOINT_COUNT) throw new RangeError('Unexpected MANO skinJointCount.');
  if (raw.landmarkCount !== MANO_LANDMARK_COUNT) throw new RangeError('Unexpected MANO landmarkCount.');
  const faceCount = Number(raw.faceCount);
  if (!Number.isInteger(faceCount) || faceCount <= 0) throw new RangeError('Invalid MANO faceCount.');
  const jointParents = flattenNumeric(
    raw.jointParents,
    1,
    MANO_SKIN_JOINT_COUNT,
    'jointParents',
    Int8Array,
  );
  assertObject(raw.hands, 'hands');
  return {
    format: MANO_BUNDLE_FORMAT,
    private: true,
    redistributable: false,
    vertexCount: MANO_VERTEX_COUNT,
    faceCount,
    skinJointCount: MANO_SKIN_JOINT_COUNT,
    landmarkCount: MANO_LANDMARK_COUNT,
    jointParents,
    hands: {
      left: validateHand(raw.hands.left, 'left', faceCount),
      right: validateHand(raw.hands.right, 'right', faceCount),
    },
  };
}

export async function loadManoBundleFile(file) {
  if (!file || typeof file.text !== 'function') throw new TypeError('Select a MANO browser bundle JSON file.');
  if (file.size > 25 * 1024 * 1024) throw new RangeError('MANO browser bundle is unexpectedly large.');
  let parsed;
  try {
    parsed = JSON.parse(await file.text());
  } catch (error) {
    throw new TypeError(`MANO browser bundle is not valid JSON: ${error instanceof Error ? error.message : error}`);
  }
  return validateManoBundle(parsed);
}

function createMemoryAdapter() {
  const values = new Map();
  return {
    async get(key) { return values.has(key) ? structuredClone(values.get(key)) : null; },
    async set(key, value) { values.set(key, structuredClone(value)); },
    async delete(key) { values.delete(key); },
  };
}

function openDatabase(dbName, storeName) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, 1);
    request.addEventListener('upgradeneeded', () => {
      if (!request.result.objectStoreNames.contains(storeName)) request.result.createObjectStore(storeName);
    });
    request.addEventListener('success', () => resolve(request.result), { once: true });
    request.addEventListener('error', () => reject(request.error ?? new Error('IndexedDB open failed.')), { once: true });
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
      } finally { db.close(); }
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
      } finally { db.close(); }
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
      } finally { db.close(); }
    },
  };
}

export class ManoBundleStore {
  constructor({ adapter = null, key = DEFAULT_KEY } = {}) {
    this.adapter = adapter ?? (typeof indexedDB === 'undefined' ? createMemoryAdapter() : createIndexedDbAdapter());
    this.key = key;
  }

  async load() {
    const value = await this.adapter.get(this.key);
    return value == null ? null : validateManoBundle(value);
  }

  async save(value) {
    const bundle = validateManoBundle(value);
    await this.adapter.set(this.key, bundle);
    return bundle;
  }

  async clear() {
    await this.adapter.delete(this.key);
  }
}

export function createMemoryBundleStore() {
  return new ManoBundleStore({ adapter: createMemoryAdapter() });
}
