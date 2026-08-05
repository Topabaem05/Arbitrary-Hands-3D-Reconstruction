export const BONE_NAMES = [
  'wrist',
  'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
  'index_mcp', 'index_pip', 'index_dip', 'index_tip',
  'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
  'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
  'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
];

export const BONE_PARENTS = new Int8Array([
  -1,
  0, 1, 2, 3,
  0, 5, 6, 7,
  0, 9, 10, 11,
  0, 13, 14, 15,
  0, 17, 18, 19,
]);

const CHILD = new Int8Array([
  -1,
  2, 3, 4, -1,
  6, 7, 8, -1,
  10, 11, 12, -1,
  14, 15, 16, -1,
  18, 19, 20, -1,
]);
const EPSILON = 1e-8;

function pointAt(points, index) {
  const offset = index * 3;
  return [points[offset], points[offset + 1], points[offset + 2]];
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(a, value) {
  return [a[0] * value, a[1] * value, a[2] * value];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function magnitude(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

function normalize(a, fallback = [1, 0, 0]) {
  const length = magnitude(a);
  return length > EPSILON ? scale(a, 1 / length) : fallback.slice();
}

function distance(a, b) {
  return magnitude(subtract(a, b));
}

function palmBasis(points) {
  const wrist = pointAt(points, 0);
  const index = pointAt(points, 5);
  const middle = pointAt(points, 9);
  const pinky = pointAt(points, 17);
  let forward = normalize(subtract(middle, wrist), [1, 0, 0]);
  let across = normalize(subtract(index, pinky), [0, 1, 0]);
  let normal = normalize(cross(across, forward), [0, 0, 1]);
  across = normalize(cross(forward, normal), across);
  normal = normalize(cross(across, forward), normal);
  forward = normalize(cross(normal, across), forward);
  return [across, forward, normal];
}

function matrixFromBases(source, target) {
  const matrix = new Array(9).fill(0);
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      matrix[row * 3 + column] =
        target[0][row] * source[0][column]
        + target[1][row] * source[1][column]
        + target[2][row] * source[2][column];
    }
  }
  return matrix;
}

function quatNormalize(q) {
  const length = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(length > EPSILON)) return [0, 0, 0, 1];
  return q.map((value) => value / length);
}

function quatFromMatrix(matrix) {
  const m00 = matrix[0];
  const m01 = matrix[1];
  const m02 = matrix[2];
  const m10 = matrix[3];
  const m11 = matrix[4];
  const m12 = matrix[5];
  const m20 = matrix[6];
  const m21 = matrix[7];
  const m22 = matrix[8];
  const trace = m00 + m11 + m22;
  let x;
  let y;
  let z;
  let w;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  return quatNormalize([x, y, z, w]);
}

function quatProduct(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

function quatMultiply(a, b) {
  return quatNormalize(quatProduct(a, b));
}

function quatConjugate(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}

export function quatRotateVector(q, vector) {
  const pure = [vector[0], vector[1], vector[2], 0];
  const rotated = quatProduct(quatProduct(q, pure), quatConjugate(q));
  return rotated.slice(0, 3);
}

function quatFromUnitVectors(fromValue, toValue) {
  const from = normalize(fromValue);
  const to = normalize(toValue);
  let real = 1 + dot(from, to);
  let imaginary;
  if (real < EPSILON) {
    real = 0;
    imaginary = Math.abs(from[0]) > Math.abs(from[2])
      ? [-from[1], from[0], 0]
      : [0, -from[2], from[1]];
  } else {
    imaginary = cross(from, to);
  }
  return quatNormalize([imaginary[0], imaginary[1], imaginary[2], real]);
}

function toFlatLandmarks(value) {
  if (value instanceof Float32Array || value instanceof Float64Array) {
    if (value.length !== 63) throw new RangeError('Expected 21 landmarks.');
    return new Float32Array(value);
  }
  if (!Array.isArray(value)) throw new TypeError('Landmarks must be a flat or nested array.');
  if (value.length === 63 && !Array.isArray(value[0])) return new Float32Array(value);
  if (value.length !== 21) throw new RangeError('Expected 21 landmarks.');
  const output = new Float32Array(63);
  for (let index = 0; index < 21; index += 1) {
    const point = value[index];
    output[index * 3] = Number(point.x ?? point[0]) || 0;
    output[index * 3 + 1] = Number(point.y ?? point[1]) || 0;
    output[index * 3 + 2] = Number(point.z ?? point[2]) || 0;
  }
  return output;
}

export function landmarksFromHand(hand) {
  const source = hand?.points?.length === 21 ? hand.points : hand?.worldPoints;
  if (!source || source.length !== 21) {
    throw new RangeError('Hand result must contain 21 landmarks.');
  }
  const usesScreen = source === hand.points;
  const output = new Float32Array(63);
  for (let index = 0; index < 21; index += 1) {
    output[index * 3] = Number(source[index].x) || 0;
    output[index * 3 + 1] = usesScreen
      ? -(Number(source[index].y) || 0)
      : Number(source[index].y) || 0;
    output[index * 3 + 2] = Number(source[index].z) || 0;
  }
  return output;
}

function canonicalTarget(rest, target) {
  const restBasis = palmBasis(rest);
  const targetBasis = palmBasis(target);
  const restWrist = pointAt(rest, 0);
  const targetWrist = pointAt(target, 0);
  const restScale = distance(restWrist, pointAt(rest, 9));
  const targetScale = distance(targetWrist, pointAt(target, 9));
  const ratio = targetScale > EPSILON ? restScale / targetScale : 1;
  const canonical = new Float32Array(63);
  for (let index = 0; index < 21; index += 1) {
    const relative = subtract(pointAt(target, index), targetWrist);
    const local = [
      dot(relative, targetBasis[0]),
      dot(relative, targetBasis[1]),
      dot(relative, targetBasis[2]),
    ];
    const mapped = add(
      restWrist,
      add(
        scale(restBasis[0], local[0] * ratio),
        add(
          scale(restBasis[1], local[1] * ratio),
          scale(restBasis[2], local[2] * ratio),
        ),
      ),
    );
    canonical.set(mapped, index * 3);
  }
  return {
    canonical,
    rootQuaternion: quatFromMatrix(matrixFromBases(restBasis, targetBasis)),
  };
}

export function solveRigPose(restValue, targetValue, handedness = 'left') {
  const rest = toFlatLandmarks(restValue);
  const target = toFlatLandmarks(targetValue);
  const { canonical, rootQuaternion } = canonicalTarget(rest, target);
  const localQuaternions = new Float32Array(21 * 4);
  const globalQuaternions = Array.from({ length: 21 }, () => [0, 0, 0, 1]);
  for (let bone = 0; bone < 21; bone += 1) {
    const child = CHILD[bone];
    let local = [0, 0, 0, 1];
    if (child >= 0) {
      const restDirection = subtract(pointAt(rest, child), pointAt(rest, bone));
      const targetDirection = subtract(pointAt(canonical, child), pointAt(canonical, bone));
      const parent = BONE_PARENTS[bone];
      const parentGlobal = parent >= 0 ? globalQuaternions[parent] : [0, 0, 0, 1];
      const targetInParent = quatRotateVector(
        quatConjugate(parentGlobal),
        normalize(targetDirection),
      );
      local = quatFromUnitVectors(restDirection, targetInParent);
      globalQuaternions[bone] = quatMultiply(parentGlobal, local);
    } else {
      const parent = BONE_PARENTS[bone];
      globalQuaternions[bone] = parent >= 0 ? globalQuaternions[parent] : [0, 0, 0, 1];
    }
    localQuaternions.set(local, bone * 4);
  }
  return {
    rootQuaternion: new Float32Array(rootQuaternion),
    boneQuaternions: localQuaternions,
    canonicalLandmarks: canonical,
    mirrorY: handedness === 'right',
  };
}

export class HandLandmarkSmoother {
  constructor({ halfLifeMs = 32 } = {}) {
    if (!(halfLifeMs > 0)) throw new RangeError('halfLifeMs must be positive.');
    this.halfLifeMs = halfLifeMs;
    this.current = null;
    this.target = null;
    this.lastTime = null;
  }

  setTarget(value, timestamp = performance.now()) {
    const target = toFlatLandmarks(value);
    if (!this.current) this.current = new Float32Array(target);
    this.target = target;
    if (this.lastTime == null) this.lastTime = timestamp;
  }

  sample(timestamp = performance.now()) {
    if (!this.current || !this.target) return null;
    const elapsed = Math.max(0, timestamp - (this.lastTime ?? timestamp));
    const alpha = elapsed === 0 ? 0 : 1 - Math.pow(0.5, elapsed / this.halfLifeMs);
    for (let index = 0; index < this.current.length; index += 1) {
      this.current[index] += (this.target[index] - this.current[index]) * alpha;
    }
    this.lastTime = timestamp;
    return this.current;
  }

  reset() {
    this.current = null;
    this.target = null;
    this.lastTime = null;
  }
}
