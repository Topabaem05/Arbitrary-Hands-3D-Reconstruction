export const MANO_TO_MEDIAPIPE = new Uint8Array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]);
export const LANDMARK_CHILDREN = new Uint8Array([9, 6, 7, 8, 10, 11, 12, 18, 19, 20, 14, 15, 16, 2, 3, 4]);

const EPSILON = 1e-8;

function pointAt(points, index) {
  const offset = index * 3;
  return [points[offset], points[offset + 1], points[offset + 2]];
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(value) {
  return Math.hypot(value[0], value[1], value[2]);
}

function normalize(value, fallback = [1, 0, 0]) {
  const magnitude = length(value);
  if (!(magnitude > EPSILON)) return fallback.slice();
  return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
}

function distance(a, b) {
  return length(subtract(a, b));
}

function palmBasis(points) {
  const wrist = pointAt(points, 0);
  const index = pointAt(points, 5);
  const middle = pointAt(points, 9);
  const pinky = pointAt(points, 17);
  let y = normalize(subtract(middle, wrist), [0, 1, 0]);
  const across = normalize(subtract(index, pinky), [1, 0, 0]);
  let z = normalize(cross(across, y), [0, 0, 1]);
  let x = normalize(cross(y, z), across);
  z = normalize(cross(x, y), z);
  y = normalize(cross(z, x), y);
  return [x, y, z];
}

function fingerBasis(points, currentIndex, childIndex, normalReference) {
  const current = pointAt(points, currentIndex);
  const child = pointAt(points, childIndex);
  const y = normalize(subtract(child, current), [0, 1, 0]);
  let x = normalize(cross(y, normalReference), [1, 0, 0]);
  let z = normalize(cross(x, y), normalReference);
  x = normalize(cross(y, z), x);
  return [x, y, z];
}

function rotationFromBases(source, target, output, offset) {
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      output[offset + row * 3 + column] =
        target[0][row] * source[0][column]
        + target[1][row] * source[1][column]
        + target[2][row] * source[2][column];
    }
  }
}

function rotatePoint(matrix, offset, point) {
  return [
    matrix[offset] * point[0] + matrix[offset + 1] * point[1] + matrix[offset + 2] * point[2],
    matrix[offset + 3] * point[0] + matrix[offset + 4] * point[1] + matrix[offset + 5] * point[2],
    matrix[offset + 6] * point[0] + matrix[offset + 7] * point[1] + matrix[offset + 8] * point[2],
  ];
}

function normalizeTargetScale(target, rest) {
  if (target.length !== 21 * 3) throw new RangeError('Expected 21 target landmarks.');
  const targetWrist = pointAt(target, 0);
  const targetMiddle = pointAt(target, 9);
  const restWrist = pointAt(rest, 0);
  const restMiddle = pointAt(rest, 9);
  const targetScale = distance(targetWrist, targetMiddle);
  const restScale = distance(restWrist, restMiddle);
  const ratio = targetScale > EPSILON ? restScale / targetScale : 1;
  const normalized = new Float32Array(target.length);
  for (let index = 0; index < target.length; index += 3) {
    normalized[index] = targetWrist[0] + (target[index] - targetWrist[0]) * ratio;
    normalized[index + 1] = targetWrist[1] + (target[index + 1] - targetWrist[1]) * ratio;
    normalized[index + 2] = targetWrist[2] + (target[index + 2] - targetWrist[2]) * ratio;
  }
  return normalized;
}

export function landmarksFromHand(hand) {
  const source = hand?.points?.length === 21 ? hand.points : hand?.worldPoints;
  if (!source || source.length !== 21) throw new RangeError('Hand result must contain 21 landmarks.');
  const usesScreenPoints = source === hand.points;
  const output = new Float32Array(21 * 3);
  for (let index = 0; index < 21; index += 1) {
    const point = source[index];
    output[index * 3] = Number(point.x) || 0;
    output[index * 3 + 1] = usesScreenPoints ? -(Number(point.y) || 0) : Number(point.y) || 0;
    output[index * 3 + 2] = Number(point.z) || 0;
  }
  return output;
}

export class LandmarkInterpolator {
  constructor({ halfLifeMs = 32 } = {}) {
    if (!(halfLifeMs > 0)) throw new RangeError('halfLifeMs must be positive.');
    this.halfLifeMs = halfLifeMs;
    this.current = null;
    this.target = null;
    this.lastSampleTime = null;
    this.targetTime = 0;
  }

  setTarget(points, timestamp = performance.now()) {
    const typed = points instanceof Float32Array ? points : new Float32Array(points);
    if (typed.length !== 21 * 3) throw new RangeError('Expected 21 target landmarks.');
    if (!this.current) this.current = new Float32Array(typed);
    this.target = new Float32Array(typed);
    this.targetTime = timestamp;
    if (this.lastSampleTime == null) this.lastSampleTime = timestamp;
  }

  sample(timestamp = performance.now()) {
    if (!this.current || !this.target) return null;
    const elapsed = Math.max(0, timestamp - (this.lastSampleTime ?? timestamp));
    const alpha = elapsed === 0 ? 0 : 1 - Math.pow(0.5, elapsed / this.halfLifeMs);
    for (let index = 0; index < this.current.length; index += 1) {
      this.current[index] += (this.target[index] - this.current[index]) * alpha;
    }
    this.lastSampleTime = timestamp;
    return this.current;
  }

  reset() {
    this.current = null;
    this.target = null;
    this.lastSampleTime = null;
  }
}

export class ManoHandDeformer {
  constructor(hand) {
    for (const key of ['vertices', 'restSkinJoints', 'restLandmarks', 'influenceJoints', 'influenceWeights']) {
      if (!hand?.[key]) throw new TypeError(`Missing MANO hand field: ${key}`);
    }
    this.vertices = hand.vertices;
    this.restSkinJoints = hand.restSkinJoints;
    this.restLandmarks = hand.restLandmarks;
    this.influenceJoints = hand.influenceJoints;
    this.influenceWeights = hand.influenceWeights;
    this.vertexCount = this.vertices.length / 3;
    if (this.influenceJoints.length !== this.vertexCount * 4 || this.influenceWeights.length !== this.vertexCount * 4) {
      throw new RangeError('Packed MANO influences must contain four values per vertex.');
    }
    this.rotations = new Float32Array(16 * 9);
    this.translations = new Float32Array(16 * 3);
    this.output = new Float32Array(this.vertices.length);
    this.restPalmBasis = palmBasis(this.restLandmarks);
  }

  deform(targetLandmarks) {
    const target = normalizeTargetScale(
      targetLandmarks instanceof Float32Array ? targetLandmarks : new Float32Array(targetLandmarks),
      this.restLandmarks,
    );
    const targetPalmBasis = palmBasis(target);
    const restNormal = this.restPalmBasis[2];
    const targetNormal = targetPalmBasis[2];

    for (let joint = 0; joint < 16; joint += 1) {
      const sourceBasis = joint === 0
        ? this.restPalmBasis
        : fingerBasis(
          this.restLandmarks,
          MANO_TO_MEDIAPIPE[joint],
          LANDMARK_CHILDREN[joint],
          restNormal,
        );
      const destinationBasis = joint === 0
        ? targetPalmBasis
        : fingerBasis(
          target,
          MANO_TO_MEDIAPIPE[joint],
          LANDMARK_CHILDREN[joint],
          targetNormal,
        );
      const rotationOffset = joint * 9;
      rotationFromBases(sourceBasis, destinationBasis, this.rotations, rotationOffset);
      const restJoint = pointAt(this.restSkinJoints, joint);
      const targetJoint = pointAt(target, MANO_TO_MEDIAPIPE[joint]);
      const rotated = rotatePoint(this.rotations, rotationOffset, restJoint);
      const translationOffset = joint * 3;
      this.translations[translationOffset] = targetJoint[0] - rotated[0];
      this.translations[translationOffset + 1] = targetJoint[1] - rotated[1];
      this.translations[translationOffset + 2] = targetJoint[2] - rotated[2];
    }

    for (let vertex = 0; vertex < this.vertexCount; vertex += 1) {
      const vertexOffset = vertex * 3;
      const vx = this.vertices[vertexOffset];
      const vy = this.vertices[vertexOffset + 1];
      const vz = this.vertices[vertexOffset + 2];
      let ox = 0;
      let oy = 0;
      let oz = 0;
      const influenceOffset = vertex * 4;
      for (let influence = 0; influence < 4; influence += 1) {
        const weight = this.influenceWeights[influenceOffset + influence];
        if (weight === 0) continue;
        const joint = this.influenceJoints[influenceOffset + influence];
        const rotationOffset = joint * 9;
        const translationOffset = joint * 3;
        ox += weight * (
          this.rotations[rotationOffset] * vx
          + this.rotations[rotationOffset + 1] * vy
          + this.rotations[rotationOffset + 2] * vz
          + this.translations[translationOffset]
        );
        oy += weight * (
          this.rotations[rotationOffset + 3] * vx
          + this.rotations[rotationOffset + 4] * vy
          + this.rotations[rotationOffset + 5] * vz
          + this.translations[translationOffset + 1]
        );
        oz += weight * (
          this.rotations[rotationOffset + 6] * vx
          + this.rotations[rotationOffset + 7] * vy
          + this.rotations[rotationOffset + 8] * vz
          + this.translations[translationOffset + 2]
        );
      }
      this.output[vertexOffset] = ox;
      this.output[vertexOffset + 1] = oy;
      this.output[vertexOffset + 2] = oz;
    }
    return this.output;
  }
}
