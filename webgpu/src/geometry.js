const TAU = Math.PI * 2;

export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

export function normalizeAngle(angle) {
  return angle - TAU * Math.floor((angle + Math.PI) / TAU);
}

function vectorAngle(start, end) {
  return Math.atan2(end.y - start.y, end.x - start.x);
}

function boundedRoi(cx, cy, size, rotation, width, height) {
  const maxSide = Math.max(width, height);
  return {
    cx: clamp(cx, -maxSide * 0.25, width + maxSide * 0.25),
    cy: clamp(cy, -maxSide * 0.25, height + maxSide * 0.25),
    size: clamp(size, 32, maxSide * 2),
    rotation: normalizeAngle(rotation),
  };
}

export function computeRoiFromPalm(detection, width, height) {
  if (!detection?.box || !Array.isArray(detection.keypoints) || detection.keypoints.length < 3) {
    throw new TypeError('A palm detection with a box and seven keypoints is required.');
  }
  const { box, keypoints } = detection;
  const boxWidth = Math.max(1, box.x2 - box.x1);
  const boxHeight = Math.max(1, box.y2 - box.y1);
  const baseSize = Math.max(boxWidth, boxHeight);
  const wrist = keypoints[0];
  const middle = keypoints[2];
  const dx = middle.x - wrist.x;
  const dy = middle.y - wrist.y;
  const length = Math.hypot(dx, dy) || 1;
  const shift = baseSize * 0.4;
  const cx = (box.x1 + box.x2) * 0.5 + (dx / length) * shift;
  const cy = (box.y1 + box.y2) * 0.5 + (dy / length) * shift;
  const rotation = vectorAngle(wrist, middle) + Math.PI / 2;
  return boundedRoi(cx, cy, baseSize * 3, rotation, width, height);
}

export function computeRoiFromLandmarks(points, width, height) {
  if (!Array.isArray(points) || points.length < 21) {
    throw new TypeError('Twenty-one projected landmarks are required.');
  }
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }
  const boxWidth = Math.max(1, maxX - minX);
  const boxHeight = Math.max(1, maxY - minY);
  const baseSize = Math.max(boxWidth, boxHeight);
  const wrist = points[0];
  const middle = points[9];
  const dx = middle.x - wrist.x;
  const dy = middle.y - wrist.y;
  const length = Math.hypot(dx, dy) || 1;
  const cx = (minX + maxX) * 0.5 + (dx / length) * baseSize * 0.1;
  const cy = (minY + maxY) * 0.5 + (dy / length) * baseSize * 0.1;
  const rotation = vectorAngle(wrist, middle) + Math.PI / 2;
  return boundedRoi(cx, cy, baseSize * 1.65, rotation, width, height);
}

export function projectPointFromRoi(x, y, roi, cropSize) {
  const localX = (x - cropSize * 0.5) * (roi.size / cropSize);
  const localY = (y - cropSize * 0.5) * (roi.size / cropSize);
  const cosine = Math.cos(roi.rotation);
  const sine = Math.sin(roi.rotation);
  return {
    x: roi.cx + localX * cosine - localY * sine,
    y: roi.cy + localX * sine + localY * cosine,
  };
}

export function projectLandmarksFromRoi(flatLandmarks, roi, cropSize) {
  if (flatLandmarks.length % 3 !== 0) {
    throw new RangeError('Landmark output length must be divisible by three.');
  }
  const points = [];
  const zScale = roi.size / cropSize;
  for (let index = 0; index < flatLandmarks.length; index += 3) {
    const projected = projectPointFromRoi(
      flatLandmarks[index],
      flatLandmarks[index + 1],
      roi,
      cropSize,
    );
    points.push({
      x: projected.x,
      y: projected.y,
      z: flatLandmarks[index + 2] * zScale,
    });
  }
  return points;
}

export function createRoiCrop(source, roi, cropSize, canvas = null) {
  const output = canvas ?? (typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(cropSize, cropSize)
    : document.createElement('canvas'));
  output.width = cropSize;
  output.height = cropSize;
  const context = output.getContext('2d', { willReadFrequently: true });
  context.save();
  context.clearRect(0, 0, cropSize, cropSize);
  context.fillStyle = '#000';
  context.fillRect(0, 0, cropSize, cropSize);
  const scale = cropSize / roi.size;
  context.translate(cropSize / 2, cropSize / 2);
  context.rotate(-roi.rotation);
  context.scale(scale, scale);
  context.translate(-roi.cx, -roi.cy);
  context.drawImage(source, 0, 0);
  context.restore();
  return output;
}
