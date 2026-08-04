export function sigmoid(value) {
  const clipped = Math.max(-100, Math.min(100, value));
  return 1 / (1 + Math.exp(-clipped));
}

export function generatePalmAnchors() {
  const anchors = [];
  const appendGrid = (gridSize, anchorsPerCell) => {
    for (let y = 0; y < gridSize; y += 1) {
      for (let x = 0; x < gridSize; x += 1) {
        const center = [(x + 0.5) / gridSize, (y + 0.5) / gridSize];
        for (let repeat = 0; repeat < anchorsPerCell; repeat += 1) {
          anchors.push(center.slice());
        }
      }
    }
  };
  appendGrid(24, 2);
  appendGrid(12, 6);
  return anchors;
}

export function intersectionOverUnion(a, b) {
  const left = Math.max(a.x1, b.x1);
  const top = Math.max(a.y1, b.y1);
  const right = Math.min(a.x2, b.x2);
  const bottom = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

export function nonMaxSuppression(candidates, iouThreshold = 0.3, limit = 2) {
  const remaining = [...candidates].sort((a, b) => b.score - a.score);
  const selected = [];
  while (remaining.length > 0 && selected.length < limit) {
    const candidate = remaining.shift();
    selected.push(candidate);
    for (let index = remaining.length - 1; index >= 0; index -= 1) {
      if (intersectionOverUnion(candidate.box, remaining[index].box) > iouThreshold) {
        remaining.splice(index, 1);
      }
    }
  }
  return selected;
}

function sourceCoordinate(normalized, axis, transform) {
  const detectorPixels = normalized * transform.inputSize;
  const pad = axis === 'x' ? transform.padX : transform.padY;
  return (detectorPixels - pad) / transform.scale;
}

export function decodePalmDetections(boxValues, scoreValues, options = {}) {
  const anchors = options.anchors ?? generatePalmAnchors();
  const inputSize = options.inputSize ?? 192;
  const transform = {
    inputSize,
    sourceWidth: options.sourceWidth ?? inputSize,
    sourceHeight: options.sourceHeight ?? inputSize,
    scale: options.scale ?? 1,
    padX: options.padX ?? 0,
    padY: options.padY ?? 0,
  };
  const scoreThreshold = options.scoreThreshold ?? 0.5;
  const iouThreshold = options.iouThreshold ?? 0.3;
  const maxHands = options.maxHands ?? 2;
  const coordinatesPerAnchor = Math.floor(boxValues.length / anchors.length);
  if (coordinatesPerAnchor < 18 || scoreValues.length < anchors.length) {
    throw new RangeError(
      `Unexpected palm output sizes: boxes=${boxValues.length}, scores=${scoreValues.length}, anchors=${anchors.length}`,
    );
  }

  const candidates = [];
  for (let anchorIndex = 0; anchorIndex < anchors.length; anchorIndex += 1) {
    const score = sigmoid(scoreValues[anchorIndex]);
    if (score < scoreThreshold) continue;
    const offset = anchorIndex * coordinatesPerAnchor;
    const anchor = anchors[anchorIndex];
    const centerX = boxValues[offset] / inputSize + anchor[0];
    const centerY = boxValues[offset + 1] / inputSize + anchor[1];
    const width = Math.abs(boxValues[offset + 2] / inputSize);
    const height = Math.abs(boxValues[offset + 3] / inputSize);
    const box = {
      x1: sourceCoordinate(centerX - width / 2, 'x', transform),
      y1: sourceCoordinate(centerY - height / 2, 'y', transform),
      x2: sourceCoordinate(centerX + width / 2, 'x', transform),
      y2: sourceCoordinate(centerY + height / 2, 'y', transform),
    };
    const keypoints = [];
    for (let keypointIndex = 0; keypointIndex < 7; keypointIndex += 1) {
      const x = boxValues[offset + 4 + keypointIndex * 2] / inputSize + anchor[0];
      const y = boxValues[offset + 5 + keypointIndex * 2] / inputSize + anchor[1];
      keypoints.push({
        x: sourceCoordinate(x, 'x', transform),
        y: sourceCoordinate(y, 'y', transform),
      });
    }
    if (![box.x1, box.y1, box.x2, box.y2].every(Number.isFinite)) continue;
    candidates.push({ score, box, keypoints });
  }
  return nonMaxSuppression(candidates, iouThreshold, maxHands);
}

export function identifyPalmOutputs(outputs) {
  const entries = Object.entries(outputs);
  const boxEntry = entries.find(([, tensor]) => tensor.data.length % 18 === 0 && tensor.data.length > 1000);
  const scoreEntry = entries.find(([, tensor]) => tensor.data.length >= 1000 && tensor.data.length < 10000 && tensor !== boxEntry?.[1]);
  if (!boxEntry || !scoreEntry) {
    const signature = entries.map(([name, tensor]) => `${name}:${tensor.dims.join('x')}`).join(', ');
    throw new Error(`Unable to identify palm detector outputs (${signature}).`);
  }
  return { boxes: boxEntry[1], scores: scoreEntry[1] };
}
