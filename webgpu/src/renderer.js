export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

function fitCanvas(canvas) {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height };
}

function sourceDimensions(source) {
  return {
    width: source.videoWidth || source.displayWidth || source.naturalWidth || source.width,
    height: source.videoHeight || source.displayHeight || source.naturalHeight || source.height,
  };
}

function coverRect(sourceWidth, sourceHeight, canvasWidth, canvasHeight) {
  const scale = Math.max(canvasWidth / sourceWidth, canvasHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return { x: (canvasWidth - width) / 2, y: (canvasHeight - height) / 2, width, height, scale };
}

function transformPoint(point, rect, sourceWidth, mirrored) {
  const x = mirrored ? sourceWidth - point.x : point.x;
  return { x: rect.x + x * rect.scale, y: rect.y + point.y * rect.scale };
}

export class OverlayRenderer {
  constructor(canvas, { mirrored = true } = {}) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d', { alpha: false });
    this.mirrored = mirrored;
  }

  draw(source, hands = []) {
    const { width, height } = fitCanvas(this.canvas);
    const dimensions = sourceDimensions(source);
    if (!(dimensions.width > 0 && dimensions.height > 0)) return;
    const rect = coverRect(dimensions.width, dimensions.height, width, height);
    const context = this.context;
    context.fillStyle = '#000';
    context.fillRect(0, 0, width, height);
    context.save();
    if (this.mirrored) {
      context.translate(rect.x + rect.width, rect.y);
      context.scale(-1, 1);
      context.drawImage(source, 0, 0, rect.width, rect.height);
    } else {
      context.drawImage(source, rect.x, rect.y, rect.width, rect.height);
    }
    context.restore();

    const lineWidth = Math.max(1.5, width / 260);
    for (const hand of hands) {
      const points = hand.points.map((point) => transformPoint(point, rect, dimensions.width, this.mirrored));
      context.strokeStyle = hand.handedness === 'right' ? '#ffb86b' : '#6fd7ff';
      context.lineWidth = lineWidth;
      context.lineCap = 'round';
      context.lineJoin = 'round';
      context.globalAlpha = 0.92;
      for (const [start, end] of HAND_CONNECTIONS) {
        context.beginPath();
        context.moveTo(points[start].x, points[start].y);
        context.lineTo(points[end].x, points[end].y);
        context.stroke();
      }
      context.globalAlpha = 1;
    }
  }
}
