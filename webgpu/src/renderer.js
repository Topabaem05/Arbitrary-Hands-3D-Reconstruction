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
  return { width, height, ratio };
}

function sourceDimensions(source) {
  return {
    width: source.videoWidth || source.naturalWidth || source.width,
    height: source.videoHeight || source.naturalHeight || source.height,
  };
}

function containRect(sourceWidth, sourceHeight, canvasWidth, canvasHeight) {
  const scale = Math.min(canvasWidth / sourceWidth, canvasHeight / sourceHeight);
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

  setMirrored(value) {
    this.mirrored = Boolean(value);
  }

  draw(source, hands = [], stats = {}) {
    const { width, height } = fitCanvas(this.canvas);
    const dimensions = sourceDimensions(source);
    const rect = containRect(dimensions.width, dimensions.height, width, height);
    const context = this.context;
    context.fillStyle = '#07090d';
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

    const lineWidth = Math.max(2, width / 420);
    for (const hand of hands) {
      const accent = hand.handedness === 'right' ? '#ffb86b' : '#6fd7ff';
      const points = hand.points.map((point) => transformPoint(point, rect, dimensions.width, this.mirrored));
      context.strokeStyle = accent;
      context.lineWidth = lineWidth;
      context.lineCap = 'round';
      context.lineJoin = 'round';
      context.globalAlpha = 0.9;
      for (const [start, end] of HAND_CONNECTIONS) {
        context.beginPath();
        context.moveTo(points[start].x, points[start].y);
        context.lineTo(points[end].x, points[end].y);
        context.stroke();
      }
      context.globalAlpha = 1;
      context.fillStyle = '#ffffff';
      for (const point of points) {
        context.beginPath();
        context.arc(point.x, point.y, lineWidth * 1.25, 0, Math.PI * 2);
        context.fill();
      }
    }

    const padding = 14 * (window.devicePixelRatio || 1);
    context.font = `${13 * (window.devicePixelRatio || 1)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    context.textBaseline = 'top';
    context.fillStyle = 'rgba(5, 8, 12, 0.72)';
    context.fillRect(padding, padding, Math.min(width - padding * 2, 430 * (window.devicePixelRatio || 1)), 54 * (window.devicePixelRatio || 1));
    context.fillStyle = '#f7f8fa';
    const display = Number.isFinite(stats.displayFps) ? stats.displayFps.toFixed(1) : '0.0';
    const inference = Number.isFinite(stats.inferenceFps) ? stats.inferenceFps.toFixed(1) : '0.0';
    context.fillText(`display ${display} FPS  ·  inference ${inference} FPS`, padding * 1.55, padding * 1.45);
    context.fillStyle = '#9fa7b5';
    context.fillText(`${stats.backend || 'not loaded'}  ·  ${hands.length} hand${hands.length === 1 ? '' : 's'}`, padding * 1.55, padding * 1.45 + 22 * (window.devicePixelRatio || 1));
  }
}

function normalizedWorldPoints(hand) {
  const source = hand.worldPoints?.length === 21
    ? hand.worldPoints
    : hand.points.map((point) => ({ x: point.x, y: point.y, z: point.z }));
  const root = source[0];
  const centered = source.map((point) => ({ x: point.x - root.x, y: point.y - root.y, z: point.z - root.z }));
  const reference = centered[9];
  const scale = Math.hypot(reference.x, reference.y, reference.z) || 1;
  return centered.map((point) => ({ x: point.x / scale, y: point.y / scale, z: point.z / scale }));
}

function rotatePoint(point, yaw, pitch) {
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);
  const x1 = point.x * cosYaw - point.z * sinYaw;
  const z1 = point.x * sinYaw + point.z * cosYaw;
  return {
    x: x1,
    y: point.y * cosPitch - z1 * sinPitch,
    z: point.y * sinPitch + z1 * cosPitch,
  };
}

export class Skeleton3DRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.yaw = -0.35;
    this.pitch = 0.35;
    this.dragging = false;
    this.lastPointer = null;
    canvas.addEventListener('pointerdown', (event) => {
      this.dragging = true;
      this.lastPointer = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!this.dragging || !this.lastPointer) return;
      this.yaw += (event.clientX - this.lastPointer.x) * 0.01;
      this.pitch += (event.clientY - this.lastPointer.y) * 0.01;
      this.lastPointer = { x: event.clientX, y: event.clientY };
    });
    const stop = () => { this.dragging = false; this.lastPointer = null; };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
  }

  draw(hands = []) {
    const { width, height } = fitCanvas(this.canvas);
    const context = this.context;
    const gradient = context.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, '#10141c');
    gradient.addColorStop(1, '#05070b');
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);
    context.strokeStyle = 'rgba(255,255,255,0.06)';
    context.lineWidth = 1;
    const grid = 36 * (window.devicePixelRatio || 1);
    for (let x = width / 2 % grid; x < width; x += grid) {
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
    }
    for (let y = height / 2 % grid; y < height; y += grid) {
      context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
    }

    if (hands.length === 0) {
      context.fillStyle = '#7f8795';
      context.font = `${15 * (window.devicePixelRatio || 1)}px system-ui, sans-serif`;
      context.textAlign = 'center';
      context.fillText('3D hand skeleton', width / 2, height / 2);
      return;
    }

    hands.forEach((hand, handIndex) => {
      const points3d = normalizedWorldPoints(hand).map((point) => rotatePoint(point, this.yaw, this.pitch));
      const centerX = width * (hands.length === 1 ? 0.5 : handIndex === 0 ? 0.32 : 0.68);
      const centerY = height * 0.56;
      const zoom = Math.min(width, height) * 0.27;
      const projected = points3d.map((point) => {
        const perspective = 1 / Math.max(0.45, 1.8 + point.z * 0.35);
        return {
          x: centerX + point.x * zoom * perspective,
          y: centerY + point.y * zoom * perspective,
          z: point.z,
          perspective,
        };
      });
      const accent = hand.handedness === 'right' ? '#ffb86b' : '#6fd7ff';
      context.strokeStyle = accent;
      context.lineCap = 'round';
      for (const [start, end] of HAND_CONNECTIONS) {
        const depth = (projected[start].perspective + projected[end].perspective) * 0.5;
        context.lineWidth = Math.max(2, depth * 5 * (window.devicePixelRatio || 1));
        context.globalAlpha = Math.min(1, 0.55 + depth * 0.35);
        context.beginPath();
        context.moveTo(projected[start].x, projected[start].y);
        context.lineTo(projected[end].x, projected[end].y);
        context.stroke();
      }
      context.globalAlpha = 1;
      for (const point of projected) {
        context.fillStyle = '#f7f8fa';
        context.beginPath();
        context.arc(point.x, point.y, Math.max(2, point.perspective * 4 * (window.devicePixelRatio || 1)), 0, Math.PI * 2);
        context.fill();
      }
    });
  }
}
