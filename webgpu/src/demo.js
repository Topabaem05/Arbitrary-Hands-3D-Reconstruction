const FINGER_BASES = [-0.55, -0.28, 0, 0.28, 0.52];

function makeHand(time, handedness, width, height, phase = 0) {
  const direction = handedness === 'right' ? 1 : -1;
  const centerX = width * (handedness === 'right' ? 0.63 : 0.37) + Math.sin(time * 0.0007 + phase) * width * 0.04;
  const centerY = height * 0.67 + Math.cos(time * 0.0009 + phase) * height * 0.025;
  const scale = Math.min(width, height) * 0.19;
  const points = [{ x: centerX, y: centerY, z: 0 }];
  const worldPoints = [{ x: 0, y: 0, z: 0 }];
  const fingers = [
    { indices: [1, 2, 3, 4], base: FINGER_BASES[0], length: 0.82 },
    { indices: [5, 6, 7, 8], base: FINGER_BASES[1], length: 1.15 },
    { indices: [9, 10, 11, 12], base: FINGER_BASES[2], length: 1.28 },
    { indices: [13, 14, 15, 16], base: FINGER_BASES[3], length: 1.17 },
    { indices: [17, 18, 19, 20], base: FINGER_BASES[4], length: 0.98 },
  ];
  for (const [fingerIndex, finger] of fingers.entries()) {
    const curl = 0.08 + 0.12 * (Math.sin(time * 0.0014 + fingerIndex + phase) + 1) * 0.5;
    for (let joint = 0; joint < 4; joint += 1) {
      const progress = (joint + 1) / 4;
      const x = finger.base * scale * direction * (0.82 + progress * 0.18);
      const y = -finger.length * scale * progress + curl * scale * progress * progress;
      const z = Math.sin(progress * Math.PI) * curl * scale;
      points[finger.indices[joint]] = { x: centerX + x, y: centerY + y, z };
      worldPoints[finger.indices[joint]] = { x: x / scale, y: y / scale, z: z / scale };
    }
  }
  return { handedness, handednessScore: handedness === 'right' ? 0.9 : 0.1, score: 1, points, worldPoints };
}

export function syntheticHands(time, width, height, count = 1) {
  if (count === 1) return [makeHand(time, 'right', width, height)];
  return [makeHand(time, 'left', width, height, 1.7), makeHand(time, 'right', width, height, 0)];
}

export function createDemoFrame(width = 1280, height = 720) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  const gradient = context.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#1b2433');
  gradient.addColorStop(0.48, '#141922');
  gradient.addColorStop(1, '#080a0f');
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  context.fillStyle = 'rgba(255,255,255,0.045)';
  for (let y = 0; y < height; y += 48) {
    for (let x = 0; x < width; x += 48) {
      context.beginPath();
      context.arc(x + 24, y + 24, 1.5, 0, Math.PI * 2);
      context.fill();
    }
  }
  return canvas;
}
