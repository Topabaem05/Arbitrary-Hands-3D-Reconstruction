import { LandmarkInterpolator, ManoHandDeformer, landmarksFromHand } from './mano-deformer.js';

const VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
uniform mat4 uViewProjection;
uniform mat3 uRotation;
uniform vec3 uCenter;
uniform vec3 uOffset;
uniform float uScale;
out vec3 vNormal;
out vec3 vPosition;
void main() {
  vec3 local = uRotation * ((aPosition - uCenter) * uScale) + uOffset;
  vPosition = local;
  vNormal = normalize(uRotation * aNormal);
  gl_Position = uViewProjection * vec4(local, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vPosition;
uniform vec3 uColor;
uniform float uOpacity;
out vec4 outColor;
void main() {
  vec3 normal = normalize(vNormal);
  vec3 key = normalize(vec3(-0.45, 0.72, 0.52));
  vec3 rimDirection = normalize(vec3(0.55, 0.25, -0.80));
  float diffuse = max(dot(normal, key), 0.0);
  float rim = pow(1.0 - max(dot(normal, normalize(-vPosition)), 0.0), 2.4);
  float fill = 0.20 + diffuse * 0.72 + max(dot(normal, rimDirection), 0.0) * 0.10;
  vec3 color = uColor * fill + vec3(0.12, 0.16, 0.20) * rim;
  outColor = vec4(color, uOpacity);
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || 'Unknown shader compilation error.';
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function createProgram(gl) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) || 'Unknown WebGL program link error.';
    gl.deleteProgram(program);
    throw new Error(message);
  }
  return program;
}

function normalize3(x, y, z) {
  const magnitude = Math.hypot(x, y, z) || 1;
  return [x / magnitude, y / magnitude, z / magnitude];
}

export function computeVertexNormals(vertices, faces, target = new Float32Array(vertices.length)) {
  if (target.length !== vertices.length) throw new RangeError('Normal target must match vertex array length.');
  target.fill(0);
  for (let index = 0; index < faces.length; index += 3) {
    const a = faces[index] * 3;
    const b = faces[index + 1] * 3;
    const c = faces[index + 2] * 3;
    const abx = vertices[b] - vertices[a];
    const aby = vertices[b + 1] - vertices[a + 1];
    const abz = vertices[b + 2] - vertices[a + 2];
    const acx = vertices[c] - vertices[a];
    const acy = vertices[c + 1] - vertices[a + 1];
    const acz = vertices[c + 2] - vertices[a + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    for (const vertex of [a, b, c]) {
      target[vertex] += nx;
      target[vertex + 1] += ny;
      target[vertex + 2] += nz;
    }
  }
  for (let index = 0; index < target.length; index += 3) {
    const [x, y, z] = normalize3(target[index], target[index + 1], target[index + 2]);
    target[index] = x;
    target[index + 1] = y;
    target[index + 2] = z;
  }
  return target;
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const inverse = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * inverse, -1,
    0, 0, 2 * far * near * inverse, 0,
  ]);
}

function multiply4(a, b) {
  const output = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let inner = 0; inner < 4; inner += 1) {
        sum += a[inner * 4 + row] * b[column * 4 + inner];
      }
      output[column * 4 + row] = sum;
    }
  }
  return output;
}

function translation4(x, y, z) {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
}

function rotation3(yaw, pitch) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  return new Float32Array([
    cy, sy * sp, sy * cp,
    0, cp, -sp,
    -sy, cy * sp, cy * cp,
  ]);
}

function fitCanvas(canvas, gl) {
  const ratio = Math.min(2, globalThis.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    gl.viewport(0, 0, width, height);
  }
  return { width, height };
}

function meshScale(vertices) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < vertices.length; index += 3) {
    minX = Math.min(minX, vertices[index]);
    maxX = Math.max(maxX, vertices[index]);
    minY = Math.min(minY, vertices[index + 1]);
    maxY = Math.max(maxY, vertices[index + 1]);
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-4);
  return 1.65 / span;
}

export class HandMeshRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: true,
      depth: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('WebGL2 is required for the MANO surface renderer.');
    this.gl = gl;
    this.program = createProgram(gl);
    this.locations = {
      viewProjection: gl.getUniformLocation(this.program, 'uViewProjection'),
      rotation: gl.getUniformLocation(this.program, 'uRotation'),
      center: gl.getUniformLocation(this.program, 'uCenter'),
      offset: gl.getUniformLocation(this.program, 'uOffset'),
      scale: gl.getUniformLocation(this.program, 'uScale'),
      color: gl.getUniformLocation(this.program, 'uColor'),
      opacity: gl.getUniformLocation(this.program, 'uOpacity'),
    };
    this.states = new Map();
    this.yaw = -0.28;
    this.pitch = 0.22;
    this.dragging = false;
    this.lastPointer = null;
    gl.clearColor(0, 0, 0, 1);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.CULL_FACE);
    this.bindPointerControls();
  }

  bindPointerControls() {
    this.canvas.addEventListener('pointerdown', (event) => {
      this.dragging = true;
      this.lastPointer = { x: event.clientX, y: event.clientY };
      this.canvas.setPointerCapture?.(event.pointerId);
    });
    this.canvas.addEventListener('pointermove', (event) => {
      if (!this.dragging || !this.lastPointer) return;
      this.yaw += (event.clientX - this.lastPointer.x) * 0.008;
      this.pitch = Math.max(-1.1, Math.min(1.1, this.pitch + (event.clientY - this.lastPointer.y) * 0.008));
      this.lastPointer = { x: event.clientX, y: event.clientY };
    });
    const stop = () => { this.dragging = false; this.lastPointer = null; };
    this.canvas.addEventListener('pointerup', stop);
    this.canvas.addEventListener('pointercancel', stop);
  }

  createHandState(hand) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    const positionBuffer = gl.createBuffer();
    const normalBuffer = gl.createBuffer();
    const indexBuffer = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, hand.vertices.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, hand.vertices.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, hand.faces, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    return {
      hand,
      vao,
      positionBuffer,
      normalBuffer,
      indexBuffer,
      deformer: new ManoHandDeformer(hand),
      interpolator: new LandmarkInterpolator({ halfLifeMs: 26 }),
      normals: new Float32Array(hand.vertices.length),
      lastSeen: -Infinity,
      targetAvailable: false,
      scale: meshScale(hand.vertices),
    };
  }

  setBundle(bundle) {
    this.disposeStates();
    if (!bundle) return;
    this.states.set('left', this.createHandState(bundle.hands.left));
    this.states.set('right', this.createHandState(bundle.hands.right));
  }

  updateHands(hands, timestamp = performance.now()) {
    const seen = new Set();
    for (const hand of hands ?? []) {
      const side = hand.handedness === 'left' ? 'left' : 'right';
      if (seen.has(side)) continue;
      const state = this.states.get(side);
      if (!state) continue;
      try {
        state.interpolator.setTarget(landmarksFromHand(hand), timestamp);
        state.lastSeen = timestamp;
        state.targetAvailable = true;
        seen.add(side);
      } catch {
        // Ignore malformed individual detections while preserving the previous valid target.
      }
    }
  }

  render(timestamp = performance.now()) {
    const gl = this.gl;
    const { width, height } = fitCanvas(this.canvas, gl);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (this.states.size === 0) return;
    const active = [...this.states.entries()].filter(([, state]) => state.targetAvailable && timestamp - state.lastSeen < 260);
    if (active.length === 0) return;

    const projection = perspective(Math.PI / 4.8, width / height, 0.1, 20);
    const viewProjection = multiply4(projection, translation4(0, 0, -3.15));
    const rotation = rotation3(this.yaw, this.pitch);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.locations.viewProjection, false, viewProjection);
    gl.uniformMatrix3fv(this.locations.rotation, false, rotation);

    active.forEach(([side, state], index) => {
      const landmarks = state.interpolator.sample(timestamp);
      if (!landmarks) return;
      const vertices = state.deformer.deform(landmarks);
      computeVertexNormals(vertices, state.hand.faces, state.normals);
      gl.bindBuffer(gl.ARRAY_BUFFER, state.positionBuffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, vertices);
      gl.bindBuffer(gl.ARRAY_BUFFER, state.normalBuffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, state.normals);
      gl.bindVertexArray(state.vao);
      const offsetX = active.length === 1 ? 0 : index === 0 ? -0.58 : 0.58;
      gl.uniform3f(this.locations.center, landmarks[0], landmarks[1], landmarks[2]);
      gl.uniform3f(this.locations.offset, offsetX, -0.08, 0);
      gl.uniform1f(this.locations.scale, state.scale);
      if (side === 'left') gl.uniform3f(this.locations.color, 0.48, 0.72, 0.86);
      else gl.uniform3f(this.locations.color, 0.92, 0.66, 0.46);
      gl.uniform1f(this.locations.opacity, Math.min(1, Math.max(0, (260 - (timestamp - state.lastSeen)) / 90)));
      gl.drawElements(gl.TRIANGLES, state.hand.faces.length, gl.UNSIGNED_SHORT, 0);
    });
    gl.bindVertexArray(null);
  }

  resetTracking() {
    for (const state of this.states.values()) {
      state.interpolator.reset();
      state.targetAvailable = false;
      state.lastSeen = -Infinity;
    }
  }

  disposeStates() {
    const gl = this.gl;
    for (const state of this.states.values()) {
      gl.deleteBuffer(state.positionBuffer);
      gl.deleteBuffer(state.normalBuffer);
      gl.deleteBuffer(state.indexBuffer);
      gl.deleteVertexArray(state.vao);
    }
    this.states.clear();
  }

  dispose() {
    this.disposeStates();
    this.gl.deleteProgram(this.program);
  }
}
