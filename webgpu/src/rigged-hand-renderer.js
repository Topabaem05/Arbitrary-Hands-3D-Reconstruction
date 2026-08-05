import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

import {
  BONE_NAMES,
  HandLandmarkSmoother,
  landmarksFromHand,
  solveRigPose,
} from './hand-retarget.js';

export const REQUIRED_HAND_BONES = Object.freeze([...BONE_NAMES]);

function findRigParts(root) {
  let mesh = null;
  const bones = new Map();
  root.traverse((object) => {
    if (!mesh && object.isSkinnedMesh) mesh = object;
    if (object.isBone) bones.set(object.name, object);
  });
  return { mesh, bones };
}

export function validateRiggedHand(root) {
  if (!root?.traverse) throw new TypeError('Rigged hand root must be a Three.js Object3D.');
  const { mesh, bones } = findRigParts(root);
  if (!mesh) throw new TypeError('Rigged hand GLB must contain a SkinnedMesh.');
  const missing = BONE_NAMES.filter((name) => !bones.has(name));
  if (missing.length > 0) {
    throw new TypeError(`Missing required hand bones: ${missing.join(', ')}`);
  }
  const skinIndex = mesh.geometry.getAttribute('skinIndex');
  const skinWeight = mesh.geometry.getAttribute('skinWeight');
  if (!skinIndex || skinIndex.itemSize !== 4 || !skinWeight || skinWeight.itemSize !== 4) {
    throw new TypeError(
      'Rigged hand mesh must contain four-component skinIndex and skinWeight attributes.',
    );
  }
  if (!mesh.skeleton || mesh.skeleton.bones.length < 21) {
    throw new TypeError('Rigged hand skin is incomplete.');
  }
  return { mesh, bones };
}

function parseGlb(loader, bytes) {
  return new Promise((resolve, reject) => {
    loader.parse(bytes, '', resolve, reject);
  });
}

function restPositionsFor(root, bones) {
  root.updateMatrixWorld(true);
  const inverseRoot = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const result = new Float32Array(63);
  const point = new THREE.Vector3();
  for (let index = 0; index < BONE_NAMES.length; index += 1) {
    bones.get(BONE_NAMES[index]).getWorldPosition(point).applyMatrix4(inverseRoot);
    result[index * 3] = point.x;
    result[index * 3 + 1] = point.y;
    result[index * 3 + 2] = point.z;
  }
  return result;
}

function disposeObject(root) {
  const geometries = new Set();
  const materials = new Set();
  root.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    const values = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of values) if (material) materials.add(material);
  });
  for (const geometry of geometries) geometry.dispose?.();
  for (const material of materials) material.dispose?.();
}

function disposeSlot(slot) {
  slot.helper?.geometry?.dispose?.();
  slot.helper?.material?.dispose?.();
}

function overrideMaterials(root) {
  const oldMaterials = new Set();
  const material = new THREE.MeshStandardMaterial({
    color: 0xc9ced8,
    roughness: 0.48,
    metalness: 0.03,
    side: THREE.DoubleSide,
  });
  root.traverse((object) => {
    if (object.isSkinnedMesh) {
      const values = Array.isArray(object.material) ? object.material : [object.material];
      for (const value of values) if (value) oldMaterials.add(value);
      object.material = material;
      object.frustumCulled = false;
      object.castShadow = false;
      object.receiveShadow = false;
    }
  });
  for (const oldMaterial of oldMaterials) oldMaterial.dispose?.();
}

function makeRigInstance(template, baseScale, debugBones) {
  const root = SkeletonUtils.clone(template);
  const { mesh, bones } = validateRiggedHand(root);
  const restPositions = restPositionsFor(root, bones);
  const restQuaternions = new Map(
    BONE_NAMES.map((name) => [name, bones.get(name).quaternion.clone()]),
  );
  const modelGroup = new THREE.Group();
  modelGroup.name = 'RiggedHandModel';
  modelGroup.scale.setScalar(baseScale);
  modelGroup.add(root);
  const poseGroup = new THREE.Group();
  poseGroup.name = 'RiggedHandPose';
  poseGroup.add(modelGroup);
  const slotGroup = new THREE.Group();
  slotGroup.visible = false;
  slotGroup.add(poseGroup);
  const helper = debugBones ? new THREE.SkeletonHelper(root) : null;
  if (helper) {
    helper.material.depthTest = false;
    helper.renderOrder = 10;
    slotGroup.add(helper);
  }
  return {
    slotGroup,
    poseGroup,
    modelGroup,
    root,
    mesh,
    bones,
    restPositions,
    restQuaternions,
    smoother: new HandLandmarkSmoother({ halfLifeMs: 28 }),
    handedness: 'left',
    lastTargetTime: -Infinity,
    helper,
  };
}

export class RiggedHandRenderer {
  constructor(canvas, { maxHands = 1, debugBones = false } = {}) {
    if (!canvas) throw new TypeError('A canvas is required.');
    this.canvas = canvas;
    this.maxHands = Math.max(1, Math.min(2, Number(maxHands) || 1));
    this.debugBones = Boolean(debugBones);
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100);
    this.camera.position.set(0, 0, 5.2);
    this.camera.lookAt(0, 0, 0);
    this.viewGroup = new THREE.Group();
    this.viewGroup.rotation.set(-0.15, -0.32, 0.05);
    this.scene.add(this.viewGroup);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x111522, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 3.1);
    key.position.set(-2.5, 3.5, 4.5);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x6ab8ff, 1.8);
    rim.position.set(3.5, -1.5, -3.5);
    this.scene.add(rim);

    this.loader = new GLTFLoader();
    this.template = null;
    this.slots = [];
    this.dragging = false;
    this.lastPointer = null;
    this.resizeObserver = typeof ResizeObserver !== 'undefined'
      ? new ResizeObserver(() => this.resize())
      : null;
    this.resizeObserver?.observe(canvas);
    this.bindPointerControls();
    this.resize();
  }

  bindPointerControls() {
    this.canvas.addEventListener('pointerdown', (event) => {
      this.dragging = true;
      this.lastPointer = { x: event.clientX, y: event.clientY };
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener('pointermove', (event) => {
      if (!this.dragging || !this.lastPointer) return;
      this.viewGroup.rotation.y += (event.clientX - this.lastPointer.x) * 0.008;
      this.viewGroup.rotation.x += (event.clientY - this.lastPointer.y) * 0.008;
      this.lastPointer = { x: event.clientX, y: event.clientY };
    });
    const stop = () => {
      this.dragging = false;
      this.lastPointer = null;
    };
    this.canvas.addEventListener('pointerup', stop);
    this.canvas.addEventListener('pointercancel', stop);
  }

  resize() {
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, this.canvas.clientHeight);
    const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  async load(bytes) {
    const gltf = await parseGlb(this.loader, bytes);
    validateRiggedHand(gltf.scene);
    overrideMaterials(gltf.scene);
    gltf.scene.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maximum = Math.max(size.x, size.y, size.z) || 1;
    gltf.scene.position.sub(center);
    const baseScale = 2.75 / maximum;

    for (const slot of this.slots) {
      this.viewGroup.remove(slot.slotGroup);
      disposeSlot(slot);
    }
    if (this.template) disposeObject(this.template);
    this.template = gltf.scene;
    this.slots = [];
    for (let index = 0; index < this.maxHands; index += 1) {
      const slot = makeRigInstance(
        this.template,
        baseScale * (this.maxHands === 2 ? 0.78 : 1),
        this.debugBones,
      );
      slot.slotGroup.position.x = this.maxHands === 2
        ? (index === 0 ? -1.25 : 1.25)
        : 0;
      this.slots.push(slot);
      this.viewGroup.add(slot.slotGroup);
    }
    return {
      triangleCount: this.slots[0].mesh.geometry.index?.count / 3 || 0,
      boneCount: BONE_NAMES.length,
    };
  }

  setHands(hands = [], timestamp = performance.now()) {
    for (let index = 0; index < this.slots.length; index += 1) {
      const slot = this.slots[index];
      const hand = hands[index];
      if (!hand) continue;
      slot.handedness = hand.handedness || 'left';
      slot.smoother.setTarget(landmarksFromHand(hand), timestamp);
      slot.lastTargetTime = timestamp;
      slot.slotGroup.visible = true;
    }
  }

  applySlotPose(slot, timestamp) {
    if (timestamp - slot.lastTargetTime > 350) {
      slot.slotGroup.visible = false;
      return;
    }
    const landmarks = slot.smoother.sample(timestamp);
    if (!landmarks) return;
    const pose = solveRigPose(slot.restPositions, landmarks, slot.handedness);
    slot.poseGroup.quaternion.fromArray(pose.rootQuaternion);
    slot.modelGroup.scale.y = Math.abs(slot.modelGroup.scale.y) * (pose.mirrorY ? -1 : 1);
    for (let index = 0; index < BONE_NAMES.length; index += 1) {
      const name = BONE_NAMES[index];
      const bone = slot.bones.get(name);
      const rest = slot.restQuaternions.get(name);
      const offset = index * 4;
      const delta = new THREE.Quaternion(
        pose.boneQuaternions[offset],
        pose.boneQuaternions[offset + 1],
        pose.boneQuaternions[offset + 2],
        pose.boneQuaternions[offset + 3],
      );
      bone.quaternion.copy(rest).multiply(delta);
    }
    slot.root.updateMatrixWorld(true);
    slot.helper?.update();
  }

  render(timestamp = performance.now()) {
    for (const slot of this.slots) this.applySlotPose(slot, timestamp);
    this.renderer.render(this.scene, this.camera);
  }

  reset() {
    for (const slot of this.slots) {
      slot.smoother.reset();
      slot.slotGroup.visible = false;
      slot.lastTargetTime = -Infinity;
      for (const name of BONE_NAMES) {
        slot.bones.get(name).quaternion.copy(slot.restQuaternions.get(name));
      }
    }
  }

  dispose() {
    this.resizeObserver?.disconnect();
    for (const slot of this.slots) disposeSlot(slot);
    if (this.template) disposeObject(this.template);
    this.renderer.dispose();
  }
}
