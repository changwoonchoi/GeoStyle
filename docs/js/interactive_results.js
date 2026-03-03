import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

// ─── Example Data ────────────────────────────────────────────────
const EXAMPLES = [
  {
    name: 'Beagle',
    source: './assets/results/source_meshes/beagle.obj',
    style: './assets/results/style_images/moon_gazing_hare.jpg',
    deformed: './assets/results/deformed_meshes/ours/beagle_moon_gazing_hare.obj',
  },
  {
    name: 'Hand',
    source: './assets/results/source_meshes/hand.obj',
    style: './assets/results/style_images/maman.jpg',
    deformed: './assets/results/deformed_meshes/ours/hand_maman.obj',
  },
  {
    name: 'Homer',
    source: './assets/results/source_meshes/homer.obj',
    style: './assets/results/style_images/hydrant.jpg',
    deformed: './assets/results/deformed_meshes/ours/homer_hydrant.obj',
  },
  {
    name: 'Eiffel Tower',
    source: './assets/results/source_meshes/eiffel_tower.obj',
    style: './assets/results/style_images/stones.png',
    deformed: './assets/results/deformed_meshes/ours/eiffel_tower_stones.obj',
  },
  {
    name: 'Crab',
    source: './assets/results/source_meshes/crab.obj',
    style: './assets/results/style_images/koons.jpg',
    deformed: './assets/results/deformed_meshes/ours/crab_koons.obj',
  },
  {
    name: 'Big Ben',
    source: './assets/results/source_meshes/bigben.obj',
    style: './assets/results/style_images/brancusi.png',
    deformed: './assets/results/deformed_meshes/ours/bigben_brancusi.obj',
  }
];

// ─── DOM ─────────────────────────────────────────────────────────
const sourceContainer = document.getElementById('viewer-source');
const deformedContainer = document.getElementById('viewer-deformed');
const styleImage = document.getElementById('style-image');
const loadingOverlay = document.getElementById('results-loading');
const tabsContainer = document.getElementById('example-tabs');

// ─── State ───────────────────────────────────────────────────────
let sourceScene, sourceCamera, sourceRenderer;
let deformedScene, deformedCamera, deformedRenderer;
let currentSourceMesh = null;
let currentDeformedMesh = null;
const meshCache = new Map();

// ─── Shared Orbit State ──────────────────────────────────────────
// Single source of truth for camera orientation — guarantees perfect sync.
const orbit = {
  theta: Math.PI / 4,   // horizontal angle
  phi: Math.PI / 3,     // vertical angle (from top)
  radius: 6.0,          // distance from origin
  target: new THREE.Vector3(0, 0, 0),
  // Damping
  thetaVel: 0,
  phiVel: 0,
  radiusVel: 0,
  damping: 0.90,
};

const ORBIT_SPEED = 0.006;
const ZOOM_SPEED = 0.003;
const PHI_MIN = 0.05;
const PHI_MAX = Math.PI - 0.05;
const RADIUS_MIN = 1.5;
const RADIUS_MAX = 20;

let isDragging = false;
let lastPointer = { x: 0, y: 0 };

// ─── Init (wait for DOM) ─────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  try {
    init();
    animate();
  } catch (e) {
    console.error('GeoStyle Viewer Init Error:', e);
  }
});

function createViewer(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xfafafa);

  const w = container.clientWidth || 350;
  const h = container.clientHeight || 350;
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.VSMShadowMap;
  container.appendChild(renderer.domElement);

  scene.add(camera);

  // Hemisphere light
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x888888, 1.0);
  hemiLight.position.set(0, 20, 0);
  camera.add(hemiLight);

  // Key light (softer, neutral white)
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(5, 10, 7);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 1024;
  dirLight.shadow.mapSize.height = 1024;
  dirLight.shadow.bias = -0.0001;
  dirLight.shadow.radius = 15;
  dirLight.shadow.blurSamples = 20;
  camera.add(dirLight);

  // Fill light (side)
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
  fillLight.position.set(-5, 5, 0);
  camera.add(fillLight);

  // Bottom fill (weaker, to allow some shadowing underneath)
  const bottomLight = new THREE.DirectionalLight(0xffffff, 0.3);
  bottomLight.position.set(0, -10, 0);
  camera.add(bottomLight);

  // Ambient (higher to soften shadows)
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  camera.add(ambientLight);

  // Ground shadow plane
  const groundGeo = new THREE.PlaneGeometry(100, 100);
  const groundMat = new THREE.ShadowMaterial({ opacity: 0.25 });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  ground.name = 'Ground';
  camera.add(ground);

  return { scene, camera, renderer };
}

// ─── Shared orbit controls (pointer events) ─────────────────────
function attachOrbitListeners(el) {
  el.addEventListener('pointerdown', (e) => {
    isDragging = true;
    lastPointer.x = e.clientX;
    lastPointer.y = e.clientY;
    el.setPointerCapture(e.pointerId);
  });

  el.addEventListener('pointermove', (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastPointer.x;
    const dy = e.clientY - lastPointer.y;
    lastPointer.x = e.clientX;
    lastPointer.y = e.clientY;

    orbit.thetaVel = -dx * ORBIT_SPEED;
    orbit.phiVel = -dy * ORBIT_SPEED;
    orbit.theta += orbit.thetaVel;
    orbit.phi = Math.max(PHI_MIN, Math.min(PHI_MAX, orbit.phi + orbit.phiVel));
  });

  el.addEventListener('pointerup', (e) => {
    isDragging = false;
    el.releasePointerCapture(e.pointerId);
  });

  el.addEventListener('pointercancel', (e) => {
    isDragging = false;
  });

  el.addEventListener('wheel', (e) => {
    e.preventDefault();
    orbit.radiusVel += e.deltaY * ZOOM_SPEED;
  }, { passive: false });

  // Prevent context menu on right-click drag
  el.addEventListener('contextmenu', (e) => e.preventDefault());
}

function updateCamerasFromOrbit() {
  // Apply damping when not dragging
  if (!isDragging) {
    orbit.thetaVel *= orbit.damping;
    orbit.phiVel *= orbit.damping;
  }
  orbit.radiusVel *= orbit.damping;

  // Apply velocities
  if (!isDragging) {
    orbit.theta += orbit.thetaVel;
    orbit.phi = Math.max(PHI_MIN, Math.min(PHI_MAX, orbit.phi + orbit.phiVel));
  }
  orbit.radius = Math.max(RADIUS_MIN, Math.min(RADIUS_MAX, orbit.radius + orbit.radiusVel));

  // Spherical to Cartesian
  const x = orbit.radius * Math.sin(orbit.phi) * Math.sin(orbit.theta);
  const y = orbit.radius * Math.cos(orbit.phi);
  const z = orbit.radius * Math.sin(orbit.phi) * Math.cos(orbit.theta);

  // Apply to BOTH cameras identically
  [sourceCamera, deformedCamera].forEach((cam) => {
    if (!cam) return;
    cam.position.set(
      orbit.target.x + x,
      orbit.target.y + y,
      orbit.target.z + z
    );
    cam.lookAt(orbit.target);
  });
}

// ─── Init ────────────────────────────────────────────────────────
function init() {
  if (!sourceContainer || !deformedContainer) {
    console.error('Viewer containers not found');
    return;
  }

  const src = createViewer(sourceContainer);
  sourceScene = src.scene;
  sourceCamera = src.camera;
  sourceRenderer = src.renderer;

  const def = createViewer(deformedContainer);
  deformedScene = def.scene;
  deformedCamera = def.camera;
  deformedRenderer = def.renderer;

  // Attach shared orbit controls to BOTH canvases
  attachOrbitListeners(src.renderer.domElement);
  attachOrbitListeners(def.renderer.domElement);

  // Set initial camera positions
  updateCamerasFromOrbit();

  // Resize
  window.addEventListener('resize', onWindowResize);

  // Setup tabs
  setupTabs();

  // Load first example
  loadExample(0);
}

function setupTabs() {
  if (!tabsContainer) return;
  const ul = tabsContainer.querySelector('ul');
  ul.innerHTML = '';

  EXAMPLES.forEach((ex, i) => {
    const li = document.createElement('li');
    if (i === 0) li.classList.add('is-active');
    li.dataset.index = i;
    li.innerHTML = `<a><span>${ex.name}</span></a>`;
    li.addEventListener('click', () => {
      ul.querySelectorAll('li').forEach(l => l.classList.remove('is-active'));
      li.classList.add('is-active');
      loadExample(i);
    });
    ul.appendChild(li);
  });
}

function loadExample(index) {
  const ex = EXAMPLES[index];

  // Update style image
  if (styleImage) {
    styleImage.src = ex.style;
    styleImage.alt = ex.name + ' style reference';
  }

  // Load meshes
  showLoading(true);
  let loaded = 0;
  const onLoaded = () => {
    loaded++;
    if (loaded >= 2) showLoading(false);
  };

  loadMesh(ex.source, sourceScene, 'source', (mesh) => {
    if (currentSourceMesh) sourceScene.remove(currentSourceMesh);
    currentSourceMesh = mesh;
    sourceScene.add(mesh);
    onLoaded();
  });

  loadMesh(ex.deformed, deformedScene, 'deformed', (mesh) => {
    if (currentDeformedMesh) deformedScene.remove(currentDeformedMesh);
    currentDeformedMesh = mesh;
    deformedScene.add(mesh);
    onLoaded();
  });
}

function loadMesh(url, scene, tag, callback) {
  if (meshCache.has(url)) {
    callback(meshCache.get(url));
    return;
  }

  const loader = new OBJLoader();
  loader.load(url, (object) => {
    processMesh(object);
    meshCache.set(url, object);
    callback(object);
  }, undefined, (err) => {
    console.error(`Error loading ${tag} mesh:`, err);
    showLoading(false);
  });
}

function processMesh(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const maxDim = Math.max(size.x, size.y, size.z);
  if (maxDim > 0) {
    const scaleFactor = 3.0 / maxDim;
    object.scale.setScalar(scaleFactor);
    object.position.copy(center).multiplyScalar(-scaleFactor);
  } else {
    object.position.sub(center);
  }

  object.traverse((child) => {
    if (child.isMesh) {
      const oldMat = child.material;
      child.material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x9bc2d5), // Light powder blue/slate from teaser
        side: THREE.DoubleSide,
        roughness: 0.85,  // Very matte, like modeling clay
        metalness: 0.0,   // No plastic shiny highlights
        flatShading: false,
      });
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });
}

function showLoading(show) {
  if (loadingOverlay) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
  }
}

function onWindowResize() {
  [
    [sourceCamera, sourceRenderer, sourceContainer],
    [deformedCamera, deformedRenderer, deformedContainer],
  ].forEach(([cam, ren, el]) => {
    if (!cam || !ren || !el) return;
    const aspect = el.clientWidth / el.clientHeight;
    cam.aspect = aspect;
    cam.updateProjectionMatrix();
    ren.setSize(el.clientWidth, el.clientHeight);
  });
}

function animate() {
  requestAnimationFrame(animate);

  // Update BOTH cameras from the shared orbit state
  updateCamerasFromOrbit();

  // Update ground position
  [sourceCamera, deformedCamera].forEach((cam) => {
    if (!cam) return;
    const ground = cam.getObjectByName('Ground');
    if (ground) {
      ground.position.z = -orbit.radius;
      ground.position.y = -2.0;
    }
  });

  if (sourceRenderer && sourceScene) sourceRenderer.render(sourceScene, sourceCamera);
  if (deformedRenderer && deformedScene) deformedRenderer.render(deformedScene, deformedCamera);
}
