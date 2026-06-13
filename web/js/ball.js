// 3D foil football for the hero. three.js via importmap.
// Drag to spin, gentle auto-rotate, vertical float, cursor parallax.
// Respects prefers-reduced-motion (freezes motion). Disposes cleanly.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export function mountBall(container, { reducedMotion = false } = {}) {
  if (!container) return () => {};

  const size = () => Math.min(container.clientWidth, container.clientHeight) || 360;

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(0, 0, 6);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(size(), size());
  container.appendChild(renderer.domElement);

  // --- Ball: white base sphere + subtle pentagon/hex hint via dark patches ---
  const group = new THREE.Group();
  scene.add(group);

  const ballGeo = new THREE.IcosahedronGeometry(1.6, 5);
  const ballMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    metalness: 0.35,
    roughness: 0.22,
    envMapIntensity: 1.1,
  });
  const ball = new THREE.Mesh(ballGeo, ballMat);
  group.add(ball);

  // Pentagon hints: small dark dodecahedron-vertex patches give a "panel" feel
  // without modelling a true Telstar. Lightweight.
  const patchGeo = new THREE.CircleGeometry(0.42, 5); // pentagon-ish
  const patchMat = new THREE.MeshStandardMaterial({
    color: 0x2b2a26,
    metalness: 0.3,
    roughness: 0.4,
    side: THREE.DoubleSide,
  });
  // Dodecahedron vertices ~ centres of a football's 12 pentagons.
  const phi = (1 + Math.sqrt(5)) / 2;
  const pentCentres = [];
  [[0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
   [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
   [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]]
    .forEach((v) => pentCentres.push(new THREE.Vector3(...v).normalize()));

  pentCentres.forEach((dir) => {
    const patch = new THREE.Mesh(patchGeo, patchMat);
    patch.position.copy(dir.clone().multiplyScalar(1.605));
    patch.lookAt(dir.clone().multiplyScalar(3));
    group.add(patch);
  });

  // Gold-foil rim glow sprite behind the ball
  const glowMat = new THREE.MeshBasicMaterial({ color: 0xe8c66b, transparent: true, opacity: 0.18 });
  const glow = new THREE.Mesh(new THREE.SphereGeometry(1.95, 32, 32), glowMat);
  glow.material.side = THREE.BackSide;
  group.add(glow);

  // --- Lighting: soft key + warm fill so it glints like foil ---
  scene.add(new THREE.AmbientLight(0xfff4d6, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(3, 4, 5);
  scene.add(key);
  const warm = new THREE.PointLight(0xffd27a, 1.1, 30);
  warm.position.set(-4, -2, 3);
  scene.add(warm);
  const rim = new THREE.PointLight(0x3da35d, 0.6, 30);
  rim.position.set(0, 3, -4);
  scene.add(rim);

  // --- Controls: rotate only ---
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.enableRotate = !reducedMotion;
  controls.autoRotate = !reducedMotion;
  controls.autoRotateSpeed = 1.1;
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  // Cursor parallax
  const parallax = { x: 0, y: 0 };
  function onPointerMove(e) {
    const nx = (e.clientX / window.innerWidth) * 2 - 1;
    const ny = (e.clientY / window.innerHeight) * 2 - 1;
    parallax.x = nx * 0.18;
    parallax.y = -ny * 0.12;
  }
  if (!reducedMotion) window.addEventListener("pointermove", onPointerMove, { passive: true });

  function resize() {
    const s = size();
    renderer.setSize(s, s);
    camera.aspect = 1;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);

  const clock = new THREE.Clock();
  let raf = 0;
  let running = true;

  function frame() {
    if (!running) return;
    raf = requestAnimationFrame(frame);
    const t = clock.getElapsedTime();
    if (!reducedMotion) {
      group.position.y = Math.sin(t * 1.1) * 0.12; // float bob
      group.rotation.x = parallax.y;
      group.rotation.z = parallax.x * 0.3;
    }
    controls.update();
    renderer.render(scene, camera);
  }
  frame();

  // Pause when offscreen to save the GPU.
  const io = new IntersectionObserver((entries) => {
    running = entries[0].isIntersecting;
    if (running) { clock.start(); frame(); }
  }, { threshold: 0.05 });
  io.observe(container);

  // Dispose handle
  return function dispose() {
    running = false;
    cancelAnimationFrame(raf);
    io.disconnect();
    ro.disconnect();
    window.removeEventListener("pointermove", onPointerMove);
    controls.dispose();
    ballGeo.dispose(); ballMat.dispose();
    patchGeo.dispose(); patchMat.dispose();
    glow.geometry.dispose(); glowMat.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
  };
}
