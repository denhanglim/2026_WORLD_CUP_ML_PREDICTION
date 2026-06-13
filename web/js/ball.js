// Glossy classic football for the hero — a real Telstar-style ball:
// white body with 12 black pentagons, physical clearcoat material lit by a
// procedural studio environment so it actually shines, soft bloom, a contact
// shadow on the paper, gentle float + cursor parallax + drag-to-spin.
// Respects prefers-reduced-motion (freezes motion). Disposes cleanly.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export function mountBall(container, { reducedMotion = false } = {}) {
  if (!container) return () => {};

  const size = () => Math.min(container.clientWidth, container.clientHeight) || 360;

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0.2, 0.25, 6.2);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, premultipliedAlpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(size(), size());
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // --- Studio environment for crisp reflections (gives the gloss) ---
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
  scene.environment = envRT.texture;

  const group = new THREE.Group();
  scene.add(group);
  const RADIUS = 1.55;

  // --- Dodecahedron geometry: its 12 faces ARE the pentagons of a football ---
  const dodeca = new THREE.DodecahedronGeometry(RADIUS);
  const pos = dodeca.attributes.position;

  // Recover the 12 face centres (face normals) by clustering vertices.
  const faceNormals = [];
  const phi = (1 + Math.sqrt(5)) / 2;
  // Dodecahedron face centres == icosahedron vertices direction set.
  [
    [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
    [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
  ].forEach((v) => faceNormals.push(new THREE.Vector3(...v).normalize()));

  // White glossy leather body — slightly inflated sphere so the pentagons sit
  // recessed, like stitched panels.
  const bodyGeo = new THREE.IcosahedronGeometry(RADIUS, 8);
  const bodyMat = new THREE.MeshPhysicalMaterial({
    color: 0xf6f4ee,
    metalness: 0.0,
    roughness: 0.18,
    clearcoat: 1.0,
    clearcoatRoughness: 0.12,
    sheen: 0.4,
    sheenColor: new THREE.Color(0xffffff),
    envMapIntensity: 1.25,
  });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.castShadow = true;
  group.add(body);

  // Black pentagon panels: a real pentagon face slightly above the surface,
  // with its own glossy black material. 12 of them, oriented to each face.
  const pentShape = (() => {
    const s = new THREE.Shape();
    const r = 0.55;
    for (let i = 0; i < 5; i++) {
      const a = (Math.PI / 2) + (i * 2 * Math.PI) / 5;
      const x = Math.cos(a) * r;
      const y = Math.sin(a) * r;
      i === 0 ? s.moveTo(x, y) : s.lineTo(x, y);
    }
    s.closePath();
    return s;
  })();
  const pentGeo = new THREE.ExtrudeGeometry(pentShape, {
    depth: 0.02, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.04, bevelSegments: 4, steps: 1,
  });
  pentGeo.center();
  const pentMat = new THREE.MeshPhysicalMaterial({
    color: 0x14130f,
    metalness: 0.0,
    roughness: 0.28,
    clearcoat: 0.9,
    clearcoatRoughness: 0.18,
    envMapIntensity: 1.0,
  });

  const pentMeshes = [];
  const up = new THREE.Vector3(0, 0, 1);
  faceNormals.forEach((n) => {
    const pent = new THREE.Mesh(pentGeo, pentMat);
    // Seat flush against the body surface, facing outward, so the silhouette
    // stays a clean round ball (no protruding studs at the rim).
    const seat = RADIUS * 0.955;
    pent.position.copy(n.clone().multiplyScalar(seat));
    const q = new THREE.Quaternion().setFromUnitVectors(up, n);
    pent.quaternion.copy(q);
    pent.castShadow = true;
    group.add(pent);
    pentMeshes.push(pent);
  });

  // Subtle warm rim glow shell to feed bloom and lift the ball off the paper.
  const glowMat = new THREE.MeshBasicMaterial({
    color: 0xfff0c8, transparent: true, opacity: 0.10, side: THREE.BackSide,
  });
  const glow = new THREE.Mesh(new THREE.SphereGeometry(RADIUS * 1.22, 48, 48), glowMat);
  group.add(glow);

  // --- Soft contact shadow on the "paper": a blurred radial sprite ---
  const shadowTex = (() => {
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d");
    const grad = g.createRadialGradient(128, 128, 8, 128, 128, 128);
    grad.addColorStop(0, "rgba(43,42,38,0.42)");
    grad.addColorStop(0.55, "rgba(43,42,38,0.16)");
    grad.addColorStop(1, "rgba(43,42,38,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 256);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  })();
  const contact = new THREE.Mesh(
    new THREE.PlaneGeometry(4.4, 1.9),
    new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false }),
  );
  contact.position.set(0, -RADIUS - 0.62, -0.2);
  contact.rotation.x = -Math.PI / 2.05;
  contact.renderOrder = -1;
  scene.add(contact);

  // --- Lighting: a bright key for the specular hotspot + warm + cool kicker ---
  scene.add(new THREE.AmbientLight(0xfff6e6, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(3.2, 4.5, 4.0);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 14;
  key.shadow.bias = -0.0006;
  scene.add(key);
  const warm = new THREE.PointLight(0xffcf7a, 1.1, 30);
  warm.position.set(-4, -1.5, 3);
  scene.add(warm);
  const kick = new THREE.PointLight(0x6fc3e6, 0.7, 30);
  kick.position.set(-2.5, 3.5, -4);
  scene.add(kick);

  // --- Controls: drag to spin only ---
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.enableRotate = !reducedMotion;
  controls.autoRotate = !reducedMotion;
  controls.autoRotateSpeed = 0.9;
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);

  // Tilt the ball a little so a pentagon faces the viewer.
  group.rotation.set(0.35, 0.5, 0.1);

  // The "bloom"/glow is done with the transparent backside glow shell above
  // plus a CSS radial halo behind the canvas (.ball-wrap::before). Avoiding
  // EffectComposer keeps the canvas alpha clean (no black box over the paper).

  // Cursor parallax
  const parallax = { x: 0, y: 0 };
  function onPointerMove(e) {
    const nx = (e.clientX / window.innerWidth) * 2 - 1;
    const ny = (e.clientY / window.innerHeight) * 2 - 1;
    parallax.x = nx * 0.22;
    parallax.y = -ny * 0.16;
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
  const baseRot = group.rotation.clone();

  function frame() {
    if (!running) return;
    raf = requestAnimationFrame(frame);
    const t = clock.getElapsedTime();
    if (!reducedMotion) {
      group.position.y = Math.sin(t * 1.1) * 0.10; // float bob
      group.rotation.x = baseRot.x + parallax.y;
      group.rotation.z = baseRot.z + parallax.x * 0.25;
      const sShadow = 1 - Math.abs(Math.sin(t * 1.1)) * 0.12;
      contact.scale.set(sShadow, sShadow, sShadow);
      contact.material.opacity = 0.85 - Math.abs(Math.sin(t * 1.1)) * 0.18;
    }
    controls.update();
    renderer.render(scene, camera);
  }
  frame();

  // Render one static frame even under reduced motion.
  if (reducedMotion) renderer.render(scene, camera);

  // Pause when offscreen to save the GPU.
  const io = new IntersectionObserver((entries) => {
    running = entries[0].isIntersecting && !reducedMotion;
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
    bodyGeo.dispose(); bodyMat.dispose();
    pentGeo.dispose(); pentMat.dispose();
    dodeca.dispose();
    glow.geometry.dispose(); glowMat.dispose();
    contact.geometry.dispose(); contact.material.dispose(); shadowTex.dispose();
    envRT.dispose(); pmrem.dispose();
    renderer.dispose();
    if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
  };
}
