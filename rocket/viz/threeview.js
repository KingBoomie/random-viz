// Minimal Three.js scene to show rocket pose and snapshots
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

export function setupThree(canvas) {
  // Use Z-up to match sim coordinates (Three constant is DEFAULT_UP)
  if (THREE.Object3D && THREE.Object3D.DEFAULT_UP) {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
  }
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  const width = canvas.clientWidth || canvas.parentElement.clientWidth || 480;
  const height = canvas.clientHeight || 320;
  renderer.setSize(width, height, false);

  const scene = new THREE.Scene();
  // Light background for better contrast
  scene.background = new THREE.Color(0xf5f7fb);
  scene.fog = new THREE.Fog(0xf5f7fb, 80, 1200);

  const camera = new THREE.PerspectiveCamera(50, width/height, 0.1, 1000);
  if (!THREE.Object3D.DEFAULT_UP) {
    // Fallback if DEFAULT_UP is unavailable
    camera.up.set(0, 0, 1);
  }
  camera.position.set(6, 6, 6);
  camera.lookAt(0, 0, 0);

  // Lights
  const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 0.8);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 10, 7);
  scene.add(dir);

  // Ground grid
  const grid = new THREE.GridHelper(100, 50, 0x94a3b8, 0xcbd5e1);
  grid.rotation.x = Math.PI / 2; // align xy-plane as ground with z up
  scene.add(grid);

  // --- Earth horizon (simple sphere, diameter 1000 m) ---
  const EARTH_RADIUS = 3000;
  const earthGeom = new THREE.SphereGeometry(EARTH_RADIUS, 64, 48);
  const earthMat = new THREE.MeshStandardMaterial({ color: 0x8db4e8, metalness: 0.0, roughness: 0.9 });
  const earth = new THREE.Mesh(earthGeom, earthMat);
  earth.position.set(0, 0, -EARTH_RADIUS); // tangent at z=0
  scene.add(earth);

  // Rocket body (two cylinders). In Three.js, cylinder axis is along Y; we want along Z.
  const radius = 0.15;
  const segLen = 1.0;
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x93a3b1, metalness: 0.3, roughness: 0.6 });

  const rotZ = new THREE.Euler(0, 0, 0);
  const rotToZ = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI/2, 0, 0)); // rotate cylinder Y-axis to Z-axis

  const engine = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, segLen, 24, 1), bodyMat);
  engine.quaternion.copy(rotToZ);
  engine.position.z = segLen/2;
  const payload = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, segLen, 24, 1), bodyMat);
  payload.quaternion.copy(rotToZ);
  payload.position.z = 1.5*segLen;

  // Nose cone (Three cone axis is along Y; rotate to Z)
  const cone = new THREE.Mesh(new THREE.ConeGeometry(radius*0.95, 0.6, 24), new THREE.MeshStandardMaterial({ color: 0xcbd5e1 }));
  cone.quaternion.copy(rotToZ);
  cone.position.z = 2.0*segLen + 0.3;

  // Fins: thin boxes positioned radially around base
  const finGeom = new THREE.BoxGeometry(0.02, 0.25, 0.12);
  const finMat = new THREE.MeshStandardMaterial({ color: 0x1f2937 });
  const fins = new THREE.Group();
  for (let k = 0; k < 4; k++) {
    const fin = new THREE.Mesh(finGeom, finMat);
    const angle = (k * Math.PI) / 2;
    const r = radius + 0.1;
    fin.position.set(Math.cos(angle) * r, Math.sin(angle) * r, 0.9);
    // orient fin so its long dimension extends outward from body
    fin.lookAt(new THREE.Vector3(0, 0, 0.9));
    fins.add(fin);
  }

  const rocket = new THREE.Group();
  rocket.add(engine);
  rocket.add(payload);
  rocket.add(cone);
  rocket.add(fins);
  if (!THREE.Object3D.DEFAULT_UP) {
    rocket.up.set(0, 0, 1);
  }

  // Pivot so that the rocket origin is at base (z=0)
  rocket.position.set(0, 0, 0);

  scene.add(rocket);

  const axes = new THREE.AxesHelper(1.0);
  rocket.add(axes);

  function render() { renderer.render(scene, camera); }
  render();

  function updateCameraFollow(targetPos) {
    // Keep a fixed offset in rocket body/world coordinates
    const offset = new THREE.Vector3(6, 6, 4);
    // simple smoothing to reduce jitter
    const desired = new THREE.Vector3().copy(targetPos).add(offset);
    camera.position.lerp(desired, 0.25);
    camera.lookAt(targetPos.x, targetPos.y, targetPos.z);
  }

  // --- Clouds: billboard sprites ---
  function makeCloudTexture() {
    const c = document.createElement('canvas');
    c.width = 128; c.height = 128;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(64, 64, 20, 64, 64, 60);
    g.addColorStop(0, 'rgba(255,255,255,0.9)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(64, 64, 60, 0, Math.PI*2);
    ctx.fill();
    const tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
    tex.anisotropy = 2;
    return tex;
  }
  const cloudTex = makeCloudTexture();
  const clouds = new THREE.Group();
  const CLOUD_COUNT = 280;
  for (let i = 0; i < CLOUD_COUNT; i++) {
    const mat = new THREE.SpriteMaterial({ map: cloudTex, transparent: true, opacity: 0.55, depthWrite: false });
    const s = new THREE.Sprite(mat);
    // spread over a wide area to give parallax; random heights
    const R = 1500;
    const x = (Math.random()*2-1) * R;
    const y = (Math.random()*2-1) * R;
    const z = 80 + Math.random()*350; // 80m to 430m
    s.position.set(x, y, z);
    const size = 20 + Math.random()*50; // meters
    s.scale.set(size, size, 1);
    clouds.add(s);
  }
  scene.add(clouds);

  return { scene, camera, renderer, rocket, render, updateCameraFollow };
}

export function updateThreePose(ctx, pos, quat) {
  if (!ctx) return;
  const { rocket, render, updateCameraFollow } = ctx;
  // quat is [w,x,y,z]
  rocket.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
  rocket.position.set(pos[0], pos[1], pos[2]);
  // Move camera with rocket
  updateCameraFollow(rocket.position);
  render();
}

export function addSnapshot(container, ctx, pos, quat, t) {
  const w = 100, h = 80;
  const rt = new THREE.WebGLRenderTarget(w, h);

  const tempCam = ctx.camera.clone();
  tempCam.aspect = w/h;
  tempCam.updateProjectionMatrix();

  // Place a temporary dummy to set pose, then render
  const dummy = new THREE.Object3D();
  dummy.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
  dummy.position.set(pos[0], pos[1], pos[2]);
  dummy.updateMatrixWorld();
  // We will temporarily override rocket transform, render, then restore
  const oldPos = ctx.rocket.position.clone();
  const oldQuat = ctx.rocket.quaternion.clone();

  ctx.rocket.position.copy(dummy.position);
  ctx.rocket.quaternion.copy(dummy.quaternion);
  ctx.renderer.setRenderTarget(rt);
  ctx.renderer.render(ctx.scene, tempCam);
  ctx.renderer.setRenderTarget(null);

  // Restore
  ctx.rocket.position.copy(oldPos);
  ctx.rocket.quaternion.copy(oldQuat);

  const pixels = new Uint8Array(w*h*4);
  ctx.renderer.readRenderTargetPixels(rt, 0, 0, w, h, pixels);
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const c2d = canvas.getContext('2d');
  const imageData = c2d.createImageData(w, h);
  imageData.data.set(pixels);
  // Flip Y
  const flipped = new ImageData(w, h);
  for (let row=0; row<h; row++) {
    const srcStart = (h-1-row)*w*4;
    const dstStart = row*w*4;
    flipped.data.set(imageData.data.slice(srcStart, srcStart+w*4), dstStart);
  }
  c2d.putImageData(flipped, 0, 0);

  const div = document.createElement('div');
  div.className = 'snapshotItem';
  const label = document.createElement('div');
  label.textContent = `t=${t.toFixed(2)}s`;
  label.style.marginTop = '4px';
  div.appendChild(canvas);
  div.appendChild(label);
  container.appendChild(div);
}

// =====================
// Navball implementation
// =====================

function createNavballMarker(THREE_NS, type) {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');

  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 8;
  ctx.shadowColor = 'rgba(0,255,0,0.7)';
  ctx.shadowBlur = 15;

  ctx.beginPath();
  ctx.arc(size/2, size/2, size/2 - ctx.lineWidth, 0, 2 * Math.PI);
  ctx.stroke();

  if (type === 'prograde') {
    ctx.beginPath();
    ctx.moveTo(size/2, ctx.lineWidth); ctx.lineTo(size/2, size - ctx.lineWidth);
    ctx.moveTo(ctx.lineWidth, size/2); ctx.lineTo(size - ctx.lineWidth, size/2);
    ctx.stroke();
  } else {
    const offset = size * 0.25;
    ctx.beginPath();
    ctx.moveTo(offset, offset); ctx.lineTo(size - offset, size - offset);
    ctx.moveTo(size - offset, offset); ctx.lineTo(offset, size - offset);
    ctx.stroke();
  }

  const texture = new THREE_NS.CanvasTexture(canvas);
  const material = new THREE_NS.SpriteMaterial({ map: texture, depthTest: false, transparent: true });
  const sprite = new THREE_NS.Sprite(material);
  sprite.scale.set(1.5, 1.5, 1.5);
  return sprite;
}

export function setupNavball(canvas, speedEl, headingEl) {
  if (!canvas) return null;

  const NAVBALL_RADIUS = 5;

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 1000);
  camera.position.z = 10;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, canvas });
  renderer.setPixelRatio(window.devicePixelRatio);

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const dir = new THREE.DirectionalLight(0xffffff, 1.8);
  dir.position.set(5, 10, 7.5);
  scene.add(dir);

  // Navball mesh
  const navballGroup = new THREE.Group();
  scene.add(navballGroup);

  const textureLoader = new THREE.TextureLoader();
  const navballTexture = textureLoader.load(
    'https://raw.githubusercontent.com/linuxgurugamer/NavBallTextureChanger/refs/heads/master/GameData/NavBallTextureChanger/PluginData/Skins/stock.png',
    () => render()
  );
  navballTexture.colorSpace = THREE.SRGBColorSpace;
  const navballMaterial = new THREE.MeshStandardMaterial({ map: navballTexture, roughness: 0.6, metalness: 0.1 });
  const navballGeometry = new THREE.SphereGeometry(NAVBALL_RADIUS, 64, 32);
  const navballMesh = new THREE.Mesh(navballGeometry, navballMaterial);
  navballGroup.add(navballMesh);

  // Markers
  const progradeMarker = createNavballMarker(THREE, 'prograde');
  const retrogradeMarker = createNavballMarker(THREE, 'retrograde');
  scene.add(progradeMarker, retrogradeMarker);

  function render() { renderer.render(scene, camera); }

  // Resize handling
  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    const aspect = rect.width / Math.max(1, rect.height);
    const frustumSize = 15;
    camera.left = -frustumSize * aspect / 2;
    camera.right = frustumSize * aspect / 2;
    camera.top = frustumSize / 2;
    camera.bottom = -frustumSize / 2;
    camera.updateProjectionMatrix();
    renderer.setSize(rect.width, rect.height, false);
    render();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  window.addEventListener('resize', resize);
  // Initial draw
  resize();

  return {
    scene, camera, renderer,
    navballGroup,
    progradeMarker,
    retrogradeMarker,
    NAVBALL_RADIUS,
    speedEl,
    headingEl,
    render,
  };
}

export function updateNavball(ctx, velVec3, quatScalarFirst) {
  if (!ctx || !velVec3 || !quatScalarFirst) return;

  const { navballGroup, progradeMarker, retrogradeMarker, NAVBALL_RADIUS, speedEl, headingEl, render } = ctx;

  // Convert [w,x,y,z] to THREE Quaternion (x,y,z,w)
  const [w, x, y, z] = quatScalarFirst;
  const vehicleQ = new THREE.Quaternion(x, y, z, w).normalize();
  // Navball rotates opposite of the vehicle orientation
  navballGroup.quaternion.copy(vehicleQ).invert();

  const v = new THREE.Vector3(velVec3[0], velVec3[1], velVec3[2]);
  const speed = v.length();

  if (Number.isFinite(speed) && speed > 1e-6) {
    const progradeWorld = v.clone().divideScalar(speed);
    const progradeDisplay = progradeWorld.clone().applyQuaternion(navballGroup.quaternion);
    progradeMarker.position.copy(progradeDisplay).multiplyScalar(NAVBALL_RADIUS * 1.01);
    progradeMarker.visible = progradeDisplay.z > 0;

    const retrogradeDisplay = progradeWorld.clone().negate().applyQuaternion(navballGroup.quaternion);
    retrogradeMarker.position.copy(retrogradeDisplay).multiplyScalar(NAVBALL_RADIUS * 1.01);
    retrogradeMarker.visible = retrogradeDisplay.z > 0;
  } else {
    progradeMarker.visible = false;
    retrogradeMarker.visible = false;
  }

  if (speedEl) speedEl.textContent = `${speed.toFixed(1)}m/s`;

  // Heading: world-forward of rocket projected onto XZ plane
  const forwardLocal = new THREE.Vector3(0, 0, 1);
  const forwardWorld = forwardLocal.clone().applyQuaternion(vehicleQ);
  const proj = Math.hypot(forwardWorld.x, forwardWorld.z);
  let hdgText = 'HDG ---°';
  if (proj > 1e-2) {
    const hdg = (Math.atan2(forwardWorld.x, forwardWorld.z) * 180 / Math.PI + 360) % 360;
    hdgText = `HDG ${Math.round(hdg).toString().padStart(3, '0')}°`;
  }
  if (headingEl) headingEl.textContent = hdgText;

  render();
}
