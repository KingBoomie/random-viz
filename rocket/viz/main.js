import { drawLines, drawCursor, clearNode } from './plots.js';
import { loadParquet, loadCSV } from './parsers.js';
import { setupThree, updateThreePose, addSnapshot, setupNavball, updateNavball } from './threeview.js';

const fileInput = document.getElementById('fileInput');
const demoBtn = document.getElementById('demoBtn');
const timeSlider = document.getElementById('timeSlider');
const tVal = document.getElementById('tVal');
const playPause = document.getElementById('playPause');
const speedEl = document.getElementById('speed');
const modeSelect = document.getElementById('modeSelect');
const addSnapshotBtn = document.getElementById('addSnapshot');

const posPlot = document.getElementById('posPlot');
const velPlot = document.getElementById('velPlot');
const eulerPlot = document.getElementById('eulerPlot');
const readout = document.getElementById('readout');
// Navball elements
const navballCanvas = document.getElementById('navballCanvas');
const navballSpeedEl = document.getElementById('navballSpeed');
const navballHeadingEl = document.getElementById('navballHeading');

let data = null; // { t, pos: Nx3, vel: Nx3, quat: Nx4, euler: Nx3, omega: Nx3, fuel: N }
let playing = false;
let rafId = null;
let three = null; // { scene, camera, renderer, rocket }
let cursorIndices = []; // for snapshots
let navball = null; // navball context

function quatToEuler(q) {
  // q = [qw, qx, qy, qz] scalar-first
  const [w, x, y, z] = q;
  // aerospace: roll (x), pitch (y), yaw (z), intrinsic
  const sinr_cosp = 2 * (w * x + y * z);
  const cosr_cosp = 1 - 2 * (x * x + y * y);
  const roll = Math.atan2(sinr_cosp, cosr_cosp);

  const sinp = 2 * (w * y - z * x);
  const pitch = Math.abs(sinp) >= 1 ? Math.sign(sinp) * (Math.PI / 2) : Math.asin(sinp);

  const siny_cosp = 2 * (w * z + x * y);
  const cosy_cosp = 1 - 2 * (y * y + z * z);
  const yaw = Math.atan2(siny_cosp, cosy_cosp);

  return [roll, pitch, yaw];
}

function computeEulerAll(quats) {
  return quats.map(quatToEuler);
}

function setData(parsed) {
  data = parsed;
  // derive euler
  data.euler = computeEulerAll(data.quat);

  timeSlider.max = String(data.t.length - 1);
  timeSlider.value = '0';
  tVal.textContent = `t = ${data.t[0].toFixed(2)} s`;

  // draw plots
  drawLines(posPlot, data.t, data.pos, ['#ef4444', '#22c55e', '#3b82f6'], 'm');
  drawLines(velPlot, data.t, data.vel, ['#ef4444', '#22c55e', '#3b82f6'], 'm/s');
  drawLines(eulerPlot, data.t, data.euler.map(rpy => rpy.map(v => v * 180 / Math.PI)), ['#ef4444', '#22c55e', '#3b82f6'], 'deg');

  // setup 3D
  if (!three) {
    three = setupThree(document.getElementById('threeCanvas'));
  }
  updateThreePose(three, data.pos[0], data.quat[0]);

  // setup navball once
  if (!navball) {
    navball = setupNavball(navballCanvas, navballSpeedEl, navballHeadingEl);
  }
  // Initial navball update
  updateNavball(navball, data.vel[0], data.quat[0]);

  // reset snapshots
  cursorIndices = [];
  clearNode(document.getElementById('snapshots'));
}

function tick() {
  if (!playing || !data) return;
  const dt = Number(speedEl.value || 1);
  const next = Math.min(data.t.length - 1, Number(timeSlider.value) + dt);
  timeSlider.value = String(next);
  onTimeChange();
  if (Number(timeSlider.value) >= data.t.length - 1) {
    playing = false;
    playPause.textContent = 'Play';
    return;
  }
  rafId = requestAnimationFrame(tick);
}

function onTimeChange() {
  const i = Number(timeSlider.value);
  const t = data.t[i];
  tVal.textContent = `t = ${t.toFixed(2)} s`;
  drawCursor(posPlot, data.t, i);
  drawCursor(velPlot, data.t, i);
  drawCursor(eulerPlot, data.t, i);
  updateThreePose(three, data.pos[i], data.quat[i]);
  updateNavball(navball, data.vel[i], data.quat[i]);
  updateReadout(i);
}

playPause.addEventListener('click', () => {
  playing = !playing;
  playPause.textContent = playing ? 'Pause' : 'Play';
  if (playing) {
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(tick);
  }
});

timeSlider.addEventListener('input', () => {
  playing = false;
  playPause.textContent = 'Play';
  onTimeChange();
});

posPlot.addEventListener('click', (e) => handlePlotClick(e, posPlot));
velPlot.addEventListener('click', (e) => handlePlotClick(e, velPlot));
eulerPlot.addEventListener('click', (e) => handlePlotClick(e, eulerPlot));

function handlePlotClick(e, container) {
  if (!data) return;
  const rect = container.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const w = rect.width;
  const idx = Math.round((x / w) * (data.t.length - 1));
  timeSlider.value = String(Math.max(0, Math.min(data.t.length - 1, idx)));
  onTimeChange();
}

function handleSnapshots() {
  const i = Number(timeSlider.value);
  if (cursorIndices.includes(i)) return;
  cursorIndices.push(i);
  addSnapshot(document.getElementById('snapshots'), three, data.pos[i], data.quat[i], data.t[i]);
}

// Simple shortcut: double-click 3D to add snapshot
const threeCanvas = document.getElementById('threeCanvas');
threeCanvas.addEventListener('dblclick', handleSnapshots);
addSnapshotBtn.addEventListener('click', handleSnapshots);

// Mode handling (simple mode hides 3D + snapshots + SVG)
modeSelect.addEventListener('change', () => {
  const simple = modeSelect.value === 'simple';
  const threeD = document.querySelector('.threeD');
  if (threeD) threeD.style.display = simple ? 'none' : '';
  const svgPanel = document.querySelector('.svgPanel');
  if (svgPanel) svgPanel.style.display = simple ? 'none' : '';
});


// File handling
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const ext = file.name.toLowerCase().split('.').pop();
  try {
    const parsed = ext === 'csv' ? await loadCSV(file) : await loadParquet(file);
    setData(parsed);
  } catch (err) {
    console.error(err);
    alert('Failed to load file: ' + err.message);
  }
});

// Demo button assumes rocket_traj.parquet next to index.html or at repo root served statically
// Try local path first, then root.
demoBtn.addEventListener('click', async () => {
  for (const path of ['../rocket_traj.parquet', '../../rocket_traj.parquet', '/rocket_traj.parquet']) {
    try {
      const resp = await fetch(path);
      if (!resp.ok) continue;
      const blob = await resp.blob();
      const parsed = await loadParquet(blob);
      setData(parsed);
      return;
    } catch {}
  }
  alert('Could not locate rocket_traj.parquet. Run rocket_sim_jax.py first.');
});

function fmt(n, p=2) { return (Number.isFinite(n) ? n.toFixed(p) : '—'); }
function updateReadout(i) {
  if (!data) { readout.textContent = ''; return; }
  const t = data.t[i];
  const p = data.pos[i];
  const v = data.vel[i];
  const e = data.euler[i];
  const o = data.omega?.[i] || [NaN, NaN, NaN];
  const f = data.fuel?.[i];
  readout.innerHTML = `
    <div class="item"><div class="label">t</div><div class="value">${fmt(t,2)} s</div></div>
    <div class="item"><div class="label">x</div><div class="value">${fmt(p[0])} m</div></div>
    <div class="item"><div class="label">y</div><div class="value">${fmt(p[1])} m</div></div>
    <div class="item"><div class="label">z</div><div class="value">${fmt(p[2])} m</div></div>
    <div class="item"><div class="label">vx</div><div class="value">${fmt(v[0])} m/s</div></div>
    <div class="item"><div class="label">vy</div><div class="value">${fmt(v[1])} m/s</div></div>
    <div class="item"><div class="label">vz</div><div class="value">${fmt(v[2])} m/s</div></div>
    <div class="item"><div class="label">roll</div><div class="value">${fmt(e[0]*180/Math.PI)}°</div></div>
    <div class="item"><div class="label">pitch</div><div class="value">${fmt(e[1]*180/Math.PI)}°</div></div>
    <div class="item"><div class="label">yaw</div><div class="value">${fmt(e[2]*180/Math.PI)}°</div></div>
    <div class="item"><div class="label">ωx</div><div class="value">${fmt(o[0])} rad/s</div></div>
    <div class="item"><div class="label">ωy</div><div class="value">${fmt(o[1])} rad/s</div></div>
    <div class="item"><div class="label">ωz</div><div class="value">${fmt(o[2])} rad/s</div></div>
    <div class="item"><div class="label">fuel</div><div class="value">${fmt(f ?? NaN)} kg</div></div>
  `;
}
