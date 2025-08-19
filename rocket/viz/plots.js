// Lightweight plotting utilities using SVG

export function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function extent(arr) {
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (lo === hi) return [lo - 1, hi + 1];
  return [lo, hi];
}

function pad([lo, hi], p = 0.05) {
  const d = hi - lo;
  return [lo - p * d, hi + p * d];
}

export function drawLines(container, t, series3, colors=['#ef4444','#22c55e','#3b82f6'], units='') {
  clearNode(container);
  const W = container.clientWidth || 600;
  const H = container.clientHeight || 220;
  const svg = el('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` });

  const margin = { l: 36, r: 12, t: 8, b: 20 };
  const iw = W - margin.l - margin.r;
  const ih = H - margin.t - margin.b;

  const x = (i) => margin.l + (i / (t.length - 1)) * iw;

  const flat = [];
  for (let k = 0; k < 3; k++) {
    for (let i = 0; i < t.length; i++) flat.push(series3[i][k]);
  }
  const [ymin,ymax] = pad(extent(flat));
  const y = (v) => margin.t + ih - (v - ymin) / (ymax - ymin) * ih;

  // axes
  svg.appendChild(line(margin.l, margin.t, margin.l, margin.t + ih, '#cbd5e1'));
  svg.appendChild(line(margin.l, margin.t + ih, margin.l + iw, margin.t + ih, '#cbd5e1'));

  const names = ['x','y','z'];
  for (let k = 0; k < 3; k++) {
    const path = [];
    for (let i = 0; i < t.length; i++) {
      path.push(`${i === 0 ? 'M' : 'L'} ${x(i)} ${y(series3[i][k])}`);
    }
    svg.appendChild(el('path', { d: path.join(' '), fill: 'none', stroke: colors[k], 'stroke-width': 2 }));

    const label = el('text', { x: W - margin.r - 6, y: 14 + 16*k, 'text-anchor': 'end', fill: colors[k], 'font-size': 11 });
    label.textContent = `${names[k]} (${units})`;
    svg.appendChild(label);
  }

  container.appendChild(svg);
  container._plot = { svg, margin, iw, ih, x, y, t };
}

export function drawCursor(container, t, i) {
  const plot = container._plot;
  if (!plot) return;
  const { svg, margin, ih, x } = plot;
  let cursor = svg.querySelector('line.cursor');
  if (!cursor) {
    cursor = line(0, margin.t, 0, margin.t + ih, '#64748b');
    cursor.classList.add('cursor');
    cursor.setAttribute('stroke-dasharray', '4 3');
    svg.appendChild(cursor);
  }
  cursor.setAttribute('x1', x(i));
  cursor.setAttribute('x2', x(i));
}

function el(name, attrs={}) { const n = document.createElementNS('http://www.w3.org/2000/svg', name); for (const k in attrs) n.setAttribute(k, attrs[k]); return n; }
function line(x1,y1,x2,y2,stroke) { return el('line', { x1, y1, x2, y2, stroke }); }
