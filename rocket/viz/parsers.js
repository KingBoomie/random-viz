// Parquet + CSV loaders for browser
// Uses parquet-wasm + Apache Arrow for parquet decoding

// Load Arrow JS lazily
let arrowModulePromise = null;
async function loadArrow() {
  if (!arrowModulePromise) {
    arrowModulePromise = import('https://cdn.jsdelivr.net/npm/apache-arrow@14.0.2/+esm');
  }
  return arrowModulePromise;
}

export async function loadParquet(blob) {
  const arrow = await loadArrow();
  // parquet-wasm ESM requires explicit WASM initialization
  const parquet = await import('https://cdn.jsdelivr.net/npm/parquet-wasm@0.6.1/esm/+esm');
  // Initialize WASM from CDN (version must match the import)
  await parquet.default('https://cdn.jsdelivr.net/npm/parquet-wasm@0.6.1/esm/parquet_wasm_bg.wasm');

  const buf = new Uint8Array(await blob.arrayBuffer());
  const wasmTable = parquet.readParquet(buf);
  const table = arrow.tableFromIPC(wasmTable.intoIPCStream());

  // Extract columns by name (Arrow JS compatibility across versions)
  const getCol = (name) => {
    try {
      const fields = table.schema?.fields ?? [];
      let idx = -1;
      for (let i = 0; i < fields.length; i++) {
        if (fields[i]?.name === name) { idx = i; break; }
      }
      let col = null;
      if (idx >= 0 && typeof table.getColumnAt === 'function') col = table.getColumnAt(idx);
      else if (typeof table.getChild === 'function') col = table.getChild(name);
      if (!col) return [];
      const arr = col.toArray ? col.toArray() : (col.data?.[0]?.values ?? []);
      return Array.from(arr);
    } catch (_) {
      return [];
    }
  };

  const t = getCol('t');
  const x = getCol('x');
  const y = getCol('y');
  const z = getCol('z');
  const vx = getCol('vx');
  const vy = getCol('vy');
  const vz = getCol('vz');
  const qx = getCol('qx');
  const qy = getCol('qy');
  const qz = getCol('qz');
  const qw = getCol('qw');
  const omx = getCol('omx');
  const omy = getCol('omy');
  const omz = getCol('omz');
  const fuelCol = getCol('fuel');

  const N = t.length;
  const pos = new Array(N);
  const vel = new Array(N);
  const quat = new Array(N);
  const omega = new Array(N);
  const fuel = new Array(N);
  for (let i = 0; i < N; i++) {
    pos[i] = [x[i], y[i], z[i]];
    vel[i] = [vx[i], vy[i], vz[i]];
    quat[i] = [qw[i], qx[i], qy[i], qz[i]]; // scalar-first
    omega[i] = [omx[i], omy[i], omz[i]];
    fuel[i] = fuelCol[i] ?? NaN;
  }

  return { t, pos, vel, quat, omega, fuel };
}

export async function loadCSV(blob) {
  const text = await (blob.text ? blob.text() : new Response(blob).text());
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(',').map(s => s.trim());
  const idx = Object.fromEntries(headers.map((h,i)=>[h,i]));
  const rows = lines.slice(1).map(line => line.split(',').map(Number));
  const take = (name) => rows.map(r => r[idx[name]]);
  const t = take('t');
  const pos = t.map((_,i) => [rows[i][idx['x']], rows[i][idx['y']], rows[i][idx['z']]]);
  const vel = t.map((_,i) => [rows[i][idx['vx']], rows[i][idx['vy']], rows[i][idx['vz']]]);
  const quat = t.map((_,i) => [rows[i][idx['qw']], rows[i][idx['qx']], rows[i][idx['qy']], rows[i][idx['qz']]]);
  const omega = t.map((_,i) => [rows[i][idx['omx']], rows[i][idx['omy']], rows[i][idx['omz']]]);
  const fuel = idx['fuel'] !== undefined ? take('fuel') : new Array(t.length).fill(NaN);
  return { t, pos, vel, quat, omega, fuel };
}
