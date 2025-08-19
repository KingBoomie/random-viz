# rocket sim + viz

- goal: fast-ish, jit-friendly 3d amateur rocket sim with simple aero and rotating dynamics. outputs parquet.
- stack: python 3.13, jax + diffrax for ode integration; pyarrow for parquet. small web viz under `viz/`.

## run

- sim (writes `rocket_traj.parquet`):
  - from repo root or `rocket/`: `python rocket/rocket_sim_jax.py`
  - quick sanity with extra logging: `python rocket/debug_tools.py`
- view: open `rocket/viz/index.html` in a browser. it expects `rocket_traj.parquet` one dir up (adjust in `parsers.js` if you move it).

## files

- `rocket_sim_jax.py`: dynamics, integrator, and parquet logger. tweak `params_1kg()` in `params.py` to change vehicle.
- `debug_tools.py`: runs the sim, checks for nans, prints a summary.
- `params.py`: two presets (~1 kg and ~5 kg). swap as you like.
- `viz/`: static html/js plots.

## caveats

- aero is toy-level: linear lift curve, clipped aoa, no compressibility, flat-plate-ish drag. good enough for vibes, not for cfd.
- ground collision is a linear interpolation to z=0 and then stop; no bounce.
