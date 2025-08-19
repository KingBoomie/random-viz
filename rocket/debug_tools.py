from __future__ import annotations
import math
from typing import Dict, Any

import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

from .rocket_sim_jax import simulate, default_params, SimState


def to_numpy_tree(sim_out: Dict[str, jnp.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in sim_out.items()}


def find_first_nan(sim_out_np: Dict[str, np.ndarray]) -> int | None:
    # Stack main states to check for any NaN across them
    stacks = [sim_out_np['pos'], sim_out_np['vel'], sim_out_np['quat'], sim_out_np['omega']]
    fuel = sim_out_np['fuel'][:, None]
    data = np.concatenate([*stacks, fuel], axis=1)
    mask = ~np.isfinite(data).all(axis=1)
    idx = int(np.argmax(mask)) if mask.any() else None
    return idx


def summarize(sim_out_np: Dict[str, np.ndarray]) -> Dict[str, Any]:
    q = sim_out_np['quat']
    qnorm = np.linalg.norm(q, axis=1)
    speed = np.linalg.norm(sim_out_np['vel'], axis=1)
    omega_n = np.linalg.norm(sim_out_np['omega'], axis=1)
    return dict(
        t0=float(sim_out_np['time'][0]),
        t1=float(sim_out_np['time'][-1]),
        qnorm_min=float(np.nanmin(qnorm)),
        qnorm_max=float(np.nanmax(qnorm)),
        speed_max=float(np.nanmax(speed)),
        omega_max=float(np.nanmax(omega_n)),
        fuel_min=float(np.nanmin(sim_out_np['fuel'])),
    )


def debug_run(tf: float = 20.0, dt: float = 0.02) -> None:
    params = default_params()
    s0 = SimState(
        pos=jnp.array([0.0, 0.0, 0.0]),
        vel=jnp.array([0.0, 0.0, 0.0]),
        quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        omega=jnp.array([0.0, 0.0, 0.0]),
        fuel=params['fuel_mass0'],
    )
    sim = simulate(s0, params, tf=tf, dt=dt)
    sim_np = to_numpy_tree(sim)
    idx = find_first_nan(sim_np)
    summary = summarize(sim_np)

    print("summary:", summary)
    if idx is None:
        print("no NaNs detected.")
        return
    t = sim_np['time'][idx]
    print(f"first NaN at index {idx}, t={t}")
    # Print the previous step values for context
    m = max(idx - 1, 0)
    print("prev state:")
    print(dict(
        t=float(sim_np['time'][m]),
        pos=sim_np['pos'][m].tolist(),
        vel=sim_np['vel'][m].tolist(),
        quat=sim_np['quat'][m].tolist(),
        qnorm=float(np.linalg.norm(sim_np['quat'][m])),
        omega=sim_np['omega'][m].tolist(),
        omega_norm=float(np.linalg.norm(sim_np['omega'][m])),
        fuel=float(sim_np['fuel'][m]),
    ))


if __name__ == '__main__':
    debug_run()
