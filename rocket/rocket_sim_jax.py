"""
rocket_sim_jax.py

jax-based 3d amateur rocket sim.
- two 1m cylinders (bottom: engine + fuel, top: payload/structure)
- quaternion orientation, body-frame angular dynamics
- vectorized wing model (4 fins)
- mass/fuel consumption updates COM and MOI each step
- uses diffrax if available; falls back to simple rk4 integrator
- includes a composable sensor noise factory and parquet logging fallback

usage:
    python rocket_sim_jax.py  # runs example sim and writes trajectory to rocket_traj.parquet or .csv

notes:
    - designed to be jit- and vmap-friendly (pure functions, no python loops for dynamics)
    - tune params in `default_params()`

"""

from __future__ import annotations
import math
from typing import NamedTuple, Callable, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

# assume diffrax is available in the environment
import diffrax

from params import params_1kg

# physical constants
g0 = 9.81
rho = 1.225

eps = 1e-8

# ------------------------------ quaternion utilities ------------------------------

def quat_normalize(q: jnp.ndarray) -> jnp.ndarray:
    return q / jnp.linalg.norm(q)


def quat_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    # scalar-first representation: q = [w, x, y, z]
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotmat(q: jnp.ndarray) -> jnp.ndarray:
    qn = quat_normalize(q)
    w, x, y, z = qn
    return jnp.stack([
        jnp.stack([1-2*(y**2+z**2), 2*(x*y - z*w), 2*(x*z + y*w)]),
        jnp.stack([2*(x*y + z*w), 1-2*(x**2+z**2), 2*(y*z - x*w)]),
        jnp.stack([2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x**2+y**2)])
    ], axis=0)

# ------------------------------ geometry / inertia ------------------------------

def cylinder_inertia(m: float, r: float, h: float) -> jnp.ndarray:
    # cylinder aligned along local z-axis
    i_z = 0.5 * m * (r**2)
    i_x = (1.0/12.0) * m * (3*(r**2) + (h**2))
    return jnp.diag(jnp.array([i_x, i_x, i_z]))


def compute_com_and_inertia(segment_masses: jnp.ndarray,
                             segment_zpos: jnp.ndarray,
                             radii: jnp.ndarray,
                             heights: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    # segment_zpos: centers along body z measured from base (m)
    total_m = jnp.sum(segment_masses)
    com_z = jnp.sum(segment_masses * segment_zpos) / (total_m + eps)

    # build inertia in body frame using parallel axis theorem
    def per_seg(i, carry):
        I_acc = carry
        m = segment_masses[i]
        r = radii[i]
        h = heights[i]
        z = segment_zpos[i]
        Ic = cylinder_inertia(m, r, h)
        d = jnp.array([0.0, 0.0, z - com_z])
        d2 = jnp.dot(d, d)
        parallel = m * (d2 * jnp.eye(3) - jnp.outer(d, d))
        return I_acc + Ic + parallel

    I0 = jnp.zeros((3, 3))
    I_total = jax.lax.fori_loop(0, segment_masses.shape[0], per_seg, I0)
    return com_z, I_total, total_m

# ------------------------------ default vehicle / wings params ------------------------------

def default_params() -> Dict[str, Any]:
    # masses (kg)
    payload_mass = 20.0
    engine_dry_mass = 10.0
    fuel_mass0 = 30.0

    # geometry
    seg_len = 1.0
    radius = 0.15

    # engine
    isp = 200.0
    burn_time = 10.0
    mass_flow = fuel_mass0 / burn_time
    thrust = mass_flow * g0 * isp

    # aero
    frontal_area = math.pi * radius**2
    Cd_body = 0.3

    # wings (4 fins)
    n_wings = 4
    wing_area = 0.04  # m^2 each
    wing_z = 0.2  # attach near base
    wing_span_offset = radius + 0.02
    Cl_alpha = 4.0  # per rad
    Cd_wing = 0.02

    return dict(
        payload_mass=payload_mass,
        engine_dry_mass=engine_dry_mass,
        fuel_mass0=fuel_mass0,
        seg_len=seg_len,
        radius=radius,
        isp=isp,
        burn_time=burn_time,
        mass_flow=mass_flow,
        thrust=thrust,
        frontal_area=frontal_area,
        Cd_body=Cd_body,
        n_wings=n_wings,
        wing_area=wing_area,
        wing_z=wing_z,
        wing_span_offset=wing_span_offset,
        Cl_alpha=Cl_alpha,
        Cd_wing=Cd_wing,
    )


# ------------------------------ aerodynamic helpers ------------------------------

def wing_attachment_vectors(n_wings: int, span_offset: float, z_attach: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # returns (span_dirs (n,3), chord_dirs (n,3), positions (n,3)) expressed in body frame
    angles = jnp.linspace(0.0, 2*jnp.pi, n_wings, endpoint=False)
    span_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)], axis=1)
    chord_dirs = jnp.stack([-jnp.sin(angles), jnp.cos(angles), jnp.zeros_like(angles)], axis=1)
    radial = span_dirs * (span_offset[:, None] if isinstance(span_offset, jnp.ndarray) else span_offset)
    positions = jnp.stack([radial[:, 0], radial[:, 1], jnp.full((n_wings,), z_attach)], axis=1)
    return span_dirs, chord_dirs, positions

# ------------------------------ dynamics function ------------------------------

class SimState(NamedTuple):
    pos: jnp.ndarray  # world (3,)
    vel: jnp.ndarray  # world (3,)
    quat: jnp.ndarray  # body->world quaternion (4,), scalar-first
    omega: jnp.ndarray  # body angular velocity (3,)
    fuel: float


def pack_state(s: SimState) -> jnp.ndarray:
    return jnp.concatenate([s.pos, s.vel, s.quat, s.omega, jnp.array([s.fuel])])


def unpack_state(y: jnp.ndarray) -> SimState:
    pos = y[0:3]
    vel = y[3:6]
    quat = y[6:10]
    omega = y[10:13]
    fuel = y[13]
    return SimState(pos=pos, vel=vel, quat=quat, omega=omega, fuel=fuel)


def dynamics(t: float, y: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
    s = unpack_state(y)

    # segments centers (body-frame z positions)
    seg_len = params['seg_len']
    z_engine = 0.5 * seg_len
    z_payload = 1.5 * seg_len
    segment_z = jnp.array([z_payload, z_engine])

    # segment masses: top(payload), bottom(engine: dry + fuel)
    m_payload = params['payload_mass']
    m_engine = params['engine_dry_mass'] + jnp.maximum(s.fuel, 0.0)
    segment_masses = jnp.array([m_payload, m_engine])
    radii = jnp.array([params['radius'], params['radius']])
    heights = jnp.array([seg_len, seg_len])

    com_z, I_body, total_m = compute_com_and_inertia(segment_masses, segment_z, radii, heights)

    # orientation (normalize to avoid drift)
    qn = quat_normalize(s.quat)
    R = quat_to_rotmat(qn)  # body -> world
    v_body = jnp.dot(R.T, s.vel)

    # thrust & fuel flow
    fuel_remaining = jnp.maximum(s.fuel, 0.0)
    # simple burn model: constant mass flow until fuel exhausted
    mass_flow = params['mass_flow']
    burning = jnp.where(fuel_remaining > 0.0, 1.0, 0.0)
    m_dot = - mass_flow * burning
    thrust_mag = jnp.where(burning > 0.0, params['thrust'], 0.0)

    thrust_body = jnp.array([0.0, 0.0, thrust_mag])

    # body drag (apply at COM)
    v_speed = jnp.linalg.norm(v_body)
    v_hat_body = v_body / (v_speed + eps)
    F_body_drag_body = -0.5 * rho * (v_speed**2) * params['Cd_body'] * params['frontal_area'] * v_hat_body

    # wings: vectorized
    n_w = params['n_wings']
    angles = jnp.linspace(0.0, 2*jnp.pi, n_w, endpoint=False)
    span_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)], axis=1)
    chord_dirs = jnp.stack([-jnp.sin(angles), jnp.cos(angles), jnp.zeros_like(angles)], axis=1)
    radial = span_dirs * params['wing_span_offset']  # (n,3) with z=0
    # Positions of wing attachment points in body frame (x, y at span; fixed z)
    # positions of wings relative to COM in body frame
    r_wings = jnp.stack([
        radial[:, 0],
        radial[:, 1],
        jnp.full((n_w,), params['wing_z'] - com_z)
    ], axis=1)  # (n,3)

    # velocity at each wing point (body frame)
    omega = s.omega
    v_point = v_body[None, :] + jnp.cross(omega, r_wings)

    speed_point = jnp.linalg.norm(v_point, axis=1)
    v_hat_point = v_point / (speed_point[:,None] + eps)

    # robust AoA using atan2; negative sign for aerodynamic convention (lift opposes flow)
    aoa = jnp.arctan2(-jnp.sum(v_point * chord_dirs, axis=1), v_point[:, 2] + eps)
    # limit AoA to avoid unrealistically large lift (stall model placeholder)
    aoa = jnp.clip(aoa, -0.5, 0.5)  # +/- ~28.6 degrees

    Cl = params['Cl_alpha'] * aoa
    Cl = jnp.clip(Cl, -1.2, 1.2)
    lift_mag = 0.5 * rho * (speed_point**2) * Cl * params['wing_area']
    # lift direction: perpendicular to flow and span (span x v_hat gives axis, cross again to align)
    lift_dir_raw = jnp.cross(v_hat_point, span_dirs)
    lift_dir_norm = jnp.linalg.norm(lift_dir_raw, axis=1) + eps
    lift_dir = (lift_dir_raw.T / lift_dir_norm).T

    lift = (lift_mag[:,None]) * lift_dir
    drag_mag = 0.5 * rho * (speed_point**2) * params['Cd_wing'] * params['wing_area']
    drag = - (drag_mag[:,None]) * v_hat_point

    F_wings_body = lift + drag  # (n,3)
    F_wings_total_body = jnp.sum(F_wings_body, axis=0)
    torque_wings_body = jnp.sum(jnp.cross(r_wings, F_wings_body), axis=0)

    # thrust application point (engine center)
    r_engine_body = jnp.array([0.0, 0.0, z_engine - com_z])
    torque_thrust_body = jnp.cross(r_engine_body, thrust_body)

    # total body forces (expressed in body frame): wings + body drag + thrust
    F_total_body = F_wings_total_body + F_body_drag_body + thrust_body

    # convert to world for linear acceleration
    F_total_world = jnp.dot(R, F_total_body)
    # gravity
    F_gravity = jnp.array([0.0, 0.0, - total_m * g0])
    accel_world = (F_total_world + F_gravity) / (total_m + eps)

    # rotational dynamics (body frame)
    torque_total_body = torque_wings_body + torque_thrust_body
    # add any gyroscopic term
    Iw = jnp.dot(I_body, omega)
    omega_dot = jnp.linalg.solve(I_body + eps*jnp.eye(3), (torque_total_body - jnp.cross(omega, Iw)))

    # quaternion kinematics
    omega_quat = jnp.concatenate([jnp.array([0.0]), omega])
    quat_dot = 0.5 * quat_mul(qn, omega_quat)

    pos_dot = s.vel
    vel_dot = accel_world
    fuel_dot = m_dot

    dydt = jnp.concatenate([pos_dot, vel_dot, quat_dot, omega_dot, jnp.array([fuel_dot])])
    return dydt

# ------------------------------ integrators / sim loop ------------------------------
def simulate(initial_state: SimState,
             params: Dict[str, Any],
             t0: float = 0.0,
             tf: float = 20.0,
             dt: float = 0.01) -> Dict[str, jnp.ndarray]:
    y0 = pack_state(initial_state)
    times = jnp.arange(t0, tf + 1e-12, dt)

    # Always use diffrax solver
    term = diffrax.ODETerm(lambda t, y, args: dynamics(t, y, params))
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=times)
    sol = diffrax.diffeqsolve(term, solver, t0=t0, y0=y0, t1=tf, dt0=dt, saveat=saveat)
    ys = sol.ys

    # unpack arrays
    pos = ys[:, 0:3]
    vel = ys[:, 3:6]
    quat = ys[:, 6:10]
    omega = ys[:, 10:13]
    fuel = ys[:, 13]

    # --- ground collision handling: stop when crossing back to z <= 0 ---
    z = pos[:, 2]
    z_np = np.asarray(z)
    hit_idx = None
    for k in range(1, len(z_np)):
        if (z_np[k - 1] > 0.0) and (z_np[k] <= 0.0):
            hit_idx = k
            break

    if hit_idx is not None:
        t1, t2 = times[hit_idx - 1], times[hit_idx]
        z1, z2 = pos[hit_idx - 1, 2], pos[hit_idx, 2]
        alpha = jnp.clip((0.0 - z1) / (z2 - z1 + eps), 0.0, 1.0)
        t_hit = t1 + alpha * (t2 - t1)

        def lerp(a, b):
            return a + alpha * (b - a)

        pos_hit = lerp(pos[hit_idx - 1], pos[hit_idx]).at[2].set(0.0)
        vel_hit = lerp(vel[hit_idx - 1], vel[hit_idx])
        quat_hit = quat_normalize(lerp(quat[hit_idx - 1], quat[hit_idx]))
        omega_hit = lerp(omega[hit_idx - 1], omega[hit_idx])
        fuel_hit = lerp(fuel[hit_idx - 1], fuel[hit_idx])

        times = jnp.concatenate([times[:hit_idx], jnp.array([t_hit])])
        pos = jnp.concatenate([pos[:hit_idx], pos_hit[None, :]], axis=0)
        vel = jnp.concatenate([vel[:hit_idx], vel_hit[None, :]], axis=0)
        quat = jnp.concatenate([quat[:hit_idx], quat_hit[None, :]], axis=0)
        omega = jnp.concatenate([omega[:hit_idx], omega_hit[None, :]], axis=0)
        fuel = jnp.concatenate([fuel[:hit_idx], jnp.array([fuel_hit])], axis=0)

    return dict(time=times, pos=pos, vel=vel, quat=quat, omega=omega, fuel=fuel)

# ------------------------------ sensor factory ------------------------------

class SensorTransform:
    def __call__(self, x, key):
        raise NotImplementedError


class GaussianNoise(SensorTransform):
    def __init__(self, std: float):
        self.std = std

    def __call__(self, x, key):
        return x + self.std * jax.random.normal(key, shape=jnp.shape(x))


class Drift(SensorTransform):
    def __init__(self, rate: float):
        self.rate = rate
        self.state = None

    def __call__(self, x, key):
        # maintain a small bias integrated over calls
        if self.state is None:
            self.state = jnp.zeros_like(x)
        self.state = self.state + self.rate
        return x + self.state


class Quantize(SensorTransform):
    def __init__(self, step: float):
        self.step = step

    def __call__(self, x, key):
        return jnp.round(x / self.step) * self.step


class Dropout(SensorTransform):
    def __init__(self, p: float, last_value=None):
        self.p = p
        self.last_value = last_value

    def __call__(self, x, key):
        keep = jax.random.uniform(key, shape=()) > self.p
        out = jnp.where(keep, x, jnp.where(self.last_value is None, jnp.nan, self.last_value))
        self.last_value = out
        return out


class SensorFactory:
    def __init__(self, transforms: list[SensorTransform]):
        self.transforms = transforms

    def sample(self, x, seed=0):
        key = jax.random.PRNGKey(seed)
        out = x
        for i, t in enumerate(self.transforms):
            key, sub = jax.random.split(key)
            out = t(out, sub)
        return out

# ------------------------------ example / main ------------------------------------

if __name__ == '__main__':
    params = params_1kg()

    # initial state: sitting on pad, small perturbation in angle
    pos0 = jnp.array([0.0, 0.0, 0.0])
    vel0 = jnp.array([0.0, 0.0, 0.0])
    quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
    omega0 = jnp.array([0.0, 0.0, 0.0])
    fuel0 = params['fuel_mass0'] if 'fuel_mass0' in params else params['fuel_mass0']

    s0 = SimState(pos0, vel0, quat0, omega0, fuel0)

    # run sim
    sim_out = simulate(s0, params, tf=40.0, dt=0.02)

    # simple logging: write parquet with pyarrow
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        't': np.asarray(sim_out['time']),
        'x': np.asarray(sim_out['pos'][:, 0]),
        'y': np.asarray(sim_out['pos'][:, 1]),
        'z': np.asarray(sim_out['pos'][:, 2]),
        'vx': np.asarray(sim_out['vel'][:, 0]),
        'vy': np.asarray(sim_out['vel'][:, 1]),
        'vz': np.asarray(sim_out['vel'][:, 2]),
        'qx': np.asarray(sim_out['quat'][:, 1]),
        'qy': np.asarray(sim_out['quat'][:, 2]),
        'qz': np.asarray(sim_out['quat'][:, 3]),
        'qw': np.asarray(sim_out['quat'][:, 0]),
        'omx': np.asarray(sim_out['omega'][:, 0]),
        'omy': np.asarray(sim_out['omega'][:, 1]),
        'omz': np.asarray(sim_out['omega'][:, 2]),
        'fuel': np.asarray(sim_out['fuel']),
    })
    pq.write_table(table, 'rocket_traj.parquet')
    print('wrote rocket_traj.parquet')

    # example sensors
    gps_sensor = SensorFactory([GaussianNoise(3.0), Drift(0.01), Quantize(0.1)])
    sample0 = gps_sensor.sample(sim_out['pos'][0], seed=42)
    print('sample gps at t0:', sample0)
