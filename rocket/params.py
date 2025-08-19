import math
from typing import Any, Dict

g0 = 9.81
rho = 1.225

eps = 1e-8

def params_5kg() -> Dict[str, Any]:
    """
    Parameters for a ~5 kg amateur rocket (initial mass ~5.0 kg).
    - Moderate T/W (~8–9) with ~5 s burn.
    - Mid-size airframe and fins; modest Cd.
    """
    # masses (kg)
    payload_mass = 3.0
    engine_dry_mass = 0.8
    fuel_mass0 = 1.2  # total ~5.0 kg

    # geometry
    seg_len = 0.75
    radius = 0.06  # ~120 mm dia

    # engine
    isp = 180.0
    burn_time = 5.0
    mass_flow = fuel_mass0 / burn_time
    thrust = mass_flow * g0 * isp  # ~= 424 N

    # aero
    frontal_area = math.pi * radius**2
    Cd_body = 0.35

    # fins
    n_wings = 4
    wing_area = 0.015
    wing_z = 0.15
    wing_span_offset = radius + 0.015
    Cl_alpha = 4.2
    Cd_wing = 0.025

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

def params_1kg() -> Dict[str, Any]:
    """
    Parameters for a ~1 kg model rocket (initial mass ~1.0 kg).
    - Strong T/W (~12) with ~2 s burn.
    - Smaller airframe and fins; slightly higher Cd.
    """
    # masses (kg)
    payload_mass = 0.50
    engine_dry_mass = 0.35
    fuel_mass0 = 0.15

    # geometry
    seg_len = 0.50
    radius = 0.04  # ~80 mm dia

    # engine
    isp = 160.0
    burn_time = 2.0
    mass_flow = fuel_mass0 / burn_time
    thrust = mass_flow * g0 * isp  # ~= 118 N

    # aero
    frontal_area = math.pi * radius**2
    Cd_body = 0.40

    # fins
    n_wings = 4
    wing_area = 0.008   # m^2 each
    wing_z = 0.10       # near base
    wing_span_offset = radius + 0.01
    Cl_alpha = 4.7
    Cd_wing = 0.03

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