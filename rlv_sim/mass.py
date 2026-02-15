"""
RLV Phase-I Ascent Simulation - Mass and inertia computations.
"""

import numpy as np

from . import constants as C


def compute_mass_flow_rate(thrust_on: bool = True, throttle: float = 1.0, stage: int = 1) -> float:
    """
    Compute mass flow rate (negative while propellant is consumed).
    """
    if not thrust_on or throttle <= 0.0:
        return 0.0

    mdot = C.STAGE2_MASS_FLOW_RATE if stage == 2 else C.MASS_FLOW_RATE
    return -mdot * float(np.clip(throttle, 0.0, 1.0))


def compute_mass_derivative(
    m: float,
    thrust_on: bool = True,
    throttle: float = 1.0,
    dry_mass: float = None,
    stage: int = 1,
) -> float:
    """
    Compute dm/dt with dry-mass floor enforcement.
    """
    if dry_mass is None:
        dry_mass = C.DRY_MASS

    propellant_remaining = m - dry_mass
    if propellant_remaining <= 0.0 or not thrust_on or throttle <= 0.0:
        return 0.0

    mdot = C.STAGE2_MASS_FLOW_RATE if stage == 2 else C.MASS_FLOW_RATE
    return -mdot * float(np.clip(throttle, 0.0, 1.0))


def update_mass(
    m: float,
    dt: float,
    thrust_on: bool = True,
    throttle: float = 1.0,
    dry_mass: float = C.DRY_MASS,
) -> float:
    """
    Euler update for mass.
    """
    dm_dt = compute_mass_derivative(m, thrust_on, throttle, dry_mass=dry_mass)
    return max(m + dm_dt * dt, dry_mass)


def is_propellant_exhausted(m: float, dry_mass: float = C.DRY_MASS) -> bool:
    """True if current mass is at/below dry mass."""
    return m <= dry_mass


def get_propellant_fraction(m: float) -> float:
    """
    Fraction of stacked S1 propellant remaining.
    """
    propellant_remaining = max(0.0, m - C.DRY_MASS)
    return min(1.0, propellant_remaining / C.PROPELLANT_MASS)


def compute_center_of_mass(m: float, vehicle_model: str = "stacked") -> float:
    """
    Compute center-of-mass location (m from vehicle base).

    vehicle_model:
      - "stacked": S1+S2 ascent model
      - "orbiter": post-separation S2 model
      - "booster": post-separation S1 recovery model
    """
    model = (vehicle_model or "stacked").lower()

    if model == "orbiter" or (model == "stacked" and m < C.DRY_MASS):
        # Legacy compatibility: old code inferred S2 only from mass.
        s2_frac = max(0.0, min(1.0, (m - C.STAGE2_DRY_MASS) / C.STAGE2_PROPELLANT_MASS))
        return 45.0 + s2_frac * 5.0

    if model == "booster":
        reserve = max(C.STAGE1_LANDING_FUEL_RESERVE, 1.0)
        s1_frac = max(0.0, min(1.0, (m - C.STAGE1_DRY_MASS) / reserve))
        return (
            C.STAGE1_RECOVERY_CG_EMPTY
            + s1_frac * (C.STAGE1_RECOVERY_CG_FULL - C.STAGE1_RECOVERY_CG_EMPTY)
        )

    frac = get_propellant_fraction(m)
    return C.H_CG_EMPTY + frac * (C.H_CG_FULL - C.H_CG_EMPTY)


def compute_inertia_tensor(m: float, vehicle_model: str = "stacked") -> np.ndarray:
    """
    Compute principal inertia tensor diag([Ixx, Iyy, Izz]) in body frame.

    vehicle_model:
      - "stacked": S1+S2 ascent interpolation
      - "orbiter": post-separation S2 interpolation
      - "booster": post-separation S1 recovery interpolation
    """
    model = (vehicle_model or "stacked").lower()

    if model == "orbiter" or (model == "stacked" and m < C.DRY_MASS):
        # Legacy compatibility: old code inferred S2 only from mass.
        s2_frac = max(0.0, min(1.0, (m - C.STAGE2_DRY_MASS) / C.STAGE2_PROPELLANT_MASS))
        ixx = 1.67e5 + s2_frac * (2.5e6 - 1.67e5)
        izz = 1.0e5 + s2_frac * (4.0e5 - 1.0e5)
        return np.diag([ixx, ixx, izz])

    if model == "booster":
        reserve = max(C.STAGE1_LANDING_FUEL_RESERVE, 1.0)
        s1_frac = max(0.0, min(1.0, (m - C.STAGE1_DRY_MASS) / reserve))
        ixx = (
            C.STAGE1_RECOVERY_IXX_EMPTY
            + s1_frac * (C.STAGE1_RECOVERY_IXX_FULL - C.STAGE1_RECOVERY_IXX_EMPTY)
        )
        izz = (
            C.STAGE1_RECOVERY_IZZ_EMPTY
            + s1_frac * (C.STAGE1_RECOVERY_IZZ_FULL - C.STAGE1_RECOVERY_IZZ_EMPTY)
        )
        return np.diag([ixx, ixx, izz])

    frac = get_propellant_fraction(m)
    return C.INERTIA_TENSOR_EMPTY + frac * (C.INERTIA_TENSOR_FULL - C.INERTIA_TENSOR_EMPTY)

