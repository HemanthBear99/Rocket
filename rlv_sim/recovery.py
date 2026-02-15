"""
Shared recovery-physics helpers for booster guidance and mission transitions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import constants as C
from .config import SimulationConfig
from .utils import compute_relative_velocity


def rotating_launch_site_eci(t: float) -> np.ndarray:
    """Launch-site position in ECI at time t (Earth rotation applied)."""
    theta = C.EARTH_ROTATION_RATE * float(t)
    c_t = float(np.cos(theta))
    s_t = float(np.sin(theta))
    x0, y0, z0 = C.INITIAL_POSITION
    return np.array([
        c_t * x0 - s_t * y0,
        s_t * x0 + c_t * y0,
        z0,
    ], dtype=float)


def target_landing_site_eci(t: float, downrange_km: float) -> np.ndarray:
    """
    RTLS landing-site position in ECI.

    `downrange_km` is a surface-arc offset from the launch site along local east.
    """
    launch = rotating_launch_site_eci(t)
    launch_hat = launch / max(np.linalg.norm(launch), 1e-9)
    k_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    east = np.cross(k_axis, launch_hat)
    east_norm = float(np.linalg.norm(east))
    if east_norm < 1e-9:
        east = np.array([0.0, 1.0, 0.0], dtype=float)
        east_norm = 1.0
    east = east / east_norm

    theta = float(downrange_km) * 1000.0 / C.R_EARTH
    target_hat = np.cos(theta) * launch_hat + np.sin(theta) * east
    target_hat = target_hat / max(np.linalg.norm(target_hat), 1e-9)
    return C.R_EARTH * target_hat


def great_circle_distance_m(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Great-circle surface distance between two geocentric position vectors."""
    a_hat = np.asarray(r_a, dtype=float)
    b_hat = np.asarray(r_b, dtype=float)
    a_hat = a_hat / max(np.linalg.norm(a_hat), 1e-9)
    b_hat = b_hat / max(np.linalg.norm(b_hat), 1e-9)
    angle = float(np.arccos(np.clip(np.dot(a_hat, b_hat), -1.0, 1.0)))
    return C.R_EARTH * angle


def booster_propellant_remaining(mass_kg: float) -> float:
    """Return booster propellant remaining above Stage-1 dry mass."""
    return max(0.0, float(mass_kg - C.STAGE1_DRY_MASS))


def booster_min_propellant_after_boostback(cfg: SimulationConfig) -> float:
    """Absolute reserve required after boostback to protect entry + landing."""
    return float(cfg.booster_entry_budget_kg + cfg.booster_landing_reserve_kg)


def estimate_ballistic_apogee(altitude_m: float, radial_velocity_mps: float, g_local: float) -> float:
    """Estimate no-thrust ballistic apogee from current altitude and radial speed."""
    if radial_velocity_mps <= 0.0:
        return float(altitude_m)
    return float(altitude_m + (radial_velocity_mps ** 2) / (2.0 * max(g_local, 1e-6)))


def estimate_suicide_burn(
    r: np.ndarray,
    v: np.ndarray,
    mass_kg: float,
    thrust_newton: float,
    safety_factor: float = 1.5,
    min_throttle: float = 0.3,
    max_throttle: float = 1.0,
    horizontal_weight: float = 1.0,
    isp: float = None,
) -> Dict[str, float]:
    """
    Energy-based landing ignition estimator.

    Computes the energy required to bring the vehicle to rest and compares
    it against available thrust energy accounting for variable mass (fuel
    burn changes T/m during the deceleration).  This replaces the earlier
    constant-acceleration heuristic (v^2 / 2a * safety_factor).

    The energy integral for a variable-mass rocket under constant thrust F
    and local gravity g over a burn that reduces mass from m0 to m_dry is:

        E_thrust = F * Isp * g0 * ln(m0 / m_dry)   (Tsiolkovsky work)

    The vehicle's current mechanical energy relative to the landing site is:

        E_mech = 0.5 * m * v_eff^2  +  m * g * h

    Ignition triggers when  E_mech >= E_thrust * margin_factor,  i.e. the
    remaining thrust work (with mass depletion) just covers the kinetic +
    potential energy that must be removed.

    Returns dict with keys: ignite, throttle, burn_altitude, v_descent,
    v_horizontal, v_effective, a_brake.
    """
    r_norm = float(np.linalg.norm(r))
    if r_norm < 1.0:
        return {
            'ignite': False,
            'throttle': 0.0,
            'burn_altitude': 0.0,
            'v_descent': 0.0,
            'a_brake': 0.0,
        }

    if isp is None:
        isp = C.ISP

    vertical = r / r_norm
    altitude = r_norm - C.R_EARTH
    v_rel = compute_relative_velocity(r, v)
    v_descent = -float(np.dot(v_rel, vertical))
    v_horiz_vec = v_rel - np.dot(v_rel, vertical) * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))
    v_effective = float(np.sqrt(
        max(v_descent, 0.0) ** 2 + horizontal_weight * (v_horiz ** 2)
    ))

    if v_descent <= 0.0 and v_effective <= 0.0:
        return {
            'ignite': False,
            'throttle': 0.0,
            'burn_altitude': 0.0,
            'v_descent': v_descent,
            'v_horizontal': v_horiz,
            'v_effective': v_effective,
            'a_brake': 0.0,
        }

    g_local = C.MU_EARTH / (r_norm ** 2)
    thrust_accel = float(thrust_newton / max(mass_kg, 1.0))
    a_brake = thrust_accel - g_local

    if a_brake <= 0.0:
        return {
            'ignite': True,
            'throttle': max_throttle,
            'burn_altitude': max(0.0, altitude),
            'v_descent': v_descent,
            'v_horizontal': v_horiz,
            'v_effective': v_effective,
            'a_brake': a_brake,
        }

    # --- Energy-based ignition decision ---
    # Integrate the 1D variable-mass braking equation to find the altitude
    # required to stop.  With constant thrust F, exhaust velocity ve, and
    # local gravity g, the deceleration over the burn is:
    #
    #     a(m) = F/m - g
    #
    # The velocity change (Tsiolkovsky) from mass m0 to m_f is:
    #     dv = ve * ln(m0/m_f) - g * t_burn
    #
    # We solve for m_f such that dv = v_effective (stop from current speed),
    # then compute the altitude consumed during that burn using the
    # work-energy theorem:
    #     h_burn = (v_eff^2) / (2 * a_mean)
    # where a_mean accounts for the mass change via the log-mean:
    #     a_mean = F * ln(m0/m_f) / (m0 - m_f) - g
    #
    # This is more accurate than v^2/(2*a_brake) because it accounts for
    # the increasing T/m ratio as fuel is consumed during the burn.

    ve = isp * C.G0
    mdot = thrust_newton / ve  # kg/s mass flow at full throttle
    propellant = max(mass_kg - C.STAGE1_DRY_MASS, 0.0)

    # Fuel needed to produce dv = v_effective (from Tsiolkovsky, gravity-free)
    # m_f = m0 * exp(-v_eff / ve)
    mass_ratio_needed = np.exp(v_effective / ve)
    m_final_ideal = mass_kg / mass_ratio_needed
    fuel_needed = mass_kg - m_final_ideal

    if fuel_needed > propellant:
        # Not enough fuel to stop — must ignite immediately
        m_final_actual = C.STAGE1_DRY_MASS
    else:
        m_final_actual = m_final_ideal

    # Log-mean acceleration during the burn (accounts for variable mass)
    dm = mass_kg - m_final_actual
    if dm > 1.0 and m_final_actual > 0.0:
        # F * ln(m0/mf) / (m0 - mf) is the log-mean of F/m over the burn
        a_mean = thrust_newton * np.log(mass_kg / m_final_actual) / dm - g_local
    else:
        a_mean = a_brake

    a_mean = max(a_mean, 0.1)  # Prevent division by zero

    # Altitude consumed during the braking burn
    h_ignite = (v_effective ** 2) / (2.0 * a_mean)

    # Ignite when altitude <= h_ignite * safety_factor
    ignite = altitude <= h_ignite * safety_factor

    if ignite:
        # Throttle command: required deceleration to stop in remaining altitude
        if altitude > 10.0:
            a_required = (v_effective ** 2) / (2.0 * altitude) + g_local
        else:
            a_required = thrust_accel
        throttle = float(np.clip(
            a_required / max(thrust_accel, 1e-6), min_throttle, max_throttle
        ))
    else:
        throttle = 0.0

    return {
        'ignite': bool(ignite),
        'throttle': float(throttle),
        'burn_altitude': float(h_ignite),
        'v_descent': float(v_descent),
        'v_horizontal': float(v_horiz),
        'v_effective': float(v_effective),
        'a_brake': float(a_brake),
    }
