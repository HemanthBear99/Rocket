"""
Guidance module implementing altitude-based gamma profile with PID tracking
and thrust-direction generation for gravity turn.

All mutable guidance state is encapsulated in GuidanceState to support
concurrent multi-vehicle simulations.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from . import constants as C
from .config import SimulationConfig
from .forces import compute_atmosphere_properties
from .recovery import (
    booster_propellant_remaining,
    estimate_ballistic_apogee,
    estimate_suicide_burn,
    target_landing_site_eci,
)
from .types import GuidanceOutput
from .utils import compute_relative_velocity


# =============================================================================
# GUIDANCE STATE (per-vehicle, replaces module-level _pid_state / _oi_state)
# =============================================================================

@dataclass
class GuidanceState:
    """Per-vehicle guidance state for PID tracking and orbit insertion.

    Each vehicle (stacked, orbiter, booster) gets its own instance so that
    guidance can run independently in parallel simulations.
    """
    # PID state for gamma tracking
    prev_gamma_meas: float = 90.0
    gamma_int: float = 0.0

    # Orbit insertion ramp-in state
    oi_start_time: Optional[float] = None
    oi_start_direction: Optional[np.ndarray] = None
    last_ascent_direction: Optional[np.ndarray] = None
    prev_cmd_direction: Optional[np.ndarray] = None
    coast_start_time: Optional[float] = None
    booster_landing_burn_started: bool = False


def create_guidance_state() -> GuidanceState:
    """Create a fresh GuidanceState. Replaces the old reset_guidance()."""
    return GuidanceState()


# Legacy compatibility: module-level default state + reset function
_default_gs = GuidanceState()


def reset_guidance():
    """Reset the module-level default guidance state.

    For backward compatibility with callers that don't pass GuidanceState.
    New code should use create_guidance_state() and pass gs explicitly.
    """
    global _default_gs
    _default_gs = GuidanceState()


# =============================================================================
# HELPER FUNCTIONS (stateless)
# =============================================================================

def _limit_aoa(thrust_dir: np.ndarray, velocity: np.ndarray,
               max_aoa_rad: float) -> np.ndarray:
    """
    Limit angle-of-attack: cap the angle between thrust direction and velocity.

    If the angle between thrust_dir and velocity exceeds max_aoa_rad,
    rotate thrust_dir toward velocity until the angle equals max_aoa_rad.

    This prevents structural loads (high Q·alpha) during atmospheric flight
    and keeps attitude error bounded during phase transitions.

    Args:
        thrust_dir: Desired thrust direction (unit vector)
        velocity: Velocity vector (does not need to be unit)
        max_aoa_rad: Maximum allowed angle-of-attack (radians)

    Returns:
        AoA-limited thrust direction (unit vector)
    """
    v_norm = np.linalg.norm(velocity)
    if v_norm < 10.0:
        return thrust_dir  # AoA undefined at low speed

    v_hat = velocity / v_norm
    cos_aoa = np.clip(np.dot(thrust_dir, v_hat), -1.0, 1.0)
    aoa = np.arccos(cos_aoa)

    if aoa <= max_aoa_rad:
        return thrust_dir  # Already within limit

    # Rotate thrust_dir toward v_hat to reduce AoA to max_aoa_rad
    # Use Rodrigues rotation about the axis perpendicular to both
    axis = np.cross(v_hat, thrust_dir)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return thrust_dir  # Parallel or anti-parallel

    axis = axis / axis_norm
    # New direction: rotate v_hat away by max_aoa_rad in the thrust_dir direction
    limited = (v_hat * np.cos(max_aoa_rad)
               + np.cross(axis, v_hat) * np.sin(max_aoa_rad)
               + axis * np.dot(axis, v_hat) * (1.0 - np.cos(max_aoa_rad)))
    limited_norm = np.linalg.norm(limited)
    if limited_norm > 1e-9:
        limited /= limited_norm
    return limited


def compute_local_vertical(r: np.ndarray) -> np.ndarray:
    r_norm = np.linalg.norm(r)
    if r_norm < C.ZERO_TOLERANCE:
        return np.array([1.0, 0.0, 0.0])
    return r / r_norm


def compute_local_frame(r: np.ndarray):
    vertical = compute_local_vertical(r)
    k_axis = np.array([0.0, 0.0, 1.0])
    east = np.cross(k_axis, vertical)
    if np.linalg.norm(east) < C.ZERO_TOLERANCE:
        east = np.array([0.0, 1.0, 0.0])
    east = east / np.linalg.norm(east)
    north = np.cross(vertical, east)
    north = north / np.linalg.norm(north)
    return vertical, east, north


def compute_local_horizontal(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    vertical = compute_local_vertical(r)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    if v_rel_norm < C.ZERO_TOLERANCE:
        return vertical
    return v_rel / v_rel_norm


def gamma_profile_from_altitude(altitude: float) -> float:
    """
    Continuous, smooth gamma profile (gravity turn) based on altitude.

    Uses a tanh-based shaping function to ensure C1 continuity (smooth derivatives)
    which helps the Attitude Control System track the command without saturating.

    Gamma starts at 90 deg (Vertical) and smoothly decays to ~18 deg at MECO.
    """
    gamma_start = 90.0
    gamma_final = 18.0

    turn_start_alt = 1000.0   # m
    turn_scale = 45000.0

    if altitude < turn_start_alt:
        return np.radians(gamma_start)

    h_norm = (altitude - turn_start_alt) / turn_scale
    progress = np.tanh(h_norm)
    gamma_deg = gamma_start + (gamma_final - gamma_start) * progress

    return np.radians(gamma_deg)


def compute_blend_parameter(altitude: float) -> float:
    """
    Blend factor (0 → vertical ascent, 1 → prograde) based on altitude.

    A smooth ramp is used so guidance phase labels and thrust blending
    remain continuous.
    """
    start = C.GRAVITY_TURN_START_ALTITUDE
    width = C.GRAVITY_TURN_TRANSITION_RANGE

    if altitude <= start:
        return 0.0
    if altitude >= start + width:
        return 1.0

    x = (altitude - start) / width
    return x * x * (3.0 - 2.0 * x)


# =============================================================================
# CORE GUIDANCE FUNCTIONS (accept/return GuidanceState)
# =============================================================================

def compute_desired_thrust_direction(
    r: np.ndarray, v: np.ndarray, t: float,
    gs: GuidanceState = None, dt: float = C.DT
) -> Tuple[np.ndarray, float, float, GuidanceState]:
    """Compute desired thrust direction for ascent guidance.

    Args:
        r, v, t: Vehicle state
        gs: Guidance state (uses module default if None)
        dt: Time step

    Returns:
        (thrust_dir, gamma_cmd_deg, gamma_meas_deg, updated_gs)
    """
    if gs is None:
        gs = _default_gs

    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    vertical, east, north = compute_local_frame(r)

    # 1. State-Dependent Gamma Command (Smooth)
    gamma_target = gamma_profile_from_altitude(altitude)
    gamma_target_deg = float(np.degrees(gamma_target))

    # 2. Measurement & PID correction
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))

    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec = v_rel - v_vert * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))

    if v_rel_norm < 1e-3:
        gamma_meas_deg = 90.0
    else:
        gamma_meas_deg = float(np.degrees(np.arctan2(v_vert, v_horiz)))

    error = gamma_target_deg - gamma_meas_deg
    gamma_rate = (gamma_meas_deg - gs.prev_gamma_meas) / max(dt, 1e-3)

    # PID Gains
    kp, ki, kd = 0.85, 0.05, 0.12
    gs.gamma_int += error * dt
    gs.gamma_int = float(np.clip(gs.gamma_int, -20.0, 20.0))
    gamma_raw = gamma_target_deg + kp * error + ki * gs.gamma_int - kd * gamma_rate

    # 3. Dynamic Pressure & Zero-Lift Logic (Prograde Locking)
    _, _, rho, _ = compute_atmosphere_properties(altitude)
    q_dyn = 0.5 * rho * v_rel_norm**2

    if v_rel_norm > 1.0:
        prograde = v_rel / v_rel_norm
    else:
        prograde = vertical

    gamma_cmd_clamped = float(np.clip(gamma_raw, 10.0, 90.0))
    pitch_cmd_rad = np.radians(90.0 - gamma_cmd_clamped)

    if v_horiz > 1.0:
        horiz_axis = v_horiz_vec / v_horiz
    else:
        horiz_axis = east

    guidance_dir = np.cos(pitch_cmd_rad) * vertical + np.sin(pitch_cmd_rad) * horiz_axis
    guidance_dir /= np.linalg.norm(guidance_dir)

    # 4. Mode Mixing: Blend from Guidance to Prograde based on ALTITUDE
    mix_start_alt = 1000.0
    mix_end_alt = 10000.0

    if altitude < mix_start_alt:
        w_prograde = 0.0
    elif altitude > mix_end_alt:
        w_prograde = 1.0
    else:
        w_prograde = (altitude - mix_start_alt) / (mix_end_alt - mix_start_alt)

    if q_dyn > 20000.0:
         w_prograde = 1.0

    thrust_dir_mixed = (1.0 - w_prograde) * guidance_dir + w_prograde * prograde

    norm_mixed = np.linalg.norm(thrust_dir_mixed)
    if norm_mixed > 1e-6:
        thrust_dir_mixed /= norm_mixed
    else:
        thrust_dir_mixed = guidance_dir

    thrust_dir = thrust_dir_mixed

    # 5. AoA limiting
    if q_dyn > 5000.0:
        max_aoa = np.radians(3.0)
    elif altitude < 50000.0:
        max_aoa = np.radians(5.0)
    else:
        max_aoa = np.radians(10.0)

    thrust_dir = _limit_aoa(thrust_dir, v_rel, max_aoa)

    cos_p = np.dot(thrust_dir, vertical)
    gamma_cmd_clamped = np.degrees(np.arcsin(np.clip(cos_p, -1.0, 1.0)))

    gs.prev_gamma_meas = gamma_meas_deg

    return thrust_dir, gamma_cmd_clamped, gamma_meas_deg, gs


def compute_guidance_output(
    r: np.ndarray, v: np.ndarray, t: float, m: float,
    meco_mass: float = None,
    gs: GuidanceState = None
) -> Tuple[GuidanceOutput, GuidanceState]:
    """
    Compute ascent guidance output for Stage 1 powered flight.

    Args:
        r: Position vector (ECI, m)
        v: Velocity vector (ECI, m/s)
        t: Current time (s)
        m: Current vehicle mass (kg)
        meco_mass: Mass at which MECO occurs.
        gs: Per-vehicle guidance state. Uses module default if None.

    Returns:
        (guidance_output_dict, updated_guidance_state)
    """
    if gs is None:
        gs = _default_gs
    if meco_mass is None:
        meco_mass = C.DRY_MASS + C.STAGE1_LANDING_FUEL_RESERVE

    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    thrust_dir, gamma_cmd, gamma_meas, gs = compute_desired_thrust_direction(r, v, t, gs)

    alpha = compute_blend_parameter(altitude)
    if alpha < 0.01:
        if C.PITCHOVER_START_ALTITUDE <= altitude <= C.PITCHOVER_END_ALTITUDE:
            phase = "PITCHOVER"
        else:
            phase = "VERTICAL_ASCENT"
    elif alpha < 0.99:
        phase = "GRAVITY_TURN"
    else:
        phase = "PROGRADE"

    thrust_on = (m > meco_mass)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))

    # Max-Q shaping
    _, _, rho, _ = compute_atmosphere_properties(altitude)
    q_dyn = 0.5 * rho * (v_rel_norm ** 2)
    q_limit = C.MAX_DYNAMIC_PRESSURE
    if q_dyn > 0.8 * q_limit:
        throttle = float(np.clip(np.sqrt(q_limit / (q_dyn + 1e-6)), 0.55, 1.0))
    else:
        throttle = 1.0

    if v_rel_norm < C.ZERO_TOLERANCE:
        prograde = compute_local_vertical(r)
    else:
        prograde = v_rel / v_rel_norm

    # Record last ascent direction for coast/orbit insertion phase continuity
    gs.last_ascent_direction = thrust_dir.copy()

    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(vertical, thrust_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))
    gamma_angle = np.radians(gamma_cmd)

    output = {
        'thrust_direction': thrust_dir,
        'phase': phase,
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'gamma_angle': gamma_angle,
        'gamma_command_deg': gamma_cmd,
        'gamma_measured_deg': gamma_meas,
        'velocity_tilt_deg': gamma_meas,
        'blend_alpha': alpha,
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': prograde,
        'throttle': throttle,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm
    }
    return output, gs


def compute_orbit_insertion_guidance(
    r: np.ndarray, v: np.ndarray, t: float, m: float,
    gs: GuidanceState = None
) -> Tuple[GuidanceOutput, GuidanceState]:
    """
    Guidance logic for S2 orbit insertion burn.

    Steering law: flight-path-angle-tracking pitch program with altitude feedback.
    """
    if gs is None:
        gs = _default_gs

    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    v_inertial = float(np.linalg.norm(v))
    r_mag = float(np.linalg.norm(r))
    vertical = compute_local_vertical(r)

    v_circular = float(np.sqrt(C.MU_EARTH / r_mag))
    v_deficit = v_circular - v_inertial

    v_radial = float(np.dot(v, vertical))
    v_horiz_vec = v - v_radial * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))

    if v_horiz > 10.0:
        horiz_hat = v_horiz_vec / v_horiz
    else:
        _, east, _ = compute_local_frame(r)
        horiz_hat = east

    # Initialize OI state on first call
    if gs.oi_start_time is None:
        gs.oi_start_time = t
        if gs.last_ascent_direction is not None:
            gs.oi_start_direction = gs.last_ascent_direction.copy()
        elif v_inertial > 10.0:
            gs.oi_start_direction = v / v_inertial
        else:
            gs.oi_start_direction = vertical.copy()

    if v_inertial > 10.0:
        prograde = v / v_inertial
    else:
        prograde = vertical

    v_progress = float(np.clip(v_horiz / v_circular, 0.0, 1.0))
    gamma_current = float(np.degrees(np.arctan2(v_radial, max(v_horiz, 1.0))))

    pitch_target_above_horiz = 6.0 * (1.0 - v_progress) ** 0.5

    pitch_correction_deg = 0.0
    if v_radial < 0.0:
        pitch_correction_deg += min(15.0, -v_radial * 0.15)

    min_safe_alt = 80000.0
    if altitude < min_safe_alt + 20000.0:
        alt_margin = (altitude - min_safe_alt) / 20000.0
        alt_margin = float(np.clip(alt_margin, 0.0, 1.0))
        pitch_correction_deg += 20.0 * (1.0 - alt_margin)

    if v_deficit < 4000.0 and v_radial > 10.0:
        flatten_factor = float(np.clip(1.0 - v_deficit / 4000.0, 0.0, 1.0))
        pitch_down_deg = flatten_factor * min(20.0, v_radial * 0.05)
        pitch_target_above_horiz -= pitch_down_deg

    pitch_target_above_horiz += pitch_correction_deg
    pitch_target_above_horiz = float(np.clip(pitch_target_above_horiz, -10.0, 75.0))

    pitch_target_deg = float(np.clip(pitch_target_above_horiz, -10.0, 75.0))
    pitch_target_rad = np.radians(pitch_target_deg)

    target_dir = np.cos(pitch_target_rad) * horiz_hat + np.sin(pitch_target_rad) * vertical
    target_dir = target_dir / np.linalg.norm(target_dir)

    dt_since_start = t - gs.oi_start_time
    ramp_duration = 20.0

    if dt_since_start < ramp_duration and gs.oi_start_direction is not None:
        x = dt_since_start / ramp_duration
        blend = x * x * (3.0 - 2.0 * x)
        start_dir = gs.oi_start_direction
        blended = (1.0 - blend) * start_dir + blend * target_dir
        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-9:
            desired_dir = blended / blended_norm
        else:
            desired_dir = target_dir
    else:
        desired_dir = target_dir

    max_aoa = np.radians(15.0)
    desired_dir = _limit_aoa(desired_dir, v, max_aoa)

    s2_prop_remaining = m - C.STAGE2_DRY_MASS
    thrust_on = s2_prop_remaining > 0.0 and v_deficit > 10.0

    if v_deficit < 100.0 and v_deficit > 10.0:
        throttle = float(np.clip(v_deficit / 100.0, 0.1, 1.0))
    elif v_deficit <= 10.0:
        throttle = 0.0
        thrust_on = False
    else:
        throttle = 1.0

    cos_pitch = np.clip(np.dot(vertical, desired_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))

    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec_rel = v_rel - v_vert * vertical
    v_horiz_rel = float(np.linalg.norm(v_horiz_vec_rel))
    gamma_actual = float(np.degrees(np.arctan2(v_vert, max(v_horiz_rel, 1e-6))))

    if v_inertial > 1.0:
        prograde = v / v_inertial
    else:
        prograde = vertical

    output = {
        'thrust_direction': desired_dir,
        'phase': "ORBIT_INSERTION",
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'gamma_angle': np.radians(gamma_actual),
        'gamma_command_deg': gamma_actual,
        'gamma_measured_deg': gamma_actual,
        'velocity_tilt_deg': gamma_actual,
        'blend_alpha': 1.0,
        'altitude': altitude,
        'velocity': v_inertial,
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': prograde,
        'throttle': throttle,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm
    }
    return output, gs


def compute_coast_guidance(
    r: np.ndarray, v: np.ndarray, t: float, m: float,
    gs: GuidanceState = None
) -> Tuple[GuidanceOutput, GuidanceState]:
    """
    Guidance logic for Phase II (Coast).

    - Thrust: OFF
    - Attitude: Hold last ascent direction, then blend toward inertial prograde.
    """
    if gs is None:
        gs = _default_gs

    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    v_inertial = float(np.linalg.norm(v))

    if v_inertial > 1.0:
        prograde = v / v_inertial
    else:
        prograde = compute_local_vertical(r)

    if gs.last_ascent_direction is not None:
        desired_dir = gs.last_ascent_direction.copy()
        max_aoa = np.radians(15.0)
        desired_dir = _limit_aoa(desired_dir, v, max_aoa)
    else:
        desired_dir = prograde

    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(vertical, desired_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))

    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec = v_rel - v_vert * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))
    gamma_actual = float(np.degrees(np.arctan2(v_vert, v_horiz)))

    output = {
        'thrust_direction': desired_dir,
        'phase': "COAST_PHASE",
        'thrust_on': False,
        'pitch_angle': pitch_angle,
        'gamma_angle': np.radians(gamma_actual),
        'gamma_command_deg': gamma_actual,
        'gamma_measured_deg': gamma_actual,
        'velocity_tilt_deg': gamma_actual,
        'blend_alpha': 1.0,
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': desired_dir,
        'throttle': 0.0,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm
    }
    return output, gs


def _compute_boostback_direction(
    r: np.ndarray,
    v: np.ndarray,
    launch_site: np.ndarray,
    apogee_target_m: float = 160000.0,
) -> np.ndarray:
    """
    Compute optimal boostback thrust direction to return to launch site.

    Uses the velocity-to-be-gained (VTG) method.
    """
    vertical = compute_local_vertical(r)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1.0:
        return -vertical

    # Base command: retrograde to reduce total kinetic energy.
    thrust_dir = -v / v_norm

    # Add horizontal bias toward launch site for RTLS shaping.
    r_to_site = launch_site - r
    r_horiz = r_to_site - np.dot(r_to_site, vertical) * vertical
    r_horiz_norm = float(np.linalg.norm(r_horiz))
    if r_horiz_norm > 1e3:
        toward_site = r_horiz / r_horiz_norm
        site_weight = float(np.clip(r_horiz_norm / 800000.0, 0.08, 0.30))
        thrust_dir = (1.0 - site_weight) * thrust_dir + site_weight * toward_site
        thrust_dir /= max(np.linalg.norm(thrust_dir), 1e-6)

    # Enforce apogee cap by biasing thrust inward while ascending.
    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    g_local = C.MU_EARTH / max(np.linalg.norm(r) ** 2, 1.0)
    v_radial = float(np.dot(v, vertical))
    h_apogee_pred = estimate_ballistic_apogee(altitude, v_radial, g_local)
    apogee_excess = max(0.0, h_apogee_pred - apogee_target_m)
    if v_radial > 0.0:
        down_weight = float(np.clip(v_radial / 900.0 + apogee_excess / 120000.0, 0.25, 0.90))
        thrust_dir = (1.0 - down_weight) * thrust_dir + down_weight * (-vertical)
        thrust_dir /= max(np.linalg.norm(thrust_dir), 1e-6)

    return thrust_dir


def compute_booster_guidance(
    r: np.ndarray, v: np.ndarray, t: float, m: float, phase: str,
    gs: GuidanceState = None,
    config: Optional[SimulationConfig] = None,
) -> Tuple[GuidanceOutput, GuidanceState]:
    """
    Guidance logic for Booster Recovery (Phase III-B).

    Physics-based guidance for each sub-phase:
    FLIP, BOOSTBACK, COAST, ENTRY, LANDING.
    """
    if gs is None:
        gs = _default_gs

    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    cfg = config

    thrust_on = False
    throttle = 0.0
    desired_dir = compute_local_vertical(r)

    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))

    prograde = v_rel / v_rel_norm if v_rel_norm > 1.0 else compute_local_vertical(r)
    retrograde = -prograde
    vertical = compute_local_vertical(r)
    propellant_remaining = booster_propellant_remaining(m)
    landing_reserve = cfg.booster_landing_reserve_kg if cfg is not None else 10000.0
    entry_min_alt = cfg.booster_entry_burn_min_altitude_m if cfg is not None else 35000.0
    entry_min_speed = cfg.booster_entry_burn_min_speed_mps if cfg is not None else 800.0
    apogee_target_m = (cfg.booster_apogee_target_km * 1000.0) if cfg is not None else 160000.0
    ignite_safety = cfg.booster_landing_ignition_safety_factor if cfg is not None else 1.5
    burn_prediction = estimate_suicide_burn(
        r, v, m, C.THRUST_MAGNITUDE, safety_factor=ignite_safety
    )

    target_downrange_km = cfg.booster_landing_target_downrange_km if cfg is not None else 9.65
    launch_site = target_landing_site_eci(t, target_downrange_km)

    if phase == "BOOSTER_FLIP":
        desired_dir = retrograde
        thrust_on = False

    elif phase == "BOOSTER_BOOSTBACK":
        desired_dir = _compute_boostback_direction(r, v, launch_site, apogee_target_m=apogee_target_m)
        thrust_on = True
        throttle = 0.55

    elif phase == "BOOSTER_COAST":
        desired_dir = retrograde
        thrust_on = False

    elif phase == "BOOSTER_ENTRY":
        desired_dir = retrograde

        entry_burn_on = (
            altitude > entry_min_alt and
            v_rel_norm > entry_min_speed and
            propellant_remaining > landing_reserve
        )

        if entry_burn_on:
            thrust_on = True
            throttle = 0.7
        else:
            thrust_on = False

    elif phase == "BOOSTER_LANDING":
        burn_params = burn_prediction

        v_descent = -float(np.dot(v, vertical))
        v_horiz_vec = v - np.dot(v, vertical) * vertical
        v_horiz_mag = float(np.linalg.norm(v_horiz_vec))
        v_total = float(np.linalg.norm(v))

        if burn_params['ignite']:
            gs.booster_landing_burn_started = True
            # Powered descent follows air-relative retrograde so touchdown
            # converges in the rotating Earth frame.
            desired_dir = retrograde if v_rel_norm > 1.0 else vertical

            thrust_on = True
            throttle = float(np.clip(burn_params['throttle'], 0.35, 1.0))
            if v_horiz_mag > 250.0:
                throttle = max(throttle, 0.75)
            elif v_horiz_mag > 100.0:
                throttle = max(throttle, 0.55)
        else:
            desired_dir = retrograde
            thrust_on = False

    # Common flight-path angle computation
    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec = v_rel - v_vert * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))

    if v_rel_norm > 1.0:
        gamma_deg = float(np.degrees(np.arctan2(v_vert, max(v_horiz, 1e-6))))
    else:
        gamma_deg = 90.0 if np.dot(v, vertical) >= 0 else -90.0

    cos_pitch = np.clip(np.dot(vertical, desired_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))

    output = {
        'thrust_direction': desired_dir,
        'phase': phase,
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'gamma_angle': np.radians(gamma_deg),
        'gamma_command_deg': gamma_deg,
        'gamma_measured_deg': gamma_deg,
        'velocity_tilt_deg': float(np.degrees(np.arctan2(v_horiz, abs(v_vert)))) if v_rel_norm > 1.0 else 0.0,
        'blend_alpha': 1.0,
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': prograde,
        'throttle': throttle,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm,
        'propellant_remaining_kg': propellant_remaining,
        'ignition_altitude_prediction_m': burn_prediction['burn_altitude'],
    }
    return output, gs
