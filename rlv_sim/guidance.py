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

    Gamma starts at 90 deg (Vertical) and smoothly decays toward the target
    pitch angle at MECO.  The scale height controls how aggressively the
    vehicle pitches over during the ascent.  A faster turn (lower scale)
    produces a higher horizontal velocity at MECO, which is needed to reach
    the typical 45-60 deg pitch-from-vertical at engine cutoff.
    """
    gamma_start = 90.0
    gamma_final = 10.0

    turn_start_alt = 500.0    # m — begin turn soon after clearing pad
    turn_scale = 18000.0      # m — aggressive but not structural-limit-critical

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
    #    Delay full prograde handover until well above Max-Q so the gamma
    #    PID has authority to drive the pitch program through the critical
    #    turn regime.  Below mix_start we use pure guidance; between start
    #    and end a linear ramp blends to prograde; above end the vehicle
    #    is velocity-locked (zero-lift gravity turn).
    mix_start_alt = 2000.0
    mix_end_alt = 40000.0

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

    Two-phase strategy targeting a circular orbit at ``TARGET_ORBIT_ALTITUDE``:

    Phase 1 — **Apogee raise**:  Burn mostly prograde (with altitude-keeping
    pitch-up near the floor) to raise apogee toward the target altitude.
    Engine shuts off once apogee >= target.

    Phase 2 — **Circularisation**:  Coast toward apogee, then re-ignite
    prograde to circularise (eliminate residual radial velocity and close
    any remaining velocity deficit at the target radius).
    """
    if gs is None:
        gs = _default_gs

    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    v_inertial = float(np.linalg.norm(v))
    r_mag = float(np.linalg.norm(r))
    vertical = compute_local_vertical(r)

    target_alt = C.TARGET_ORBIT_ALTITUDE          # 400 km
    r_target = C.R_EARTH + target_alt
    v_circ_target = float(np.sqrt(C.MU_EARTH / r_target))
    v_circ_here = float(np.sqrt(C.MU_EARTH / r_mag))
    v_deficit_here = v_circ_here - v_inertial

    v_radial = float(np.dot(v, vertical))
    v_horiz_vec = v - v_radial * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))

    if v_horiz > 10.0:
        horiz_hat = v_horiz_vec / v_horiz
    else:
        _, east, _ = compute_local_frame(r)
        horiz_hat = east

    # ── Initialise OI state on first call ────────────────────────────
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

    # ── Orbital element helpers ──────────────────────────────────────
    spec_energy = 0.5 * v_inertial**2 - C.MU_EARTH / r_mag
    if abs(spec_energy) > 1.0:
        a_sma = -C.MU_EARTH / (2.0 * spec_energy)
    else:
        a_sma = r_mag
    h_vec = np.cross(r, v)
    h_mag = float(np.linalg.norm(h_vec))
    if a_sma > 0 and C.MU_EARTH * a_sma > 0:
        ecc = float(np.sqrt(max(0.0, 1.0 - h_mag**2 / (C.MU_EARTH * a_sma))))
    else:
        ecc = 1.0
    apogee_alt = a_sma * (1.0 + ecc) - C.R_EARTH if a_sma > 0 else altitude
    perigee_alt = a_sma * (1.0 - ecc) - C.R_EARTH if a_sma > 0 else altitude

    s2_prop_remaining = m - C.STAGE2_DRY_MASS

    # ── Decide phase: raising apogee vs circularising ────────────────
    # Apogee raise complete once apogee is at or above target (with margin)
    apogee_reached_target = apogee_alt >= target_alt - 5000.0
    # Near apogee: radial velocity nearly zero and altitude near target.
    # Use tight v_radial threshold so we don't ignite too early (which
    # would overshoot apogee by adding prograde velocity while still climbing).
    near_apogee = abs(v_radial) < 15.0 and altitude > target_alt * 0.90

    # Circularisation: apogee at target and we are near apogee
    circularising = apogee_reached_target and near_apogee

    # ── Pitch program ────────────────────────────────────────────────
    if circularising:
        # Phase 2: Circularise at target altitude — burn prograde to
        # close the velocity deficit.
        v_deficit_target = v_circ_target - v_inertial
        pitch_target_deg = 0.0   # pure prograde (horizontal)
        # Small pitch correction to damp any residual radial velocity
        if abs(v_radial) > 2.0:
            pitch_target_deg -= float(np.clip(v_radial * 0.3, -8.0, 8.0))
        thrust_on = s2_prop_remaining > 0.0 and v_deficit_target > 3.0
        if v_deficit_target < 100.0 and v_deficit_target > 3.0:
            throttle = float(np.clip(v_deficit_target / 100.0, 0.05, 1.0))
        elif v_deficit_target <= 3.0:
            throttle = 0.0
            thrust_on = False
        else:
            throttle = 1.0
    elif apogee_reached_target:
        # Coasting to apogee — no thrust, hold prograde attitude
        pitch_target_deg = 0.0
        thrust_on = False
        throttle = 0.0
    else:
        # Phase 1: Raise apogee — burn mostly prograde with altitude floor
        # Baseline: small positive pitch to gain altitude while accelerating
        alt_deficit = target_alt - apogee_alt
        alt_frac = float(np.clip(alt_deficit / target_alt, 0.0, 1.0))
        pitch_target_deg = 4.0 * alt_frac ** 0.3

        # Altitude floor protection: if below 80 km, pitch up harder
        min_safe_alt = 80000.0
        if altitude < min_safe_alt + 20000.0:
            alt_margin = float(np.clip(
                (altitude - min_safe_alt) / 20000.0, 0.0, 1.0))
            pitch_target_deg += 20.0 * (1.0 - alt_margin)

        # If descending, pitch up to arrest descent
        if v_radial < 0.0:
            pitch_target_deg += float(np.clip(-v_radial * 0.15, 0.0, 15.0))

        # If climbing fast and apogee is already near target, flatten out
        if alt_deficit < 50000.0 and v_radial > 10.0:
            flatten = float(np.clip(1.0 - alt_deficit / 50000.0, 0.0, 1.0))
            pitch_target_deg -= flatten * min(15.0, v_radial * 0.04)

        pitch_target_deg = float(np.clip(pitch_target_deg, -5.0, 75.0))
        thrust_on = s2_prop_remaining > 0.0
        throttle = 1.0

    pitch_target_rad = np.radians(pitch_target_deg)
    target_dir = (np.cos(pitch_target_rad) * horiz_hat
                  + np.sin(pitch_target_rad) * vertical)
    target_dir = target_dir / np.linalg.norm(target_dir)

    # ── Smooth ramp from ascent attitude ─────────────────────────────
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
    Compute boostback thrust direction to return to launch site.

    Primary objective: reverse the horizontal velocity so the booster
    follows a ballistic arc back toward the landing pad.  The thrust
    direction is the *anti-horizontal-velocity* vector — i.e. directly
    opposing the component of velocity that carries the booster away
    from the pad.

    A secondary downward bias is blended in only when the predicted
    ballistic apogee exceeds the cap, preventing the booster from
    climbing excessively.
    """
    vertical = compute_local_vertical(r)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1.0:
        return -vertical

    # Decompose velocity into vertical and horizontal components
    v_radial = float(np.dot(v, vertical))
    v_horiz_vec = v - v_radial * vertical
    v_horiz_mag = float(np.linalg.norm(v_horiz_vec))

    # Primary: oppose horizontal velocity (the main RTLS requirement)
    if v_horiz_mag > 5.0:
        thrust_dir = -v_horiz_vec / v_horiz_mag
    else:
        # Horizontal velocity already near zero — point retrograde
        thrust_dir = -v / max(v_norm, 1.0)

    # Secondary: apogee cap — blend downward only when predicted apogee
    # significantly exceeds the target.  This prevents wasting most of
    # the burn on vertical deceleration.
    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    g_local = C.MU_EARTH / max(np.linalg.norm(r) ** 2, 1.0)
    h_apogee_pred = estimate_ballistic_apogee(altitude, v_radial, g_local)
    apogee_excess = max(0.0, h_apogee_pred - apogee_target_m)
    if v_radial > 0.0 and apogee_excess > 20000.0:
        # Proportional downward bias — small unless apogee is way too high
        down_weight = float(np.clip(apogee_excess / 200000.0, 0.0, 0.35))
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
        throttle = 0.85

    elif phase == "BOOSTER_COAST":
        desired_dir = retrograde
        thrust_on = False

    elif phase == "BOOSTER_ENTRY":
        # Entry burn: primarily retrograde to slow down, with a horizontal
        # bias toward the landing site to reduce site error accumulated
        # from imperfect boostback.
        r_to_site = launch_site - r
        r_to_site_horiz = r_to_site - np.dot(r_to_site, vertical) * vertical
        site_dist = float(np.linalg.norm(r_to_site_horiz))
        if site_dist > 1000.0:
            toward_site = r_to_site_horiz / site_dist
            # Blend toward site proportional to how far off we are
            site_bias = float(np.clip(site_dist / 200000.0, 0.0, 0.25))
            desired_dir = (1.0 - site_bias) * retrograde + site_bias * toward_site
            desired_dir /= max(np.linalg.norm(desired_dir), 1e-9)
        else:
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

        v_descent = -float(np.dot(v_rel, vertical))
        v_horiz_vec = v_rel - np.dot(v_rel, vertical) * vertical
        v_horiz_mag = float(np.linalg.norm(v_horiz_vec))

        # --- Landing burn persistence ---
        # Once ignited, keep the burn on until touchdown or propellant exhaustion.
        # The energy-based ignite flag can momentarily flip False once the vehicle
        # decelerates fast enough that the kinematics estimate h_ignite * SF drops
        # below current altitude (variable-mass effect: T/m rises as fuel burns,
        # so actual mean deceleration exceeds the constant-a estimate used by
        # estimate_suicide_burn). Without this latch the burn oscillates every
        # timestep and the vehicle impacts at near-ballistic speed.
        if burn_params['ignite'] or gs.booster_landing_burn_started:
            gs.booster_landing_burn_started = True

            # --- Lateral velocity zeroing ---
            # Blend thrust direction from retrograde toward vertical as
            # horizontal velocity decreases.  This drives v_horizontal -> 0
            # at touchdown instead of just pointing retrograde.
            #
            # At high v_horiz (>200 m/s): steer at atan(v_h/v_d) to kill both
            # At low v_horiz (<1 m/s):    pure vertical for final descent
            # Uses air-relative velocity so touchdown converges in the
            # rotating Earth frame.
            if v_horiz_mag > 1.0 and v_descent > 1.0:
                horiz_retro = -v_horiz_vec / v_horiz_mag
                steer_angle = float(np.arctan2(v_horiz_mag, max(v_descent, 1.0)))
                steer_angle = min(steer_angle, np.radians(45.0))
                desired_dir = (np.cos(steer_angle) * vertical
                               + np.sin(steer_angle) * horiz_retro)
                desired_dir /= max(np.linalg.norm(desired_dir), 1e-9)
            elif v_horiz_mag > 1.0:
                desired_dir = retrograde if v_rel_norm > 1.0 else vertical
            else:
                desired_dir = vertical

            thrust_on = True

            # Throttle — variable-mass continuous control law:
            #
            #   T_req(t) = m(t) * [v_eff²/(2h) + g_local]
            #   θ(t)     = T_req(t) / T_max
            #
            # Physical meaning: at each instant, command the thrust that would
            # produce exactly the net deceleration needed to stop in the
            # remaining altitude (kinematic estimate), plus gravity compensation.
            # Using current m(t) makes this fully mass-aware — as propellant
            # burns, m decreases, T/m increases, and θ decreases naturally
            # without needing an external correction.
            #
            # Note: we compute this directly from state rather than reading
            # burn_params['throttle'], because burn_params['throttle'] is set to
            # 0.0 when burn_params['ignite'] = False (i.e., when the persistence
            # latch is holding the burn on).  Computing directly ensures the
            # correct throttle in both the ignite=True and latch-only cases.
            g_loc = float(C.MU_EARTH / (float(np.linalg.norm(r)) ** 2))
            t_accel = float(C.THRUST_MAGNITUDE / max(m, 1.0))
            v_eff = float(np.sqrt(max(v_descent, 0.0) ** 2 + v_horiz_mag ** 2))
            if v_eff < 1.0:
                throttle = 0.0  # Effectively at rest — kill thrust
            else:
                # No lower clip: the formula can smoothly approach 0 as
                # v and h simultaneously → 0 (classical suicide-burn curve).
                # Clamping to a positive minimum would over-brake at low
                # altitude, stopping the vehicle above h=0 and causing a
                # free-fall impact.  Allowing throttle → 0 lets the engine
                # follow the exact variable-mass touchdown trajectory.
                a_req = v_eff ** 2 / (2.0 * max(altitude, 0.5)) + g_loc
                throttle = float(np.clip(a_req / max(t_accel, 1e-6), 0.0, 1.0))
            # Floor throttle when high lateral speed demands more braking
            if v_horiz_mag > 200.0:
                throttle = max(throttle, 0.80)
            elif v_horiz_mag > 80.0:
                throttle = max(throttle, 0.60)
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
