"""
Guidance module implementing altitude-based gamma profile with PID tracking
and thrust-direction generation for gravity turn.
"""

import numpy as np

from . import constants as C
from .forces import compute_atmosphere_properties
from .types import GuidanceOutput
from .utils import compute_relative_velocity


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
    """Piecewise-linear gamma profile (from horizontal) based on altitude (m).
    
    Optimized trajectory targeting ~110km at MECO without guidance divergence.
    Moderate aggression: earlier turn than balanced, but achievable.
    """
    alt_km = altitude / 1000.0
    # Optimized profile: 90° -> 18° over 100km (reaches ~110km at MECO)
    pts = [
        (0.0, 90.0),    # Vertical at launch
        (3.0, 84.0),    # Start turn early
        (10.0, 74.0),   # Accelerate gravity turn
        (25.0, 58.0),   # Moderate turn
        (45.0, 40.0),   # Push toward horizontal
        (65.0, 28.0),   # Turn for horizontal velocity
        (85.0, 21.0),   # Near horizontal
        (100.0, 18.0),  # Target gamma at 100km
    ]
    if alt_km <= pts[0][0]:
        return np.radians(pts[0][1])
    for i in range(len(pts) - 1):
        a0, g0 = pts[i]
        a1, g1 = pts[i + 1]
        if alt_km <= a1:
            frac = (alt_km - a0) / (a1 - a0)
            gamma_deg = g0 + frac * (g1 - g0)
            return np.radians(gamma_deg)
    return np.radians(pts[-1][1])


def compute_blend_parameter(altitude: float) -> float:
    """
    Blend factor (0 → vertical ascent, 1 → prograde) based on altitude.

    A smooth ramp is used so guidance phase labels and thrust blending
    remain continuous. Transition range is set by GRAVITY_TURN_START_ALTITUDE
    and GRAVITY_TURN_TRANSITION_RANGE.
    """
    start = C.GRAVITY_TURN_START_ALTITUDE
    end = start + C.GRAVITY_TURN_TRANSITION_RANGE
    if altitude <= start:
        return 0.0
    if altitude >= end:
        return 1.0
    return (altitude - start) / (end - start)


# PID state (encapsulated in module-level variables - reset between runs)
_pid_state = {
    'prev_gamma_meas': 90.0,
    'gamma_int': 0.0
}


def reset_guidance():
    """Reset guidance PID state. Call this before starting a new simulation."""
    global _pid_state
    _pid_state['prev_gamma_meas'] = 90.0
    _pid_state['gamma_int'] = 0.0


def compute_desired_thrust_direction(r: np.ndarray, v: np.ndarray, t: float, dt: float = C.DT):
    global _pid_state

    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    vertical, east, north = compute_local_frame(r)

    # Desired heading: east
    heading_dir = east

    # Air-relative velocity and prograde
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    prograde = heading_dir if v_rel_norm < C.ZERO_TOLERANCE else v_rel / v_rel_norm

    # Blend heading vs prograde (earlier crossover to align with velocity)
    blend = float(np.clip(v_rel_norm / 400.0, 0.0, 1.0))
    horizontal_dir = (1.0 - blend) * heading_dir + blend * prograde
    horizontal_dir = horizontal_dir - np.dot(horizontal_dir, vertical) * vertical
    hnorm = np.linalg.norm(horizontal_dir)
    if hnorm < C.ZERO_TOLERANCE:
        horizontal_dir = heading_dir
    else:
        hnorm = np.linalg.norm(horizontal_dir)
        horizontal_dir /= hnorm

    # Gamma tracking
    gamma_target = gamma_profile_from_altitude(altitude)
    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec = v_rel - v_vert * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))
    if v_rel_norm < 1e-3:
        gamma_meas_deg = 90.0
    else:
        # gamma is angle from horizontal (90 = vertical)
        gamma_meas_deg = float(np.degrees(np.arctan2(v_vert, v_horiz)))
    gamma_target_deg = float(np.degrees(gamma_target))

    error = gamma_target_deg - gamma_meas_deg
    gamma_rate = (gamma_meas_deg - _pid_state['prev_gamma_meas']) / max(dt, 1e-3)

    # Improved PID tuning for gradual gamma profile
    # Reduced kp to prevent overshoot, increased ki for better tracking
    kp, ki, kd = 0.85, 0.05, 0.12
    _pid_state['gamma_int'] += error * dt
    _pid_state['gamma_int'] = float(np.clip(_pid_state['gamma_int'], -30.0, 30.0))
    gamma_raw = gamma_target_deg + kp * error + ki * _pid_state['gamma_int'] - kd * gamma_rate
    gamma_cmd = float(np.clip(gamma_raw, gamma_target_deg - 15.0, gamma_target_deg + 15.0))

    # gamma floor relaxation (deg from horizontal) - optimized for ~110km MECO
    gamma_floor = float(np.interp(
        altitude,
        [0.0, 3000.0, 10000.0, 25000.0, 45000.0, 65000.0, 85000.0, 100000.0],
        [84.0, 78.0, 69.0, 54.0, 37.0, 25.0, 19.0, 16.0],
    ))
    gamma_cmd = float(np.clip(gamma_cmd, gamma_floor, 90.0))
    _pid_state['prev_gamma_meas'] = gamma_meas_deg

    # Feed-forward pitch from vertical (gamma from horizontal)
    theta_ff = np.pi/2 - np.radians(gamma_cmd)
    theta_ff = float(np.clip(theta_ff, 0.0, np.radians(89.0)))

    # Corrective bias: if gamma is too steep, tilt more toward horizontal
    gamma_err = gamma_cmd - gamma_meas_deg
    bias_gain = 0.4 if altitude < 60000.0 else 0.65
    theta_bias = np.radians(-bias_gain * gamma_err)
    theta = theta_ff + theta_bias

    # Pitch envelope (altitude dependent)
    theta_cap = float(np.interp(
        altitude,
        [0.0, 500.0, 3000.0, 10000.0, 25000.0, 60000.0],
        np.radians([15.0, 25.0, 40.0, 60.0, 75.0, 85.0])
    ))

    # Safeguards for near-zero or negative vertical rates
    # Allow larger pitch angles to achieve proper trajectory
    if v_vert < 20.0 and altitude < 3000.0:
        theta_cap = min(theta_cap, np.radians(35.0))
    if v_vert < 0.0 and altitude < 30000.0:
        theta_cap = min(theta_cap, np.radians(40.0))

    theta = float(np.clip(theta, 0.0, theta_cap))

    thrust_dir = np.cos(theta) * vertical + np.sin(theta) * horizontal_dir
    heading_proj = np.dot(thrust_dir, heading_dir)
    if heading_proj < 0.2:
        thrust_dir = thrust_dir + (0.2 - heading_proj) * heading_dir
    thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)

    return thrust_dir, gamma_cmd, gamma_meas_deg


def compute_guidance_output(r: np.ndarray, v: np.ndarray, t: float, m: float) -> GuidanceOutput:
    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    thrust_dir, gamma_cmd, gamma_meas = compute_desired_thrust_direction(r, v, t)

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

    thrust_on = (m > C.DRY_MASS)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))

    # Max-Q shaping: reduce throttle smoothly when dynamic pressure exceeds limit
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

    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(vertical, thrust_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))
    gamma_angle = np.radians(gamma_cmd)

    return {
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
