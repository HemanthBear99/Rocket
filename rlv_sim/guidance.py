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
    """
    Continuous, smooth gamma profile (gravity turn) based on altitude.
    
    Uses a tanh-based shaping function to ensure C1 continuity (smooth derivatives)
    which helps the Attitude Control System track the command without saturating.
    
    Gamma starts at 90 deg (Vertical) and smoothly decays to ~18 deg at MECO.
    """
    # Parameters for the smooth turn
    # Start turn around 1 km, main turn fits the 10-60km region.
    
    # Normalized altitude coordinate (0 to 1) for the main atmospheric ascent
    # We want a smooth transition from 90 -> target_gamma
    
    # Target gamma at 110km is ~18-20 deg
    gamma_start = 90.0
    gamma_final = 18.0 
    
    # Smoothstep-like or sigmoid transition
    # Use a modified tanh shape: y = start + (final - start) * tanh(k * alt)^n
    # Or simply: y = 90 - (90 - final) * f(alt)
    # where f(alt) goes 0->1
    
    # Turn parameters
    turn_start_alt = 1000.0   # m
    # Revert to 45000 to ensure the profile actually commands a turn compatible with physics
    turn_scale = 45000.0      

    if altitude < turn_start_alt:
        return np.radians(gamma_start)
        
    # Normalized altitude for turn curve
    h_norm = (altitude - turn_start_alt) / turn_scale
    
    # Tanh shape: goes 0 -> 1 smoothly
    progress = np.tanh(h_norm)
    
    # Gamma in degrees
    gamma_deg = gamma_start + (gamma_final - gamma_start) * progress
    
    return np.radians(gamma_deg)


def compute_blend_parameter(altitude: float) -> float:
    """
    Blend factor (0 → vertical ascent, 1 → prograde) based on altitude.

    A smooth ramp is used so guidance phase labels and thrust blending
    remain continuous. 
    """
    # Smooth sigmoid blend instead of linear ramp
    start = C.GRAVITY_TURN_START_ALTITUDE
    width = C.GRAVITY_TURN_TRANSITION_RANGE
    
    # Sigmoid function centered at (start + width/2)
    # 0.0 at < start, 1.0 at > start + width
    # Simple clamped linear is often robust enough for blending, but let's smooth it
    
    if altitude <= start:
        return 0.0
    if altitude >= start + width:
        return 1.0
        
    # Hermite interpolation (smoothstep): 3x^2 - 2x^3
    x = (altitude - start) / width
    return x * x * (3.0 - 2.0 * x)


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

    # -------------------------------------------------------------------------
    # 1. State-Dependent Gamma Command (Smooth)
    # -------------------------------------------------------------------------
    gamma_target = gamma_profile_from_altitude(altitude)
    gamma_target_deg = float(np.degrees(gamma_target))

    # -------------------------------------------------------------------------
    # 2. Measurement & PID correction
    # -------------------------------------------------------------------------
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
    gamma_rate = (gamma_meas_deg - _pid_state['prev_gamma_meas']) / max(dt, 1e-3)

    # PID Gains
    kp, ki, kd = 0.85, 0.05, 0.12
    _pid_state['gamma_int'] += error * dt
    _pid_state['gamma_int'] = float(np.clip(_pid_state['gamma_int'], -20.0, 20.0))
    gamma_raw = gamma_target_deg + kp * error + ki * _pid_state['gamma_int'] - kd * gamma_rate
    
    # -------------------------------------------------------------------------
    # 3. Dynamic Pressure & Zero-Lift Logic (Prograde Locking)
    # -------------------------------------------------------------------------
    _, _, rho, _ = compute_atmosphere_properties(altitude)
    q_dyn = 0.5 * rho * v_rel_norm**2
    
    # Calculate Velocity Vector Direction (Prograde)
    if v_rel_norm > 1.0:
        prograde = v_rel / v_rel_norm
    else:
        prograde = vertical

    # Initial "Guidance" Thrust Vector (from Gamma Profile PID)
    gamma_cmd_clamped = float(np.clip(gamma_raw, 10.0, 90.0))
    pitch_cmd_rad = np.radians(90.0 - gamma_cmd_clamped)
    
    if v_horiz > 1.0:
        horiz_axis = v_horiz_vec / v_horiz
    else:
        horiz_axis = east 
        
    guidance_dir = np.cos(pitch_cmd_rad) * vertical + np.sin(pitch_cmd_rad) * horiz_axis
    guidance_dir /= np.linalg.norm(guidance_dir)

    # -------------------------------------------------------------------------
    # 4. Mode Mixing: Blend from Guidance to Prograde based on ALTITUDE
    # -------------------------------------------------------------------------
    # We blend nicely from 1km to 10km to avoid Q-based jumps.
    # 0km - 1km: Pure Guidance (Launch/Vertical)
    # 1km - 10km: Blend in Prograde (Gravity Turn Start)
    # > 10km: Pure Prograde (Gravity Turn Lock)
    
    mix_start_alt = 1000.0
    mix_end_alt = 10000.0
    
    if altitude < mix_start_alt:
        w_prograde = 0.0
    elif altitude > mix_end_alt:
        w_prograde = 1.0
    else:
        w_prograde = (altitude - mix_start_alt) / (mix_end_alt - mix_start_alt)
        
    # Safety Override: If Q is massive (>20kPa), FORCE prograde regardless of altitude
    # to avoid structural break.
    if q_dyn > 20000.0:
         w_prograde = 1.0
        
    # Apply Blend
    # thrust_dir = (1 - w) * guidance + w * prograde
    thrust_dir_mixed = (1.0 - w_prograde) * guidance_dir + w_prograde * prograde
    
    # Re-normalize
    norm_mixed = np.linalg.norm(thrust_dir_mixed)
    if norm_mixed > 1e-6:
        thrust_dir_mixed /= norm_mixed
    else:
        thrust_dir_mixed = guidance_dir
        
    thrust_dir = thrust_dir_mixed
    
    # Update gamma_cmd simply for logging (what are we actually commanding?)
    # We back-calculate the effective command from our final vector
    cos_p = np.dot(thrust_dir, vertical)
    gamma_cmd_clamped = np.degrees(np.arcsin(np.clip(cos_p, -1.0, 1.0)))

    _pid_state['prev_gamma_meas'] = gamma_meas_deg

    return thrust_dir, gamma_cmd_clamped, gamma_meas_deg


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


def compute_coast_guidance(r: np.ndarray, v: np.ndarray, t: float, m: float) -> GuidanceOutput:
    """
    Guidance logic for Phase II (Coast).
    
    - Thrust: OFF
    - Attitude: Prograde (aligned with relative velocity)
    """
    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    
    # 1. Orientation: Align with Velocity Vector (Prograde)
    if v_rel_norm > 1.0:
        desired_dir = v_rel / v_rel_norm
    else:
        desired_dir = compute_local_vertical(r)
        
    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(vertical, desired_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))
    
    # Gamma calculation for logging
    v_vert = float(np.dot(v_rel, vertical))
    v_horiz_vec = v_rel - v_vert * vertical
    v_horiz = float(np.linalg.norm(v_horiz_vec))
    gamma_actual = float(np.degrees(np.arctan2(v_vert, v_horiz)))

    return {
        'thrust_direction': desired_dir,
        'phase': "COAST_PHASE",
        'thrust_on': False,
        'pitch_angle': pitch_angle,
        'gamma_angle': np.radians(gamma_actual),
        'gamma_command_deg': gamma_actual, # We "command" what we have since we follow velocity
        'gamma_measured_deg': gamma_actual,
        'velocity_tilt_deg': gamma_actual, 
        'blend_alpha': 1.0, # Pure prograde
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': desired_dir,
        'throttle': 0.0,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm
    }


def compute_booster_guidance(r: np.ndarray, v: np.ndarray, t: float, m: float, phase: str) -> GuidanceOutput:
    """
    Guidance logic for Booster Recovery (Phase III-B).
    
    Phases:
    - FLIP: Reorient 180 deg to point engines retrograde.
    - BOOSTBACK: Fire engines to cancel downrange velocity.
    - COAST: Ballistic return.
    - ENTRY: High-alpha for drag.
    - LANDING: Vertical descent.
    """
    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    
    # Defaults
    thrust_on = False
    throttle = 0.0
    desired_dir = compute_local_vertical(r) # Default Up
    
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    
    prograde = v_rel / v_rel_norm if v_rel_norm > 1.0 else compute_local_vertical(r)
    retrograde = -prograde
    
    # --- PHASE LOGIC ---
    if phase == "BOOSTER_FLIP":
        # Target Retrograde
        desired_dir = retrograde
        thrust_on = False # Coasting flip
        
    elif phase == "BOOSTER_BOOSTBACK":
        # Target Retrograde + Throttle Up
        desired_dir = retrograde
        thrust_on = True
        throttle = 1.0 # Max power for boostback
        
    elif phase == "BOOSTER_COAST":
        # Engines First (Retrograde) for entry prep
        desired_dir = retrograde
        thrust_on = False
        
    elif phase == "BOOSTER_ENTRY":
        # Engines First (Retrograde)
        desired_dir = retrograde
        thrust_on = False
        
    elif phase == "BOOSTER_LANDING":
        # Retrograde (Vertical at this point) + Landing Burn
        desired_dir = retrograde
        thrust_on = True # Landing burn
        # Simple suicide burn logic placeholder:
        # If impact time < burn time, throttle = 1.0
        # For now, just simplistic braking
        if altitude < 5000:
             throttle = 1.0
        else:
             throttle = 0.0
             thrust_on = False

    # Common Outputs
    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(vertical, desired_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))
    
    return {
        'thrust_direction': desired_dir,
        'phase': phase,
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'gamma_angle': 0.0, # Placeholder
        'gamma_command_deg': 0.0,
        'gamma_measured_deg': 0.0,
        'velocity_tilt_deg': 0.0,
        'blend_alpha': 0.0,
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'prograde': prograde,
        'throttle': throttle,
        'v_rel': v_rel,
        'v_rel_mag': v_rel_norm
    }

