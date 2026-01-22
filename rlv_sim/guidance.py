"""
RLV Phase-I Ascent Simulation - Guidance Law

This module implements the deterministic ascent guidance:
- Vertical ascent phase (altitude < 1,500m)
- Gravity turn (altitude-based blending per PDF Section 8)
- Smooth blending between phases

Output: desired thrust direction (inertial frame)
"""

import numpy as np

from . import constants as C
from .types import GuidanceOutput
from .utils import compute_relative_velocity


def compute_local_vertical(r: np.ndarray) -> np.ndarray:
    """
    Compute the local vertical direction (radial outward).
    PDF Section 8.2.1: r_hat = r / ||r||

    Args:
        r: Position vector in inertial frame (m)

    Returns:
        Unit vector pointing radially outward
    """
    r_norm = np.linalg.norm(r)
    if r_norm < C.ZERO_TOLERANCE:
        return np.array([1.0, 0.0, 0.0])
    return r / r_norm


def compute_local_horizontal(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute local horizontal direction based on AIR-RELATIVE velocity.
    
    Using air-relative velocity (v_rel = v_inertial - v_wind) ensures
    guidance commands align with vehicle's trajectory through the atmosphere,
    not affected by Earth's rotation (tangential wind).
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        
    Returns:
        Unit vector in local horizontal (velocity) direction
    """
    vertical = compute_local_vertical(r)
    
    # Use AIR-RELATIVE velocity for guidance
    # v_rel = v_inertial - (omega_earth × r)
    # This ensures guidance works in the rotating frame of the atmosphere
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    
    # PDF 8.2.2: Fallback to radial if velocity is zero
    if v_rel_norm < C.ZERO_TOLERANCE:
        return vertical
    
    # Use velocity direction for gravity turn
    # This aligns thrust with velocity for efficient energy gain
    return v_rel / v_rel_norm


def compute_blend_parameter(altitude: float, velocity: float = 0.0) -> float:
    """
    Compute blend parameter alpha for altitude-based gravity turn.
    PDF Section 8.7:
    alpha = clip((h - h_switch) / delta_h, 0, 1)
    
    Args:
        altitude: Altitude above Earth surface (m)
        velocity: Velocity magnitude (m/s) - kept for compatibility, not used
        
    Returns:
        Blend factor (0 to 1)
    """
    # Check altitude condition (PDF Section 8.7)
    if altitude < C.GRAVITY_TURN_START_ALTITUDE:
        return 0.0
    
    if altitude > C.GRAVITY_TURN_START_ALTITUDE + C.GRAVITY_TURN_TRANSITION_RANGE:
        return 1.0
    
    # Linear blend with altitude (can use cosine for smoother transition)
    # This is STATE-TRIGGERED (altitude-based), not time-triggered
    alpha = (altitude - C.GRAVITY_TURN_START_ALTITUDE) / C.GRAVITY_TURN_TRANSITION_RANGE
    return float(np.clip(alpha, 0.0, 1.0))


def compute_pitchover_direction(r: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    """
    Compute the pitchover kick direction in inertial frame.
    
    The pitchover tilts the thrust toward the target azimuth (East = +Y in our frame).
    This initiates the gravity turn by providing a small perturbation from vertical.
    
    Args:
        r: Position vector in inertial frame (m)
        vertical: Local vertical direction (r_hat)
        
    Returns:
        Unit vector in the pitchover direction (inertial frame)
    """
    # Pitchover azimuth is East (90°), which corresponds to +Y in ECI frame
    # when launch site is at [R_earth, 0, 0]
    # 
    # More generally, the "East" direction at position r is:
    # east = normalize(cross([0,0,1], r)) for equatorial launch
    # But for simplicity at equator launch along +X, East = +Y
    east = np.array([0.0, 1.0, 0.0])
    
    # Compute the pitchover direction by tilting vertical toward east
    # Using Rodrigues' rotation: rotate vertical by PITCHOVER_ANGLE about axis perpendicular to both
    # Simpler approach: linear blend then normalize
    # pitchover_dir = cos(theta) * vertical + sin(theta) * east
    cos_theta = np.cos(C.PITCHOVER_ANGLE)
    sin_theta = np.sin(C.PITCHOVER_ANGLE)
    
    pitchover_dir = cos_theta * vertical + sin_theta * east
    return pitchover_dir / np.linalg.norm(pitchover_dir)


def compute_pitchover_blend(altitude: float) -> float:
    """
    Compute blend factor for pitchover maneuver.
    
    Returns 0 before pitchover, quickly ramps to 1, holds, then ramps down.
    This ensures the full pitchover angle is applied through most of the phase.
    
    Args:
        altitude: Current altitude (m)
        
    Returns:
        Pitchover blend factor (0 to 1)
    """
    if altitude < C.PITCHOVER_START_ALTITUDE:
        return 0.0
    elif altitude > C.PITCHOVER_END_ALTITUDE:
        return 0.0
    else:
        # Use a short ramp-in and ramp-out for smoothness
        # Ramp zones: first 50m and last 50m of the phase
        ramp_distance = 50.0  # meters
        
        if altitude < C.PITCHOVER_START_ALTITUDE + ramp_distance:
            # Ramp up from 0 to 1 in first 50m
            return (altitude - C.PITCHOVER_START_ALTITUDE) / ramp_distance
        elif altitude > C.PITCHOVER_END_ALTITUDE - ramp_distance:
            # Ramp down from 1 to 0 in last 50m
            return (C.PITCHOVER_END_ALTITUDE - altitude) / ramp_distance
        else:
            # Hold at 1.0 for most of the phase
            return 1.0


def compute_desired_thrust_direction(r: np.ndarray, v: np.ndarray, 
                                     t: float) -> np.ndarray:
    """
    Compute desired thrust direction based on current guidance phase.
    
    The guidance law implements:
    1. Vertical ascent (alt < 100m): thrust along local vertical
    2. Pitchover (100m < alt < 1000m): small eastward kick to initiate turn
    3. Gravity turn (alt > 1500m): blend toward prograde
    4. Prograde (alt > 5500m): thrust along velocity direction
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        t: Current simulation time (s) - not used for altitude-based
        
    Returns:
        Desired thrust direction as unit vector (inertial frame)
    """
    # Compute altitude
    altitude = float(np.linalg.norm(r) - C.R_EARTH)

    # Get local reference frame
    vertical = compute_local_vertical(r)  # r_hat
    prograde = compute_local_horizontal(r, v)  # v_hat (velocity direction)

    # Phase 1: Pitchover maneuver (100m - 1000m)
    # Apply a small eastward kick to initiate the gravity turn
    pitchover_beta = compute_pitchover_blend(altitude)
    
    if pitchover_beta > 0.0:
        # Compute pitchover direction (vertical tilted toward east)
        pitchover_dir = compute_pitchover_direction(r, vertical)
        # Blend between pure vertical and pitchover direction
        base_dir = (1.0 - pitchover_beta) * vertical + pitchover_beta * pitchover_dir
    else:
        base_dir = vertical
    
    # Normalize the base direction
    base_dir = base_dir / np.linalg.norm(base_dir)

    # Phase 2: Gravity turn blend (1500m - 5500m)
    # Smoothly transition from vertical/pitchover to prograde
    alpha = compute_blend_parameter(altitude)

    # Final thrust direction: blend base_dir with prograde
    # t_cmd = (1 - alpha) * base_dir + alpha * v_hat
    thrust_dir = (1.0 - alpha) * base_dir + alpha * prograde

    # Normalize (must normalize the result)
    thrust_norm = np.linalg.norm(thrust_dir)
    if thrust_norm < 1e-10:
        return vertical  # Fallback to vertical

    return thrust_dir / thrust_norm


def compute_guidance_output(r: np.ndarray, v: np.ndarray, 
                           t: float, m: float) -> GuidanceOutput:
    """
    Compute full guidance output for logging and control.
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        t: Current simulation time (s)
        m: Current mass (kg)
        
    Returns:
        Dictionary containing guidance state and commands
    """
    # Compute altitude
    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    
    # Compute desired thrust direction
    thrust_dir = compute_desired_thrust_direction(r, v, t)
    
    # Compute blend parameter for logging
    # Use relative velocity for blend alpha
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    
    # Compute alpha based on current altitude (purely altitude-based)
    # Note: velocity parameter removed - alpha only depends on altitude
    alpha = compute_blend_parameter(altitude)
    
    # Determine guidance phase
    # Note: alpha based on altitude blend ensures smooth transitions
    if alpha < 0.01:
        # Before gravity turn starts (alpha < 0.01)
        # Pitchover phase: altitude in [C.PITCHOVER_START, C.PITCHOVER_END] range
        # Vertical ascent phase: altitude < C.PITCHOVER_START
        if C.PITCHOVER_START_ALTITUDE <= altitude <= C.PITCHOVER_END_ALTITUDE:
            phase = "PITCHOVER"
        else:
            phase = "VERTICAL_ASCENT"
    elif alpha < 0.99:
        phase = "GRAVITY_TURN"
    else:
        phase = "PROGRADE"
    
    # SAFETY: Ensure alpha is non-negative to prevent numerical issues
    # When altitude is near zero, blend calculation could have edge cases
    alpha = np.maximum(0.0, alpha)  # Ensure alpha >= 0
    
    # Check if thrust should be on (propellant remaining)
    thrust_on = (m > C.DRY_MASS)
    
    # Compute pitch angle from vertical (for logging)
    # Pitch is angle FROM vertical TO thrust_dir
    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(vertical, thrust_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))
    
    return {
        'thrust_direction': thrust_dir,
        'phase': phase,
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'blend_alpha': alpha,
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v)
    }
