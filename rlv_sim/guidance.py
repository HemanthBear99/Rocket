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
    if r_norm < 1e-10:
        return np.array([1.0, 0.0, 0.0])
    return r / r_norm


def compute_local_horizontal(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the local horizontal direction based on AIR-RELATIVE velocity.
    
    This ensures we steer into the wind for zero Angle of Attack (AoA).
    v_rel = v_inertial - (omega_earth x r)
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        
    Returns:
        Unit vector in local horizontal (velocity) direction
    """
    vertical = compute_local_vertical(r)
    
    # Compute Air-Relative Velocity (v_wind)
    omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    v_wind = np.cross(omega_earth, r)
    v_rel = v - v_wind
    
    v_norm = np.linalg.norm(v_rel)
    
    # PDF 8.2.2: Fallback to radial if velocity is zero
    if v_norm < 1e-10:
        return vertical
    
    # Use velocity direction for gravity turn
    # This aligns thrust with velocity for efficient energy gain
    return v_rel / v_norm


def compute_blend_parameter(altitude: float, velocity: float) -> float:
    """
    Compute the blend parameter alpha for altitude-based gravity turn.
    PDF Section 8.7:
    alpha = clip((h - h_switch) / delta_h, 0, 1)
    
    Args:
        altitude: Altitude above Earth surface (m)
        velocity: Velocity magnitude (m/s)
        
    Returns:
        Blend factor (0 to 1)
    """
    # Check minimum velocity condition (PDF Table 6)
    if velocity < C.MIN_VELOCITY_FOR_TURN:
        return 0.0
    
    # Check altitude condition (PDF Section 8.7)
    if altitude < C.GRAVITY_TURN_START_ALTITUDE:
        return 0.0
    
    if altitude > C.GRAVITY_TURN_START_ALTITUDE + C.GRAVITY_TURN_TRANSITION_RANGE:
        return 1.0
    
    # Linear blend with altitude (can use cosine for smoother transition)
    alpha = (altitude - C.GRAVITY_TURN_START_ALTITUDE) / C.GRAVITY_TURN_TRANSITION_RANGE
    return np.clip(alpha, 0.0, 1.0)


def compute_desired_thrust_direction(r: np.ndarray, v: np.ndarray, 
                                     t: float) -> np.ndarray:
    """
    Compute the desired thrust direction based on current guidance phase.
    PDF Section 8.6: t_cmd = (1 - alpha) * r_hat + alpha * v_hat
    
    The guidance law implements:
    1. Vertical ascent (alpha = 0): thrust along local vertical
    2. Gravity turn (0 < alpha < 1): blended transition
    3. Prograde (alpha = 1): thrust along velocity direction
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        t: Current simulation time (s) - not used for altitude-based
        
    Returns:
        Desired thrust direction as unit vector (inertial frame)
    """
    # Compute altitude and velocity magnitude
    altitude = np.linalg.norm(r) - C.R_EARTH
    
    # Use relative velocity magnitude for checks
    omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    v_wind = np.cross(omega_earth, r)
    v_rel = v - v_wind
    velocity = np.linalg.norm(v_rel)
    
    # Get local reference frame
    vertical = compute_local_vertical(r)  # r_hat
    prograde = compute_local_horizontal(r, v)  # v_hat (velocity direction)
    
    # Compute blend parameter (PDF Section 8.7)
    alpha = compute_blend_parameter(altitude, velocity)
    
    # Blended guidance (PDF Section 8.6)
    # t_cmd = (1 - alpha) * r_hat + alpha * v_hat
    thrust_dir = (1.0 - alpha) * vertical + alpha * prograde
    
    # Pitchover Maneuver (Deterministic Azimuth)
    # Apply a small kick towards East (or target azimuth) at low altitude
    # to bias the gravity turn.
    if alpha < 0.01 and C.PITCHOVER_START_ALTITUDE <= altitude <= C.PITCHOVER_END_ALTITUDE:
        # Define target direction (East)
        # In this sim, East is +Y (tangential at launch site [R, 0, 0])
        east = np.array([0.0, 1.0, 0.0])
        
        # Blend into the kick
        # Simple blend: pure vertical -> vertical + kick -> pure vertical
        # We want to TILT the thrust vector.
        # t_cmd = r_hat * cos(theta) + east * sin(theta)
        
        kick_angle = C.PITCHOVER_ANGLE
        thrust_dir = vertical * np.cos(kick_angle) + east * np.sin(kick_angle)
    
    # Normalize (PDF Section 8.6: must normalize the result)
    thrust_norm = np.linalg.norm(thrust_dir)
    if thrust_norm < 1e-10:
        return vertical  # Fallback to vertical
    
    return thrust_dir / thrust_norm


def compute_guidance_output(r: np.ndarray, v: np.ndarray, 
                           t: float, m: float) -> dict:
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
    # Compute state values
    altitude = np.linalg.norm(r) - C.R_EARTH
    velocity = np.linalg.norm(v)
    
    # Compute desired thrust direction
    thrust_dir = compute_desired_thrust_direction(r, v, t)
    
    # Compute blend parameter for logging
    # Recalculate velocity relative for blend alpha
    omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    v_wind = np.cross(omega_earth, r)
    v_rel = v - v_wind
    v_rel_norm = np.linalg.norm(v_rel)
    
    alpha = compute_blend_parameter(altitude, v_rel_norm)
    
    # Determine guidance phase
    if alpha < 0.01:
        if C.PITCHOVER_START_ALTITUDE <= altitude <= C.PITCHOVER_END_ALTITUDE:
            phase = "PITCHOVER"
        else:
            phase = "VERTICAL_ASCENT"
    elif alpha < 0.99:
        phase = "GRAVITY_TURN"
    else:
        phase = "PROGRADE"
    
    # Check if thrust should be on (propellant remaining)
    thrust_on = (m > C.DRY_MASS)
    
    # Compute pitch angle from vertical (for logging)
    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(thrust_dir, vertical), -1.0, 1.0)
    pitch_angle = np.arccos(cos_pitch)
    
    return {
        'thrust_direction': thrust_dir,
        'phase': phase,
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'blend_alpha': alpha,
        'altitude': altitude,
        'velocity': velocity,
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v)
    }
