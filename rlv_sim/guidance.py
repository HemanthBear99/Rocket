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


def compute_blend_parameter(altitude: float) -> float:
    """
    Compute blend parameter alpha for altitude-based gravity turn.
    PDF Section 8.7:
    alpha = clip((h - h_switch) / delta_h, 0, 1)
    
    Args:
        altitude: Altitude above Earth surface (m)
        
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
    
    The pitchover tilts the thrust toward the target azimuth (East).
    This initiates the gravity turn by providing a small perturbation from vertical.
    
    Args:
        r: Position vector in inertial frame (m)
        vertical: Local vertical direction (r_hat)
        
    Returns:
        Unit vector in the pitchover direction (inertial frame)
    """
    # Compute East direction at current position
    # East = normalize(Z_earth × r) for any launch site
    z_earth = np.array([0.0, 0.0, 1.0])  # Earth rotation axis
    east = np.cross(z_earth, r)
    east_norm = np.linalg.norm(east)
    
    if east_norm < 1e-10:
        # At poles, East is undefined - use +Y as fallback
        east = np.array([0.0, 1.0, 0.0])
    else:
        east = east / east_norm
    
    # Compute the pitchover direction by tilting vertical toward east
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
    else:
        # Use smooth ramp-in for pitchover
        if altitude < C.PITCHOVER_START_ALTITUDE + C.PITCHOVER_RAMP_DISTANCE:
            # Ramp up from 0 to 1 over ramp distance
            return (altitude - C.PITCHOVER_START_ALTITUDE) / C.PITCHOVER_RAMP_DISTANCE
        else:
            # Hold at 1.0 - Let Gravity Turn blend take over naturally
            return 1.0


def compute_gravity_turn_angle(altitude: float, velocity: float) -> float:
    """
    Compute flight path angle (gamma) for gravity turn.
    
    Gravity turn: γ decreases from 90° to ~10° as the vehicle accelerates.
    State-dependent: Uses both altitude and velocity for smooth transition.
    
    Reference: ROCKET_SIMULATION_RULES.md Section 6.2
    [PHASE I] Proper gravity turn: gamma_90 at launch, gamma_10 at high velocity
    Impact: Vehicle gains required horizontal velocity for Phase 2
    
    Args:
        altitude: Current altitude (m)
        velocity: Current speed (m/s)
        
    Returns:
        Flight path angle in radians (90° = vertical, 0° = horizontal)
    """
    # Velocity-based transition is more physically accurate than pure altitude
    # Minimum gamma (final): ~10 degrees (0.1745 rad) - nearly orbital
    # Maximum gamma (initial): ~89 degrees (1.553 rad) - nearly vertical
    
    gamma_max = np.radians(89.0)  # Nearly vertical at start
    gamma_min = np.radians(10.0)   # Nearly horizontal at end (~orbital insertion)
    
    # Velocity thresholds for transition
    v_start = 100.0    # m/s - Begin gravity turn transition
    v_full_turn = 5000.0  # m/s - Complete gravity turn (orbital-like speed)
    
    # Normalized velocity parameter (0 to 1)
    vel_param = np.clip((velocity - v_start) / (v_full_turn - v_start), 0.0, 1.0)
    
    # Smooth interpolation: gamma decreases from 90° to 10°
    # Use power law for smooth, accelerating descent rate
    # This makes pitch angle rise more aggressively at high speed
    gamma_angle = gamma_max - (gamma_max - gamma_min) * (vel_param ** 1.5)
    
    return float(gamma_angle)


def compute_desired_thrust_direction(r: np.ndarray, v: np.ndarray, 
                                     t: float) -> np.ndarray:
    """
    Compute desired thrust direction based on proper GRAVITY TURN guidance.
    
    This implements a state-dependent guidance law:
    - γ (flight path angle) decreases from 90° → 10° with velocity
    - Pitch angle increases as vehicle turns (0° → 60°+)
    - Uses both altitude and velocity for smooth, continuous transitions
    - Vehicle gains horizontal velocity required for Phase 2
    
    Reference: ROCKET_SIMULATION_RULES.md Section 6.2
    [PHASE I] Proper gravity turn with velocity-dependent blending
    Impact: Builds required horizontal velocity (5+ km/s) for orbital insertion
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        t: Current simulation time (s)
        
    Returns:
        Desired thrust direction as unit vector (inertial frame)
    """
    # Compute state variables
    altitude = float(np.linalg.norm(r) - C.R_EARTH)
    velocity = float(np.linalg.norm(v))
    
    # Get local reference frame
    vertical = compute_local_vertical(r)  # r_hat (radially outward)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = float(np.linalg.norm(v_rel))
    
    if v_rel_norm < C.ZERO_TOLERANCE:
        return vertical
        
    prograde = v_rel / v_rel_norm  # v_hat (velocity direction)
    
    # Compute flight path angle (gamma)
    # γ = 90° (vertical) initially, decreases to ~10° as velocity increases
    # This naturally creates the gravity turn
    gamma = compute_gravity_turn_angle(altitude, velocity)
    
    # Phase 1: Vertical ascent (altitude < 1000m, velocity < 200 m/s)
    # Stay mostly vertical with optional small pitchover kick
    if altitude < (C.PITCHOVER_START_ALTITUDE + C.PITCHOVER_RAMP_DISTANCE) or velocity < 200.0:
        # Pure vertical with optional small eastward pitchover
        pitchover_beta = compute_pitchover_blend(altitude)
        if pitchover_beta > 0.0:
            pitchover_dir = compute_pitchover_direction(r, vertical)
            thrust_dir = (1.0 - pitchover_beta) * vertical + pitchover_beta * pitchover_dir
        else:
            thrust_dir = vertical
    else:
        # Phase 2: Gravity turn - Use γ to control descent rate
        # The flight path angle γ naturally decreases with velocity
        # This causes the vehicle to gradually pitch toward prograde
        # while still gaining altitude (positive γ)
        
        # Mathematical formulation:
        # thrust_dir = cos(γ) * vertical + sin(γ) * prograde
        # When γ = 90°: pure vertical (no horizontal component)
        # When γ = 45°: balanced vertical and horizontal (pitch ~45°)
        # When γ = 10°: mostly horizontal (orbital insertion)
        
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        
        # Desired direction maintains flight path angle γ
        # This is the natural gravity turn trajectory
        thrust_dir = cos_gamma * vertical + sin_gamma * prograde
    
    # Normalize to unit vector
    thrust_norm = np.linalg.norm(thrust_dir)
    if thrust_norm < 1e-10:
        return vertical  # Safety fallback
    
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
    if alpha < 0.01:
        if C.PITCHOVER_START_ALTITUDE <= altitude <= C.PITCHOVER_END_ALTITUDE:
            phase = "PITCHOVER"
        else:
            phase = "VERTICAL_ASCENT"
    elif alpha < 0.99:
        phase = "GRAVITY_TURN"
    else:
        phase = "PROGRADE"
    
    # SAFETY: Ensure alpha is non-negative
    alpha = np.maximum(0.0, alpha)

    # Check if thrust should be on
    thrust_on = (m > C.DRY_MASS)
    
    # =========================================================================
    # MAX-Q THROTTLING LOGIC
    # =========================================================================
    # Calculate Dynamic Pressure using atmospheric properties
    from .forces import compute_atmosphere_properties
    _, _, rho, _ = compute_atmosphere_properties(altitude)
    
    q_dyn = 0.5 * rho * v_rel_norm**2
    
    throttle = 1.0
    if q_dyn > C.MAX_DYNAMIC_PRESSURE:
        # Simple throttle bucket: reduced thrust to limit Q
        throttle_target = C.MAX_DYNAMIC_PRESSURE / q_dyn
        
        # SAFETY: Ensure Thrust > Weight to prevent stall/crash
        # W = m * g
        gravity_force = m * C.G0
        
        # Max Thrust at this altitude
        # Approx T_max = C.THRUST_MAGNITUDE (for safety check, assume sea level conservative)
        # Or calculate linearly. 
        # Using T_sl is safe (T_vac is higher, so throttle % would be lower for same force).
        # We need min_force = 1.05 * Weight.
        # min_throttle = min_force / T_max_avail.
        
        # Let's use the nominal max thrust for simplification 
        # (guidance loop doesn't compute exact engine pressure)
        min_thrust_req = gravity_force * 1.05 # 5% margin
        min_throttle = min_thrust_req / C.THRUST_MAGNITUDE
        
        # Clamp throttle
        throttle = max(throttle_target, min_throttle)
        
        # Ultimate clamp
        throttle = max(0.4, min(1.0, throttle))
    else:
        throttle = 1.0
    
    # Return Guidance Output
    
    # Compute pitch angle from vertical (for logging)
    vertical = compute_local_vertical(r)
    cos_pitch = np.clip(np.dot(vertical, thrust_dir), -1.0, 1.0)
    pitch_angle = float(np.arccos(cos_pitch))
    
    # Compute flight path angle for logging
    gamma_angle = compute_gravity_turn_angle(altitude, float(np.linalg.norm(v)))
    
    return {
        'thrust_direction': thrust_dir,
        'phase': phase,
        'thrust_on': thrust_on,
        'pitch_angle': pitch_angle,
        'gamma_angle': gamma_angle,  # [NEW] Flight path angle (90° → 10°)
        'blend_alpha': alpha,
        'altitude': altitude,
        'velocity': float(np.linalg.norm(v)),
        'local_vertical': vertical,
        'local_horizontal': compute_local_horizontal(r, v),
        'throttle': throttle
    }

