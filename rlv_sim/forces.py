"""
RLV Phase-I Ascent Simulation - Force Computations

This module implements all force calculations:
- Central gravity
- Thrust (in body frame, transformed to inertial)
- Atmospheric drag
"""

import numpy as np

from . import constants as C
from .frames import quaternion_to_rotation_matrix
from .utils import compute_relative_velocity
from .types import ForceBreakdown


# =============================================================================
# ATMOSPHERE MODEL (US Standard Atmosphere 1976 - Simplified)
# =============================================================================

def compute_atmosphere_properties(altitude: float) -> tuple:
    """
    Compute atmospheric properties (Temperature, Pressure, Density, Speed of Sound).
    Based on US Standard Atmosphere 1976 (Troposphere & Stratosphere).
    
    Args:
        altitude: Geometric altitude above sea level (m)
        
    Returns:
        (temperature, pressure, density, speed_of_sound)
        T in K, P in Pa, rho in kg/m^3, a in m/s
    """
    # Use named constants from constants module
    T0 = C.ATM_T0
    P0 = C.ATM_P0
    L = C.ATM_LAPSE_RATE
    H_TROPO = C.ATM_TROPOPAUSE
    T_STRATO = C.ATM_T_STRATOSPHERE
    G = C.G0
    
    if altitude < 0:
        altitude = 0.0
        
    if altitude <= H_TROPO:
        # Troposphere
        T = T0 - L * altitude
        exponent = G / (L * C.R_GAS)
        P = P0 * (T / T0) ** exponent
    elif altitude < 25000: # Lower Stratosphere (Isothermal approx up to 25km for this sim)
        # Stratosphere starts at 11km
        # P_11 computation
        T11 = T0 - L * H_TROPO # 216.65 K
        P11 = P0 * (T11 / T0) ** (G / (L * C.R_GAS))
        
        T = T11
        # Isothermal equation: P = P11 * exp(-(h - h11) * g / (R * T))
        P = P11 * np.exp(-(G * (altitude - H_TROPO)) / (C.R_GAS * T))
    else:
        # Upper atmosphere blend/decay (simplified)
        # Continue isothermal decay but clamp density to zero eventually
        T = T_STRATO
        # Compute P11 from base constants (avoid magic number)
        T11 = T0 - L * H_TROPO  # Temperature at tropopause
        P11 = P0 * (T11 / T0) ** (G / (L * C.R_GAS))  # Pressure at tropopause (~22632 Pa)
        P = P11 * np.exp(-(G * (altitude - H_TROPO)) / (C.R_GAS * T))
        if altitude > 80000:
            P *= np.exp(-(altitude - 80000)/5000)  # Faster decay
            if altitude > 120000:
                P = 0.0
                
    # Density from Ideal Gas Law: rho = P / (R * T)
    if T > 0 and P > 0:
        rho = P / (C.R_GAS * T)
        # Speed of Sound: a = sqrt(gamma * R * T)
        speed_of_sound = np.sqrt(C.GAMMA * C.R_GAS * T)
    else:
        rho = 0.0
        speed_of_sound = C.ATM_SPEED_OF_SOUND_FALLBACK
        
    return T, P, rho, speed_of_sound



# =============================================================================
# FORCE MODELS
# =============================================================================

def compute_gravity_force(r: np.ndarray, m: float) -> np.ndarray:
    """
    Compute central gravitational force.
    F_grav = -μ * m * r / ||r||³
    """
    r_norm = np.linalg.norm(r)
    if r_norm < C.ZERO_TOLERANCE:
        return np.zeros(3)
    
    return -C.MU_EARTH * m * r / (r_norm ** 3)


def compute_drag_force(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute atmospheric drag force with Mach-dependent Cd.
    
    F_drag = -0.5 * ρ * Cd(Mach) * A * ||v_rel||² * v_rel_hat
    
    Uses Earth rotation for relative velocity (wind).
    """
    altitude = np.linalg.norm(r) - C.R_EARTH
    
    # Get atmospheric properties
    _, _, rho, speed_of_sound = compute_atmosphere_properties(altitude)
    
    if rho < C.DENSITY_FLOOR:
        return np.zeros(3)
        
    # Relative velocity (accounting for Earth rotation and wind)
    # If vehicle inertially stationary, skip wind to avoid spurious drag in tests
    if np.linalg.norm(v) < C.SMALL_VELOCITY_TOL:
        v_rel = np.zeros(3)
    else:
        v_rel = compute_relative_velocity(r, v)
    v_rel_norm = np.linalg.norm(v_rel)
    
    if v_rel_norm < C.SMALL_VELOCITY_TOL:
        return np.zeros(3)
        
    # Mach Number (guard against division by zero at high altitude)
    mach = v_rel_norm / max(speed_of_sound, 1.0)
    
    # Interpolate Cd
    cd = np.interp(mach, C.MACH_BREAKPOINTS, C.CD_VALUES)
    
    # Drag Force
    v_rel_hat = v_rel / v_rel_norm
    drag_magnitude = 0.5 * rho * cd * C.REFERENCE_AREA * v_rel_norm ** 2
    
    return -drag_magnitude * v_rel_hat


def compute_lift_force(r: np.ndarray, v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Compute lift force using small-angle slender-body approximation.
    Lift acts perpendicular to velocity and depends on angle of attack.
    """
    altitude = np.linalg.norm(r) - C.R_EARTH
    _, _, rho, speed_of_sound = compute_atmosphere_properties(altitude)
    if rho < C.DENSITY_FLOOR:
        return np.zeros(3)

    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm < C.SMALL_VELOCITY_TOL:
        return np.zeros(3)

    mach = v_rel_norm / speed_of_sound
    cl_alpha = np.interp(mach, C.MACH_BREAKPOINTS, C.CL_ALPHA_VALUES)

    # Transform v_rel to body frame to get angle of attack
    R = quaternion_to_rotation_matrix(q)
    v_body = R.T @ v_rel
    # AoA approx: angle between body +X (longitudinal) and velocity projection on x-z plane
    alpha = np.arctan2(v_body[2], v_body[0])  # radians
    q_dyn = 0.5 * rho * v_rel_norm**2
    lift_mag = q_dyn * cl_alpha * alpha * C.REFERENCE_AREA

    # Lift direction: perpendicular to v_rel and body-Y axis (assume symmetric rocket)
    # Compute unit normal in inertial frame: u_lift ~ (v_rel × (body_y_inertial)) × v_rel
    body_y_inertial = R[:, 1]
    lift_dir = np.cross(np.cross(v_rel, body_y_inertial), v_rel)
    norm = np.linalg.norm(lift_dir)
    if norm < 1e-9:
        return np.zeros(3)
    lift_dir /= norm
    return lift_mag * lift_dir


def compute_thrust_force(q: np.ndarray, r: np.ndarray, thrust_on: bool = True, throttle: float = 1.0) -> np.ndarray:
    """
    Compute thrust force in inertial frame with altitude compensation and throttling.
    
    Thrust varies with ambient pressure:
    T = T_vac + (T_sl - T_vac) * (P_amb / P_sl)
    
    Args:
        q: Orientation quaternion
        r: Position vector (for altitude/pressure)
        thrust_on: active flag
        throttle: Throttle setting (0.0 to 1.0)
    """
    if not thrust_on:
        return np.zeros(3)
    
    # Clamp throttle
    throttle = max(0.0, min(1.0, throttle))
    
    # Ambient conditions
    altitude = np.linalg.norm(r) - C.R_EARTH
    _, P_amb, _, _ = compute_atmosphere_properties(altitude)
    P0 = C.ATM_P0

    # Thrust varies with ambient pressure: T = T_vac - (T_vac - T_sl) * (P_amb / P_sl)
    # This gives T_sl at sea level, T_vac in vacuum
    thrust_sl = C.THRUST_MAGNITUDE  # Sea level thrust
    thrust_vac = C.MASS_FLOW_RATE * C.ISP_VAC * C.G0  # Vacuum thrust
    
    # Linear interpolation based on ambient pressure ratio
    pressure_ratio = P_amb / P0
    thrust_magnitude = thrust_vac - (thrust_vac - thrust_sl) * pressure_ratio
    
    # Apply throttle
    thrust_magnitude *= float(np.clip(throttle, 0.0, 1.0))
    
    # Thrust in body frame (along +Z axis - vehicle nose direction)
    F_body = np.array([0.0, 0.0, thrust_magnitude])
    
    # Transform to inertial frame
    R = quaternion_to_rotation_matrix(q)
    F_inertial = R @ F_body
    
    return F_inertial


def compute_aerodynamic_moment(r: np.ndarray, v: np.ndarray, q: np.ndarray, cg_pos_z: float) -> np.ndarray:
    """
    Compute aerodynamic moment about the Center of Mass (Body Frame).
    
    Models the aerodynamic instability (CP ahead of CG).
    M = r_arm x F_normal
    
    Args:
        r: Position (inertial)
        v: Velocity (inertial)
        q: Attitude Quaternion
        cg_pos_z: Height of CG from base (m)
        
    Returns:
        Moment vector in BODY FRAME (N*m)
    """
    altitude = np.linalg.norm(r) - C.R_EARTH
    _, _, rho, speed_of_sound = compute_atmosphere_properties(altitude)
    
    if rho < C.DENSITY_FLOOR:
        return np.zeros(3)
    
    # Early exit for stationary vehicle (same pattern as compute_drag_force)
    # This prevents spurious moments when v=0 but Earth rotation creates v_rel
    if np.linalg.norm(v) < C.SMALL_VELOCITY_TOL:
        return np.zeros(3)
        
    # Relative Velocity (Wind)
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = np.linalg.norm(v_rel)
    
    if v_rel_norm < C.SMALL_VELOCITY_TOL:
        return np.zeros(3)
        
    # Transform v_rel to Body Frame to get Angle of Attack components
    R = quaternion_to_rotation_matrix(q)
    # v_body = R.T @ v_rel  <-- R transforms Body to Inertial, so R.T transforms Inertial to Body
    v_body = R.T @ v_rel
    
    # v_body = [vx, vy, vz]
    # In body frame, Z is up (nose). Wind comes from top (-Z) mainly.
    # Transverse components cause AoA.
    vx, vy, vz = v_body
    
    # Dynamic Pressure
    q_dyn = 0.5 * rho * v_rel_norm**2
    
    # Normal Force Calculation (Linearized Aerodynamics)
    # F_normal is proportional to Angle of Attack * q_dyn * Area
    
    # Calculate transverse velocity magnitude
    v_transverse = np.sqrt(vx**2 + vy**2)
    
    if v_transverse < 1e-6:
        return np.zeros(3)
    
    # AoA (alpha) ~ v_transverse / |vz| (small angle approximation)
    # Clamp to reasonable range to prevent instability at extreme attitudes
    alpha = np.arctan2(v_transverse, abs(vz))
    alpha = np.clip(alpha, -np.radians(30), np.radians(30))  # Limit to ±30°
    
    # Normal Force Magnitude
    Fn_mag = q_dyn * C.REFERENCE_AREA * C.C_N_ALPHA * alpha
    
    # Direction of Normal Force in Body Frame:
    # Normal force acts perpendicular to the body axis in the plane of incidence,
    # effectively opposing the transverse component of velocity.
    
    # Unit vector of transverse velocity
    u_trans = np.array([vx, vy, 0.0]) / v_transverse
    
    # Normal Force Vector
    F_normal_body = -Fn_mag * u_trans
    
    # Lever Arm from CG to CP
    arm_z = C.H_CP - cg_pos_z
    r_arm = np.array([0.0, 0.0, arm_z])
    
    # Torque = r x F
    torque_aero = np.cross(r_arm, F_normal_body)
    
    return torque_aero


def compute_total_force(r: np.ndarray, v: np.ndarray, q: np.ndarray, 
                        m: float, thrust_on: bool = True) -> np.ndarray:
    """
    Compute total force acting on the vehicle.
    F_total = F_grav + F_thrust + F_drag
    """
    F_grav = compute_gravity_force(r, m)
    # Note: Updated compute_thrust_force signature to accept 'r'
    F_thrust = compute_thrust_force(q, r, thrust_on)
    F_drag = compute_drag_force(r, v)
    F_coriolis = compute_coriolis_force(v, m)
    
    return F_grav + F_thrust + F_drag + F_coriolis


def compute_coriolis_force(v: np.ndarray, m: float) -> np.ndarray:
    """
    Compute Coriolis force due to Earth's rotation.
    
    F_coriolis = -2m(ω_E × v)
    
    Reference: ROCKET_SIMULATION_RULES.md Section 2.1
    [PHASE I] Critical at high altitude (> 50 km) and high velocity (> 5 km/s)
    Impact: ~1-2% correction to acceleration at high altitude
    
    Args:
        v: Velocity in inertial frame (m/s)
        m: Vehicle mass (kg)
        
    Returns:
        Coriolis force vector (N)
    """
    # Earth's angular velocity vector (rad/s) - pointing along Z axis (North Pole)
    omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    
    # Coriolis acceleration = -2 * ω_E × v
    coriolis_accel = -2.0 * np.cross(omega_earth, v)
    
    # Force = mass × acceleration
    return m * coriolis_accel


def compute_specific_forces(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                            m: float, thrust_on: bool = True) -> ForceBreakdown:
    """
    Compute all forces and return as a dictionary for logging/analysis.
    """
    F_grav = compute_gravity_force(r, m)
    F_thrust = compute_thrust_force(q, r, thrust_on)
    F_drag = compute_drag_force(r, v)
    F_coriolis = compute_coriolis_force(v, m)
    
    return {
        'gravity': F_grav,
        'thrust': F_thrust,
        'drag': F_drag,
        'coriolis': F_coriolis,
        'total': F_grav + F_thrust + F_drag + F_coriolis,
        'gravity_magnitude': np.linalg.norm(F_grav),
        'thrust_magnitude': np.linalg.norm(F_thrust),
        'drag_magnitude': np.linalg.norm(F_drag),
        'coriolis_magnitude': np.linalg.norm(F_coriolis)
    }
