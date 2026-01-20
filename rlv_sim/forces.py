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

def compute_atmospheric_density(altitude: float) -> float:
    """Legacy wrapper for compatibility."""
    _, _, rho, _ = compute_atmosphere_properties(altitude)
    return rho

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
        
    # Relative velocity (accounting for Earth rotation)
    # In ECI frame, air co-rotates with Earth: v_air = omega × r
    v_rel = compute_relative_velocity(r, v)
    v_rel_norm = np.linalg.norm(v_rel)
    
    if v_rel_norm < C.SMALL_VELOCITY_TOL:
        return np.zeros(3)
        
    # Mach Number
    mach = v_rel_norm / speed_of_sound
    
    # Interpolate Cd
    cd = np.interp(mach, C.MACH_BREAKPOINTS, C.CD_VALUES)
    
    # Drag Force
    v_rel_hat = v_rel / v_rel_norm
    drag_magnitude = 0.5 * rho * cd * C.REFERENCE_AREA * v_rel_norm ** 2
    
    return -drag_magnitude * v_rel_hat


def compute_thrust_force(q: np.ndarray, r: np.ndarray, thrust_on: bool = True) -> np.ndarray:
    """
    Compute thrust force in inertial frame with altitude compensation.
    
    Thrust varies with ambient pressure:
    T = T_vac + (T_sl - T_vac) * (P_amb / P_sl)
    (Linear interpolation assumption)
    
    Args:
        q: Orientation quaternion
        r: Position vector (for altitude/pressure)
        thrust_on: active flag
    """
    if not thrust_on:
        return np.zeros(3)
    
    # Get Ambient Pressure
    altitude = np.linalg.norm(r) - C.R_EARTH
    _, P_amb, _, _ = compute_atmosphere_properties(altitude)
    
    # Sea Level Pressure (use named constant)
    P0 = C.ATM_P0
    
    # Calculate Thrust Magnitude
    # F = mdot * ve + (Pe - Pa) * Ae
    # We use Isp interpolation or Thrust interpolation.
    
    # Calc Vacuum Thrust from constants
    # T_sl = 7.6 MN, Isp_sl = 282
    # mdot = 7.6e6 / (282 * 9.80665)
    # T_vac = mdot * Isp_vac * 9.80665
    
    mdot = C.MASS_FLOW_RATE # Calculated from SL values in constants
    T_vac_calc = mdot * C.ISP_VAC * C.G0
    
    # Interpolate based on pressure (simplified linear model)
    # When P = P0, T = T_sl. When P = 0, T = T_vac.
    # T = T_vac - (P_amb / P0) * (T_vac - T_sl)
    
    thrust_magnitude = T_vac_calc - (P_amb / P0) * (T_vac_calc - C.THRUST_MAGNITUDE)
    
    # Thrust in body frame (along +Z axis)
    F_body = np.array([0.0, 0.0, thrust_magnitude])
    
    # Transform to inertial frame
    R = quaternion_to_rotation_matrix(q)
    F_inertial = R @ F_body
    
    return F_inertial


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
    
    return F_grav + F_thrust + F_drag


def compute_specific_forces(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                            m: float, thrust_on: bool = True) -> ForceBreakdown:
    """
    Compute all forces and return as a dictionary for logging/analysis.
    """
    F_grav = compute_gravity_force(r, m)
    F_thrust = compute_thrust_force(q, r, thrust_on)
    F_drag = compute_drag_force(r, v)
    
    return {
        'gravity': F_grav,
        'thrust': F_thrust,
        'drag': F_drag,
        'total': F_grav + F_thrust + F_drag,
        'gravity_magnitude': np.linalg.norm(F_grav),
        'thrust_magnitude': np.linalg.norm(F_thrust),
        'drag_magnitude': np.linalg.norm(F_drag)
    }
