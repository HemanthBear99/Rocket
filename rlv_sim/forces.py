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


def compute_gravity_force(r: np.ndarray, m: float) -> np.ndarray:
    """
    Compute central gravitational force.
    
    F_grav = -μ * m * r / ||r||³
    
    Args:
        r: Position vector in inertial frame (m)
        m: Vehicle mass (kg)
        
    Returns:
        Gravitational force vector in inertial frame (N)
    """
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-10:
        return np.zeros(3)
    
    return -C.MU_EARTH * m * r / (r_norm ** 3)


def compute_atmospheric_density(altitude: float) -> float:
    """
    Compute atmospheric density using exponential model.
    
    ρ(h) = ρ₀ * exp(-h / H)
    
    Args:
        altitude: Altitude above Earth's surface (m)
        
    Returns:
        Atmospheric density (kg/m³)
    """
    if altitude < 0:
        return C.RHO_0
    if altitude > 100000:  # Above 100km, negligible atmosphere
        return 0.0
    
    return C.RHO_0 * np.exp(-altitude / C.H_SCALE)


def compute_drag_force(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute atmospheric drag force.
    
    F_drag = -0.5 * ρ * Cd * A * ||v||² * v̂
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        
    Returns:
        Drag force vector in inertial frame (N)
    """
    altitude = np.linalg.norm(r) - C.R_EARTH
    rho = compute_atmospheric_density(altitude)
    
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10 or rho < 1e-15:
        return np.zeros(3)
    
    # Drag opposes velocity
    v_hat = v / v_norm
    drag_magnitude = 0.5 * rho * C.DRAG_COEFFICIENT * C.REFERENCE_AREA * v_norm ** 2
    
    return -drag_magnitude * v_hat


def compute_thrust_force(q: np.ndarray, thrust_on: bool = True) -> np.ndarray:
    """
    Compute thrust force in inertial frame.
    
    Thrust is defined in body frame along +Z axis.
    F_thrust_inertial = R(q) @ F_thrust_body
    
    Args:
        q: Orientation quaternion [w, x, y, z]
        thrust_on: Whether thrust is active
        
    Returns:
        Thrust force vector in inertial frame (N)
    """
    if not thrust_on:
        return np.zeros(3)
    
    # Thrust in body frame (along +Z axis)
    F_body = np.array([0.0, 0.0, C.THRUST_MAGNITUDE])
    
    # Transform to inertial frame
    R = quaternion_to_rotation_matrix(q)
    F_inertial = R @ F_body
    
    return F_inertial


def compute_total_force(r: np.ndarray, v: np.ndarray, q: np.ndarray, 
                        m: float, thrust_on: bool = True) -> np.ndarray:
    """
    Compute total force acting on the vehicle.
    
    F_total = F_grav + F_thrust + F_drag
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        q: Orientation quaternion [w, x, y, z]
        m: Vehicle mass (kg)
        thrust_on: Whether thrust is active
        
    Returns:
        Total force vector in inertial frame (N)
    """
    F_grav = compute_gravity_force(r, m)
    F_thrust = compute_thrust_force(q, thrust_on)
    F_drag = compute_drag_force(r, v)
    
    return F_grav + F_thrust + F_drag


def compute_specific_forces(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                            m: float, thrust_on: bool = True) -> dict:
    """
    Compute all forces and return as a dictionary for logging/analysis.
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        q: Orientation quaternion [w, x, y, z]
        m: Vehicle mass (kg)
        thrust_on: Whether thrust is active
        
    Returns:
        Dictionary containing all force components
    """
    F_grav = compute_gravity_force(r, m)
    F_thrust = compute_thrust_force(q, thrust_on)
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
