"""
RLV Phase-I Ascent Simulation - Utility Functions

This module contains shared utility functions used across multiple modules
to eliminate code duplication (DRY principle).
"""

import numpy as np

from . import constants as C


def compute_relative_velocity(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute air-relative velocity accounting for Earth rotation.
    
    The atmosphere co-rotates with Earth, so the air velocity at position r
    is v_wind = omega_earth × r. The vehicle's velocity relative to the air
    is v_rel = v - v_wind.
    
    Args:
        r: Position vector in inertial (ECI) frame (m)
        v: Velocity vector in inertial (ECI) frame (m/s)
        
    Returns:
        Air-relative velocity vector (m/s)
        
    Note:
        - Earth rotation axis is assumed to be the Z-axis (standard ECI frame)
        - For equatorial launches, this accounts for the ~465 m/s rotational velocity
    """
    omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    v_wind = np.cross(omega_earth, r)
    return v - v_wind


def compute_relative_velocity_magnitude(r: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the magnitude of air-relative velocity.
    
    Args:
        r: Position vector in inertial (ECI) frame (m)
        v: Velocity vector in inertial (ECI) frame (m/s)
        
    Returns:
        Air-relative speed (m/s)
    """
    v_rel = compute_relative_velocity(r, v)
    return np.linalg.norm(v_rel)
