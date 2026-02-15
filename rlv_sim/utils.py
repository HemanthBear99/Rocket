"""
RLV Phase-I Ascent Simulation - Utility Functions

This module contains shared utility functions used across multiple modules
to eliminate code duplication (DRY principle).
"""

import numpy as np

from . import constants as C


def _wind_vector(r: np.ndarray) -> np.ndarray:
    """Simple altitude-dependent wind in inertial frame (East/West)."""
    alt = np.linalg.norm(r) - C.R_EARTH
    if alt <= 0.0:
        return np.zeros(3)
    # power-law profile with smooth onset ramp from 0 to 5 km
    # to avoid a discontinuity in air-relative velocity at 5 km
    speed = C.WIND_REF_SPEED * (alt / C.WIND_REF_ALT) ** C.WIND_EXPONENT
    if alt < 5000.0:
        # Smooth ramp: Hermite (smoothstep) from 0 at ground to 1 at 5 km
        x = alt / 5000.0
        speed *= x * x * (3.0 - 2.0 * x)
    # Direction: azimuth from north clockwise; convert to ECI assuming launch site on equator x-axis
    # East unit at site ~ +Y, North ~ +Z, Up ~ +X; but we approximate global: use ECI basis (X radial, Y east, Z north)
    east = np.array([0.0, 1.0, 0.0])
    north = np.array([0.0, 0.0, 1.0])
    dir_vec = np.cos(C.WIND_DIRECTION_AZIMUTH) * north + np.sin(C.WIND_DIRECTION_AZIMUTH) * east
    return speed * dir_vec


def compute_relative_velocity(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute air-relative velocity removing Earth rotation and winds.
    v_rel = v_inertial - (omega_earth × r) - v_wind
    """
    omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    wind = _wind_vector(r)
    return v - np.cross(omega_earth, r) - wind


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
