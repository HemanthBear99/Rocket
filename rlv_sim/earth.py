"""
RLV Ascent Simulation - Earth Model (WGS-84)

Provides WGS-84 oblate ellipsoid geometry and ECI/ECEF/geodetic
coordinate conversions. Enabled via config.enable_wgs84 = True.

When disabled, the simulation uses a spherical Earth (R_EARTH).
"""

import numpy as np

from . import constants as C

# WGS-84 constants (defaults)
WGS84_A = 6378137.0                    # Semi-major axis (m)
WGS84_F = 1.0 / 298.257223563          # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)   # Semi-minor axis (m)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2  # First eccentricity squared


def eci_to_ecef(r_eci: np.ndarray, t: float) -> np.ndarray:
    """
    Convert ECI position to ECEF by rotating about Z-axis.

    Args:
        r_eci: Position in ECI frame (m)
        t: Time since epoch (s)

    Returns:
        Position in ECEF frame (m)
    """
    theta = C.EARTH_ROTATION_RATE * t
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_ecef = r_eci[0] * cos_t + r_eci[1] * sin_t
    y_ecef = -r_eci[0] * sin_t + r_eci[1] * cos_t
    z_ecef = r_eci[2]

    return np.array([x_ecef, y_ecef, z_ecef])


def ecef_to_eci(r_ecef: np.ndarray, t: float) -> np.ndarray:
    """Convert ECEF position to ECI."""
    theta = C.EARTH_ROTATION_RATE * t
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_eci = r_ecef[0] * cos_t - r_ecef[1] * sin_t
    y_eci = r_ecef[0] * sin_t + r_ecef[1] * cos_t
    z_eci = r_ecef[2]

    return np.array([x_eci, y_eci, z_eci])


def ecef_to_geodetic(r_ecef: np.ndarray, a: float = WGS84_A,
                     f: float = WGS84_F) -> tuple:
    """
    Convert ECEF position to geodetic coordinates (lat, lon, alt).

    Uses Bowring's iterative method for WGS-84 accuracy.

    Args:
        r_ecef: Position in ECEF frame (m)
        a: Semi-major axis (m)
        f: Flattening

    Returns:
        (latitude_rad, longitude_rad, altitude_m)
    """
    x, y, z = r_ecef
    b = a * (1.0 - f)
    e2 = 2.0 * f - f ** 2
    ep2 = e2 / (1.0 - e2)  # Second eccentricity squared

    p = np.sqrt(x ** 2 + y ** 2)
    lon = np.arctan2(y, x)

    # Bowring initial guess
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3,
        p - e2 * a * np.cos(theta) ** 3
    )

    # Iterate for convergence (typically 2-3 iterations)
    for _ in range(5):
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)

        lat_new = np.arctan2(z + e2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)

    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) - b

    return lat, lon, alt


def geodetic_to_ecef(lat: float, lon: float, alt: float,
                     a: float = WGS84_A, f: float = WGS84_F) -> np.ndarray:
    """
    Convert geodetic coordinates to ECEF position.

    Args:
        lat: Geodetic latitude (rad)
        lon: Geodetic longitude (rad)
        alt: Altitude above ellipsoid (m)
        a: Semi-major axis (m)
        f: Flattening

    Returns:
        ECEF position vector (m)
    """
    e2 = 2.0 * f - f ** 2
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)

    x = (N + alt) * cos_lat * np.cos(lon)
    y = (N + alt) * cos_lat * np.sin(lon)
    z = (N * (1.0 - e2) + alt) * sin_lat

    return np.array([x, y, z])


def wgs84_surface_altitude(r_eci: np.ndarray, t: float,
                           a: float = WGS84_A,
                           f: float = WGS84_F) -> float:
    """
    Compute altitude above the WGS-84 ellipsoid from an ECI position.

    Args:
        r_eci: Position in ECI (m)
        t: Time since epoch (s)
        a, f: WGS-84 parameters

    Returns:
        Altitude above ellipsoid (m)
    """
    r_ecef = eci_to_ecef(r_eci, t)
    _, _, alt = ecef_to_geodetic(r_ecef, a, f)
    return alt
