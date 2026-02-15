"""
RLV Ascent Simulation - Abort Mode Logic

Monitors vehicle state for abort conditions and determines
appropriate abort mode based on flight regime.

Enabled via config.enable_abort_modes = True.
"""

import numpy as np

from . import constants as C
from .forces import compute_atmosphere_properties
from .utils import compute_relative_velocity


class AbortMonitor:
    """Monitors for abort conditions during flight."""

    def __init__(self, q_alpha_threshold: float = 60000.0,
                 attitude_threshold_deg: float = 30.0,
                 max_tumble_rate: float = 1.0):
        """
        Args:
            q_alpha_threshold: Q-alpha limit (Pa*rad) before abort
            attitude_threshold_deg: Max attitude error before abort (deg)
            max_tumble_rate: Max angular rate before abort (rad/s)
        """
        self.q_alpha_threshold = q_alpha_threshold
        self.attitude_threshold = np.radians(attitude_threshold_deg)
        self.max_tumble_rate = max_tumble_rate
        self.abort_triggered = False
        self.abort_reason = None
        self.abort_mode = None
        self.abort_time = None

    def check(self, r: np.ndarray, v: np.ndarray, q: np.ndarray,
              omega: np.ndarray, t: float,
              attitude_error_rad: float = 0.0) -> dict:
        """
        Check for abort conditions.

        Args:
            r: Position (ECI, m)
            v: Velocity (ECI, m/s)
            q: Quaternion [w,x,y,z]
            omega: Angular velocity (body, rad/s)
            t: Current time (s)
            attitude_error_rad: Current attitude error magnitude (rad)

        Returns:
            dict with 'abort' (bool), 'mode' (str or None), 'reason' (str or None),
            'q_alpha' (float), 'omega_mag' (float)
        """
        if self.abort_triggered:
            return {
                'abort': True,
                'mode': self.abort_mode,
                'reason': self.abort_reason,
                'q_alpha': 0.0,
                'omega_mag': np.linalg.norm(omega),
            }

        altitude = float(np.linalg.norm(r) - C.R_EARTH)
        v_rel = compute_relative_velocity(r, v)
        v_rel_mag = float(np.linalg.norm(v_rel))
        _, _, rho, _ = compute_atmosphere_properties(altitude)

        # Q-alpha check
        # Compute AoA from body +Z vs velocity
        from .frames import rotate_vector_by_quaternion
        body_z = rotate_vector_by_quaternion(C.BODY_Z_AXIS, q)
        if v_rel_mag > 10.0:
            cos_aoa = float(np.clip(np.dot(body_z, v_rel / v_rel_mag), -1.0, 1.0))
            aoa = float(np.arccos(cos_aoa))
        else:
            aoa = 0.0

        q_dyn = 0.5 * rho * v_rel_mag ** 2
        q_alpha = q_dyn * aoa

        # Tumble rate
        omega_mag = float(np.linalg.norm(omega))

        abort = False
        reason = None
        mode = None

        # Q-alpha exceedance
        if q_alpha > self.q_alpha_threshold and altitude < 80000.0:
            abort = True
            reason = f"Q-alpha exceedance: {q_alpha:.0f} > {self.q_alpha_threshold:.0f} Pa*rad"
            mode = self._determine_mode(altitude, v_rel_mag, t)

        # Attitude divergence
        elif attitude_error_rad > self.attitude_threshold:
            abort = True
            reason = f"Attitude error: {np.degrees(attitude_error_rad):.1f} > {np.degrees(self.attitude_threshold):.1f} deg"
            mode = self._determine_mode(altitude, v_rel_mag, t)

        # Tumble
        elif omega_mag > self.max_tumble_rate:
            abort = True
            reason = f"Tumble rate: {np.degrees(omega_mag):.1f} deg/s > {np.degrees(self.max_tumble_rate):.1f} deg/s"
            mode = self._determine_mode(altitude, v_rel_mag, t)

        if abort:
            self.abort_triggered = True
            self.abort_reason = reason
            self.abort_mode = mode
            self.abort_time = t

        return {
            'abort': abort,
            'mode': mode,
            'reason': reason,
            'q_alpha': q_alpha,
            'omega_mag': omega_mag,
        }

    def _determine_mode(self, altitude: float, velocity: float, t: float) -> str:
        """Determine abort mode based on flight regime."""
        if t < 30.0:
            return "PAD_ABORT"
        elif altitude < 40000.0:
            return "RTLS"  # Return to Launch Site
        elif altitude < 80000.0 and velocity < 3000.0:
            return "DOWNRANGE_ABORT"
        else:
            return "ATO"  # Abort to Orbit (or ballistic)
