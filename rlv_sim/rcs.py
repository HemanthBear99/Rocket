"""
RLV Ascent Simulation - Reaction Control System (RCS)

Simple RCS thruster model for attitude control when main engine gimbal
authority is insufficient (coast phases, re-entry, fine pointing).

Enabled via config.enable_rcs = True.
"""

import numpy as np

from . import constants as C


class RCSState:
    """Per-vehicle RCS state."""

    def __init__(self, propellant_mass: float = 200.0):
        """
        Args:
            propellant_mass: Initial RCS propellant (kg). Typical: 100-300 kg.
        """
        self.propellant_mass = propellant_mass
        self.total_impulse_used = 0.0  # N*s

    @property
    def propellant_remaining(self) -> float:
        return max(self.propellant_mass, 0.0)

    @property
    def is_available(self) -> bool:
        return self.propellant_mass > 0.1  # 100 g minimum


def compute_rcs_torque(omega_error: np.ndarray, rcs_state: RCSState,
                       dt: float, thrust_per_thruster: float = 500.0,
                       isp: float = 220.0, arm_length: float = 1.5,
                       deadband: float = 0.001) -> tuple:
    """
    Compute RCS torque for attitude control.

    Uses a bang-bang (on-off) controller with deadband to manage
    propellant consumption. Each axis has a pair of opposing thrusters.

    Args:
        omega_error: Angular velocity error in body frame (rad/s)
        rcs_state: Current RCS state
        dt: Time step (s)
        thrust_per_thruster: Thrust per thruster (N)
        isp: Specific impulse (s)
        arm_length: Moment arm from CG to thruster (m)
        deadband: Angular rate deadband below which RCS does not fire (rad/s)

    Returns:
        (torque_vector, updated_rcs_state) - torque in body frame (N*m)
    """
    if not rcs_state.is_available:
        return np.zeros(3), rcs_state

    torque = np.zeros(3)
    mass_consumed = 0.0

    for axis in range(3):
        rate = omega_error[axis]
        if abs(rate) > deadband:
            # Fire thruster pair to oppose rate
            t_mag = thrust_per_thruster * arm_length
            torque[axis] = -np.sign(rate) * t_mag

            # Propellant consumption: m_dot = F / (Isp * g0)
            m_dot = thrust_per_thruster / (isp * C.G0)
            mass_consumed += m_dot * dt

    # Deduct propellant
    rcs_state.propellant_mass = max(0.0, rcs_state.propellant_mass - mass_consumed)
    rcs_state.total_impulse_used += np.linalg.norm(torque) * dt

    return torque, rcs_state
