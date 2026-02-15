"""
RLV Ascent Simulation - Propellant Slosh Model

Models propellant slosh as a pendulum (mechanical analogy) coupled to
the rigid-body translational and rotational dynamics.

The pendulum mass fraction moves laterally in response to vehicle
accelerations, creating a parasitic CG offset and coupling torque.

Enabled via config.enable_slosh = True.
"""

import numpy as np

from . import constants as C


class SloshState:
    """Per-vehicle propellant slosh state (pendulum model)."""

    def __init__(self, mass_fraction: float = 0.20,
                 frequency_hz: float = 0.3,
                 damping_ratio: float = 0.01):
        """
        Args:
            mass_fraction: Fraction of propellant that participates in slosh
            frequency_hz: Pendulum natural frequency (Hz)
            damping_ratio: Effective damping from baffles
        """
        self.mass_fraction = mass_fraction
        self.omega_n = 2.0 * np.pi * frequency_hz
        self.zeta = damping_ratio

        # Pendulum displacement and rate (lateral, metres)
        self.x_slosh = 0.0
        self.x_dot_slosh = 0.0

    def update(self, lateral_accel: float, propellant_mass: float,
               dt: float) -> dict:
        """
        Advance slosh state by one timestep.

        The pendulum is driven by lateral acceleration (body Y or X axis):
            x_ddot = a_lateral - 2*zeta*omega_n*x_dot - omega_n^2 * x

        Args:
            lateral_accel: Lateral acceleration in body frame (m/s^2)
            propellant_mass: Current propellant mass (kg)
            dt: Time step (s)

        Returns:
            dict with x_slosh (m), x_dot (m/s), slosh_mass (kg),
            cg_offset (m), slosh_torque (N*m)
        """
        slosh_mass = propellant_mass * self.mass_fraction

        if slosh_mass < 1.0:
            # Not enough propellant to slosh
            self.x_slosh = 0.0
            self.x_dot_slosh = 0.0
            return {
                'x_slosh': 0.0,
                'x_dot': 0.0,
                'slosh_mass': 0.0,
                'cg_offset': 0.0,
                'slosh_torque': np.zeros(3),
            }

        # Equation of motion
        x_ddot = (lateral_accel
                  - 2.0 * self.zeta * self.omega_n * self.x_dot_slosh
                  - self.omega_n ** 2 * self.x_slosh)

        # Symplectic Euler
        self.x_dot_slosh += x_ddot * dt
        self.x_slosh += self.x_dot_slosh * dt

        # Clamp to physical bounds (tank radius)
        tank_radius = C.REFERENCE_DIAMETER / 2.0 * 0.9  # ~90% of tank ID
        self.x_slosh = np.clip(self.x_slosh, -tank_radius, tank_radius)

        # CG offset from slosh
        total_mass = propellant_mass / self.mass_fraction  # approximate total
        if total_mass > 1.0:
            cg_offset = slosh_mass * self.x_slosh / total_mass
        else:
            cg_offset = 0.0

        # Slosh-induced torque about vehicle CG (body frame)
        # T = slosh_mass * g_axial * x_slosh (pendulum restoring torque)
        # Approximation: uses local axial acceleration (~g0 at launch)
        g_axial = C.G0  # Simplified; could use actual axial accel
        torque_mag = slosh_mass * g_axial * self.x_slosh
        # Acts primarily in roll (body X) direction
        slosh_torque = np.array([torque_mag, 0.0, 0.0])

        return {
            'x_slosh': self.x_slosh,
            'x_dot': self.x_dot_slosh,
            'slosh_mass': slosh_mass,
            'cg_offset': cg_offset,
            'slosh_torque': slosh_torque,
        }
