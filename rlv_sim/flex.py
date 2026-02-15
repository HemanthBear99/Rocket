"""
RLV Ascent Simulation - Flexible Body First Bending Mode

Models the first lateral bending mode of the launch vehicle as a
spring-mass-damper system coupled to the rigid-body dynamics.

The bending mode adds a parasitic torque to the control system
and can couple with the attitude controller (tail-wags-dog effect).

Enabled via config.enable_flex_body = True.
"""

import numpy as np


class FlexState:
    """Per-vehicle flexible body state (first bending mode)."""

    def __init__(self, frequency_hz: float = 2.5, damping_ratio: float = 0.02,
                 mode_slope: float = 0.1):
        """
        Args:
            frequency_hz: First bending mode natural frequency (Hz)
            damping_ratio: Structural damping ratio (zeta)
            mode_slope: Mode shape slope at gimbal point (rad/m)
        """
        self.omega_n = 2.0 * np.pi * frequency_hz  # Natural frequency (rad/s)
        self.zeta = damping_ratio
        self.mode_slope = mode_slope

        # Generalised coordinate and rate
        self.eta = 0.0       # Bending displacement (m, generalised)
        self.eta_dot = 0.0   # Bending rate (m/s)

    def update(self, gimbal_angle: float, thrust_force: float,
               generalised_mass: float, dt: float) -> dict:
        """
        Advance bending mode state by one timestep.

        The gimbal force excites the bending mode:
            M_gen * eta_ddot + 2*zeta*omega_n*M_gen*eta_dot + M_gen*omega_n^2*eta
                = F_thrust * sin(gimbal_angle) * mode_slope

        Args:
            gimbal_angle: Current gimbal deflection angle (rad)
            thrust_force: Current thrust magnitude (N)
            generalised_mass: Generalised mass of first bending mode (kg)
            dt: Time step (s)

        Returns:
            dict with eta (m), eta_dot (m/s), parasitic_angle (rad)
        """
        if generalised_mass < 1.0:
            generalised_mass = 1.0

        # Forcing function: thrust component exciting the bending mode
        forcing = thrust_force * np.sin(gimbal_angle) * self.mode_slope

        # State space: eta_ddot = forcing/M_gen - 2*zeta*omega_n*eta_dot - omega_n^2*eta
        eta_ddot = (forcing / generalised_mass
                    - 2.0 * self.zeta * self.omega_n * self.eta_dot
                    - self.omega_n ** 2 * self.eta)

        # Symplectic Euler integration
        self.eta_dot += eta_ddot * dt
        self.eta += self.eta_dot * dt

        # Clamp to prevent runaway
        max_eta = 0.5  # m
        self.eta = np.clip(self.eta, -max_eta, max_eta)

        # Parasitic angle at sensor/gimbal location
        parasitic_angle = self.eta * self.mode_slope

        return {
            'eta': self.eta,
            'eta_dot': self.eta_dot,
            'parasitic_angle': parasitic_angle,
        }

    def get_parasitic_torque(self, inertia: float) -> np.ndarray:
        """
        Get the parasitic torque that bending adds to rigid-body dynamics.

        Approximation: the bending deflection creates a body-frame torque
        in the pitch/yaw plane proportional to stiffness * deflection.

        Args:
            inertia: Vehicle moment of inertia (kg*m^2)

        Returns:
            Parasitic torque vector in body frame (N*m)
        """
        # Torque ~ omega_n^2 * eta * mode_slope * inertia
        # This couples bending into the rotational dynamics
        torque_mag = self.omega_n ** 2 * self.eta * self.mode_slope * inertia * 0.01
        # Acts primarily in the Y-axis (pitch) body frame
        return np.array([0.0, torque_mag, 0.0])
