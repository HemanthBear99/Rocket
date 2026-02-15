"""
RLV Ascent Simulation - Aerothermal Heating Model

Implements Sutton-Graves convective heating and radiative equilibrium
wall temperature for ascent and re-entry thermal protection analysis.

Enabled via config.enable_thermal = True.
"""

import numpy as np

from . import constants as C
from .forces import compute_atmosphere_properties


def sutton_graves_heating(rho: float, v: float, r_nose: float,
                          k: float = 1.7415e-4) -> float:
    """
    Sutton-Graves stagnation-point convective heating rate.

    q_dot = k * sqrt(rho / r_n) * V^3

    Args:
        rho: Atmospheric density (kg/m^3)
        v: Air-relative speed (m/s)
        r_nose: Effective nose radius (m)
        k: Sutton-Graves constant (default for N2/O2 air)

    Returns:
        Heating rate at stagnation point (W/m^2)
    """
    if rho < 1e-15 or v < 10.0 or r_nose < 0.001:
        return 0.0
    return k * np.sqrt(rho / r_nose) * v ** 3


def radiative_equilibrium_temperature(q_dot: float, emissivity: float = 0.85) -> float:
    """
    Compute radiative equilibrium wall temperature.

    T_w = (q_dot / (epsilon * sigma))^(1/4)

    where sigma = 5.67e-8 W/(m^2 K^4) is the Stefan-Boltzmann constant.

    Args:
        q_dot: Incident heat flux (W/m^2)
        emissivity: Surface emissivity (0-1)

    Returns:
        Wall temperature (K)
    """
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant
    if q_dot <= 0.0 or emissivity <= 0.0:
        return 300.0  # ambient
    return (q_dot / (emissivity * sigma)) ** 0.25


class ThermalState:
    """Mutable thermal state tracked per vehicle."""

    def __init__(self, wall_temp: float = 300.0, total_heat_load: float = 0.0):
        self.wall_temp = wall_temp        # K
        self.total_heat_load = total_heat_load  # J/m^2 (integrated)

    def update(self, altitude: float, v_rel: float, dt: float,
               r_nose: float = None, k: float = 1.7415e-4,
               emissivity: float = 0.85, wall_thickness: float = 0.005,
               material_cp: float = 900.0, material_density: float = 2700.0):
        """
        Advance thermal state by one timestep.

        Args:
            altitude: Current altitude (m)
            v_rel: Air-relative speed (m/s)
            dt: Time step (s)
            r_nose: Nose radius (m). Defaults to REFERENCE_DIAMETER / 2.
            k: Sutton-Graves constant
            emissivity: Surface emissivity
            wall_thickness: TPS wall thickness (m)
            material_cp: Specific heat (J/(kg K))
            material_density: Density (kg/m^3)

        Returns:
            dict with q_dot (W/m^2), T_rad_eq (K), T_wall (K)
        """
        if r_nose is None:
            r_nose = C.REFERENCE_DIAMETER / 2.0

        _, _, rho, _ = compute_atmosphere_properties(altitude)
        q_dot = sutton_graves_heating(rho, v_rel, r_nose, k)
        t_rad_eq = radiative_equilibrium_temperature(q_dot, emissivity)

        # Simple lumped-capacity wall temperature update
        # m_wall * cp * dT/dt = q_dot_in - q_dot_rad_out
        sigma = 5.670374419e-8
        q_rad_out = emissivity * sigma * self.wall_temp ** 4
        q_net = q_dot - q_rad_out

        mass_per_area = material_density * wall_thickness
        if mass_per_area > 0.01:
            dT = q_net * dt / (mass_per_area * material_cp)
            self.wall_temp = max(200.0, self.wall_temp + dT)

        self.total_heat_load += q_dot * dt

        return {
            'q_dot': q_dot,
            'T_rad_eq': t_rad_eq,
            'T_wall': self.wall_temp,
            'total_heat_load': self.total_heat_load,
        }
