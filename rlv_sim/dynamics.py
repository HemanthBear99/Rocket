"""
RLV Phase-I Ascent Simulation - Dynamics Equations

This module implements the equations of motion:
- Rotational dynamics: ω̇ = I⁻¹ (τ − ω × (Iω))
- Translational dynamics: r̈ = F_total / m
- Quaternion kinematics: q̇ = 0.5 * Ω(ω) * q
"""

from typing import NamedTuple

import numpy as np


from . import constants as C
from .frames import quaternion_derivative, quaternion_normalize
from .forces import compute_total_force, compute_aerodynamic_moment, compute_gravity_force, compute_drag_force, compute_thrust_force
from .mass import compute_mass_derivative, compute_inertia_tensor, compute_center_of_mass


class StateDerivative(NamedTuple):
    """Container for state derivatives."""
    r_dot: np.ndarray
    v_dot: np.ndarray
    q_dot: np.ndarray
    omega_dot: np.ndarray
    m_dot: float


def compute_angular_acceleration(omega: np.ndarray, torque: np.ndarray, 
                                 I_tensor: np.ndarray, I_inv: np.ndarray) -> np.ndarray:
    """
    Compute angular acceleration from Euler's equation with VARIABLE INERTIA.
    
    ω̇ = I⁻¹ (τ − ω × (Iω))
    
    Args:
        omega: Angular velocity in body frame (rad/s)
        torque: Total torque (Control + Aero) in body frame (N*m)
        I_tensor: Current Inertia Tensor (kg*m^2)
        I_inv: Inverse Inertia Tensor
        
    Returns:
        Angular acceleration in body frame (rad/s²)
    """
    # I * omega
    I_omega = I_tensor @ omega
    
    # omega × (I * omega)
    gyroscopic = np.cross(omega, I_omega)
    
    # τ - ω × (Iω)
    net_torque = torque - gyroscopic
    
    # I⁻¹ * net_torque
    omega_dot = I_inv @ net_torque
    
    return omega_dot


def compute_linear_acceleration(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                                m: float, thrust_on: bool = True, throttle: float = 1.0) -> np.ndarray:
    """
    Compute linear acceleration from Newton's second law.
    
    r̈ = (F_grav + F_thrust + F_drag + F_coriolis) / m
    
    Reference: ROCKET_SIMULATION_RULES.md Section 2.1
    [PHASE I] [FIX #1] Includes Coriolis force for high-altitude accuracy
    """
    from .forces import compute_coriolis_force
    
    F_grav = compute_gravity_force(r, m)
    F_thrust = compute_thrust_force(q, r, thrust_on, throttle)
    F_drag = compute_drag_force(r, v)
    F_coriolis = compute_coriolis_force(v, m)  # [FIX #1] Add Coriolis force
    
    F_total = F_grav + F_thrust + F_drag + F_coriolis  # [FIX #1] Include all forces
    
    if m < C.ZERO_TOLERANCE:
        return np.zeros(3)
    
    return F_total / m


def compute_state_derivative(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                             omega: np.ndarray, m: float, torque: np.ndarray,
                             thrust_on: bool = True, throttle: float = 1.0) -> StateDerivative:
    """
    Compute all state derivatives for the full dynamics.
    
    Includes:
    - Variable Inertia
    - Aerodynamic Moments (Instability)
    - Throttling
    """
    # Position derivative = velocity
    r_dot = v
    
    # Velocity derivative = acceleration
    v_dot = compute_linear_acceleration(r, v, q, m, thrust_on, throttle)
    
    # Quaternion derivative from kinematics
    q_dot = quaternion_derivative(q, omega)
    
    # --- ROTATIONAL DYNAMICS UPGRADE ---
    
    # 1. Update Inertia Properties
    I_tensor = compute_inertia_tensor(m)
    # Optimized: For diagonal inertia tensors, inverse is trivial
    I_inv = np.diag(1.0 / np.diag(I_tensor))
    
    # 2. Compute Aerodynamic Instability Torque
    # Need CG position for moment arm
    cg_pos_z = compute_center_of_mass(m)
    tau_aero = compute_aerodynamic_moment(r, v, q, cg_pos_z)
    
    # 3. Total Torque = Control Torque + Aero Torque
    total_torque = torque + tau_aero
    
    # Angular velocity derivative
    omega_dot = compute_angular_acceleration(omega, total_torque, I_tensor, I_inv)
    
    # Mass derivative
    # Note: throttling also affects mass flow!
    # m_dot = -mdot * throttle
    # compute_mass_derivative currently assumes constant flow.
    # Need to scale it by throttle.
    m_dot_nominal = compute_mass_derivative(m, thrust_on)
    m_dot = m_dot_nominal * throttle
    
    return StateDerivative(
        r_dot=r_dot,
        v_dot=v_dot,
        q_dot=q_dot,
        omega_dot=omega_dot,
        m_dot=m_dot
    )


def state_derivative_vector(state_vec: np.ndarray, t: float, 
                           torque: np.ndarray, thrust_on: bool = True, throttle: float = 1.0) -> np.ndarray:
    """
    Compute state derivative as a flat vector for numerical integration.
    """
    # Unpack state
    r = state_vec[0:3]
    v = state_vec[3:6]
    q = state_vec[6:10]
    omega = state_vec[10:13]
    m = state_vec[13]
    
    # Normalize quaternion (safety)
    q = quaternion_normalize(q)
    
    # Compute derivatives
    derivs = compute_state_derivative(r, v, q, omega, m, torque, thrust_on, throttle)
    
    # Pack into vector
    return np.concatenate([
        derivs.r_dot,
        derivs.v_dot,
        derivs.q_dot,
        derivs.omega_dot,
        [derivs.m_dot]
    ])




