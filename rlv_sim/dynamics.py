"""
RLV Phase-I Ascent Simulation - Dynamics Equations

This module implements the equations of motion:
- Rotational dynamics: ω̇ = I⁻¹ (τ − ω × (Iω))
- Translational dynamics: r̈ = F_total / m
- Quaternion kinematics: q̇ = 0.5 * Ω(ω) * q
"""

import numpy as np

from . import constants as C
from .frames import quaternion_derivative, quaternion_normalize
from .forces import compute_total_force
from .mass import compute_mass_derivative


def compute_angular_acceleration(omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
    """
    Compute angular acceleration from Euler's equation.
    
    ω̇ = I⁻¹ (τ − ω × (Iω))
    
    Args:
        omega: Angular velocity in body frame (rad/s)
        torque: Control torque in body frame (N*m)
        
    Returns:
        Angular acceleration in body frame (rad/s²)
    """
    # I * omega
    I_omega = C.INERTIA_TENSOR @ omega
    
    # omega × (I * omega)
    gyroscopic = np.cross(omega, I_omega)
    
    # τ - ω × (Iω)
    net_torque = torque - gyroscopic
    
    # I⁻¹ * net_torque
    omega_dot = C.INERTIA_TENSOR_INV @ net_torque
    
    return omega_dot


def compute_linear_acceleration(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                                m: float, thrust_on: bool = True) -> np.ndarray:
    """
    Compute linear acceleration from Newton's second law.
    
    r̈ = F_total / m
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        q: Orientation quaternion [w, x, y, z]
        m: Vehicle mass (kg)
        thrust_on: Whether thrust is active
        
    Returns:
        Linear acceleration in inertial frame (m/s²)
    """
    F_total = compute_total_force(r, v, q, m, thrust_on)
    
    if m < 1e-10:
        return np.zeros(3)
    
    return F_total / m


def compute_state_derivative(r: np.ndarray, v: np.ndarray, q: np.ndarray,
                             omega: np.ndarray, m: float, torque: np.ndarray,
                             thrust_on: bool = True) -> dict:
    """
    Compute all state derivatives for the full dynamics.
    
    This is the core function that computes:
    - ṙ = v
    - v̇ = F/m
    - q̇ = 0.5 * Ω(ω) * q
    - ω̇ = I⁻¹(τ - ω × Iω)
    - ṁ = -T/(Isp * g₀)
    
    Args:
        r: Position vector in inertial frame (m)
        v: Velocity vector in inertial frame (m/s)
        q: Orientation quaternion [w, x, y, z]
        omega: Angular velocity in body frame (rad/s)
        m: Vehicle mass (kg)
        torque: Control torque in body frame (N*m)
        thrust_on: Whether thrust is active
        
    Returns:
        Dictionary of state derivatives
    """
    # Position derivative = velocity
    r_dot = v
    
    # Velocity derivative = acceleration
    v_dot = compute_linear_acceleration(r, v, q, m, thrust_on)
    
    # Quaternion derivative from kinematics
    q_dot = quaternion_derivative(q, omega)
    
    # Angular velocity derivative from rotational dynamics
    omega_dot = compute_angular_acceleration(omega, torque)
    
    # Mass derivative
    m_dot = compute_mass_derivative(m, thrust_on)
    
    return {
        'r_dot': r_dot,
        'v_dot': v_dot,
        'q_dot': q_dot,
        'omega_dot': omega_dot,
        'm_dot': m_dot
    }


def state_derivative_vector(state_vec: np.ndarray, t: float, 
                           torque: np.ndarray, thrust_on: bool = True) -> np.ndarray:
    """
    Compute state derivative as a flat vector for numerical integration.
    
    This function is suitable for use with ODE integrators that require
    the derivative in vector form: dy/dt = f(t, y)
    
    State vector format: [r(3), v(3), q(4), omega(3), m(1)] = 14 elements
    
    Args:
        state_vec: Current state as flat array [14]
        t: Current time (s)
        torque: Control torque in body frame (N*m)
        thrust_on: Whether thrust is active
        
    Returns:
        State derivative as flat array [14]
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
    derivs = compute_state_derivative(r, v, q, omega, m, torque, thrust_on)
    
    # Pack into vector
    return np.concatenate([
        derivs['r_dot'],
        derivs['v_dot'],
        derivs['q_dot'],
        derivs['omega_dot'],
        [derivs['m_dot']]
    ])
