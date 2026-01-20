"""
RLV Phase-I Ascent Simulation - Numerical Integration

This module implements the RK4 integrator for all state variables
with quaternion normalization after each step.
"""

import numpy as np

from .state import State
from .frames import quaternion_normalize
from .dynamics import state_derivative_vector


def rk4_step(state: State, torque: np.ndarray, dt: float, 
             thrust_on: bool = True) -> State:
    """
    Perform a single RK4 integration step.
    
    The RK4 method computes:
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    Args:
        state: Current state
        torque: Control torque in body frame (N*m)
        dt: Time step (s)
        thrust_on: Whether thrust is active
        
    Returns:
        New state after integration
    """
    t = state.t
    y = state.to_vector()
    
    # k1 = f(t, y)
    k1 = state_derivative_vector(y, t, torque, thrust_on)
    
    # k2 = f(t + dt/2, y + dt/2 * k1)
    y2 = y + 0.5 * dt * k1
    y2[6:10] = quaternion_normalize(y2[6:10])  # Normalize quaternion
    k2 = state_derivative_vector(y2, t + 0.5*dt, torque, thrust_on)
    
    # k3 = f(t + dt/2, y + dt/2 * k2)
    y3 = y + 0.5 * dt * k2
    y3[6:10] = quaternion_normalize(y3[6:10])
    k3 = state_derivative_vector(y3, t + 0.5*dt, torque, thrust_on)
    
    # k4 = f(t + dt, y + dt * k3)
    y4 = y + dt * k3
    y4[6:10] = quaternion_normalize(y4[6:10])
    k4 = state_derivative_vector(y4, t + dt, torque, thrust_on)
    
    # Final update
    y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Normalize quaternion after full step
    y_new[6:10] = quaternion_normalize(y_new[6:10])
    
    # Ensure mass doesn't go below dry mass
    from . import constants as C
    y_new[13] = max(y_new[13], C.DRY_MASS)
    
    # Create new state
    return State.from_vector(y_new, t + dt)


def euler_step(state: State, torque: np.ndarray, dt: float,
               thrust_on: bool = True) -> State:
    """
    Perform a single Euler integration step.
    
    This is a first-order method, primarily for testing/comparison.
    
    Args:
        state: Current state
        torque: Control torque in body frame (N*m)
        dt: Time step (s)
        thrust_on: Whether thrust is active
        
    Returns:
        New state after integration
    """
    t = state.t
    y = state.to_vector()
    
    # dy/dt = f(t, y)
    dy = state_derivative_vector(y, t, torque, thrust_on)
    
    # y_new = y + dt * dy
    y_new = y + dt * dy
    
    # Normalize quaternion
    y_new[6:10] = quaternion_normalize(y_new[6:10])
    
    # Ensure mass doesn't go below dry mass
    from . import constants as C
    y_new[13] = max(y_new[13], C.DRY_MASS)
    
    return State.from_vector(y_new, t + dt)


def integrate(state: State, torque: np.ndarray, dt: float,
              thrust_on: bool = True, method: str = 'rk4') -> State:
    """
    Integrate the state forward by one timestep.
    
    Args:
        state: Current state
        torque: Control torque in body frame (N*m)
        dt: Time step (s)
        thrust_on: Whether thrust is active
        method: Integration method ('rk4' or 'euler')
        
    Returns:
        New state after integration
    """
    if method == 'rk4':
        return rk4_step(state, torque, dt, thrust_on)
    elif method == 'euler':
        return euler_step(state, torque, dt, thrust_on)
    else:
        raise ValueError(f"Unknown integration method: {method}")
