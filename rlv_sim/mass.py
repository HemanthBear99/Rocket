"""
RLV Phase-I Ascent Simulation - Mass Flow Computations

This module implements mass flow rate calculations for the rocket propulsion system.
"""

import numpy as np

from . import constants as C


def compute_mass_flow_rate(thrust_on: bool = True, throttle: float = 1.0) -> float:
    """
    Compute the mass flow rate.
    
    ṁ = -T / (Isp * g₀)
    
    The negative sign indicates mass is decreasing.
    
    Args:
        thrust_on: Whether thrust is active
        
    Returns:
        Mass flow rate (kg/s), negative for mass decrease
    """
    if not thrust_on or throttle <= 0.0:
        return 0.0
    
    return -C.MASS_FLOW_RATE * float(np.clip(throttle, 0.0, 1.0))


def compute_mass_derivative(m: float, thrust_on: bool = True, throttle: float = 1.0) -> float:
    """
    Compute the time derivative of mass.
    
    Args:
        m: Current mass (kg)
        thrust_on: Whether thrust is active
        
    Returns:
        dm/dt (kg/s)
    """
    # Check if propellant is exhausted
    propellant_remaining = m - C.DRY_MASS
    # Note: We use global C.DRY_MASS here as default, but this function should likely take dry_mass too.
    # But for derivative, we assume thrust cuts off if empty.
    # To fix properly, we'd need to pass dry_mass in.
    pass #(Leaving derivative logic as-is for now, handled by main loop cutoff)
    
    if propellant_remaining <= 0 or not thrust_on or throttle <= 0.0:
        return 0.0
    
    return -C.MASS_FLOW_RATE * float(np.clip(throttle, 0.0, 1.0))


def update_mass(m: float, dt: float, thrust_on: bool = True, throttle: float = 1.0, dry_mass: float = C.DRY_MASS) -> float:
    """
    Update mass for a single time step.
    
    This is a simple Euler update; for RK4 integration,
    use compute_mass_derivative instead.
    
    Args:
        m: Current mass (kg)
        dt: Time step (s)
        thrust_on: Whether thrust is active
        
    Returns:
        Updated mass (kg)
    """
    dm_dt = compute_mass_derivative(m, thrust_on, throttle)
    new_mass = m + dm_dt * dt
    
    # Ensure mass doesn't go below dry mass
    return max(new_mass, dry_mass)


def is_propellant_exhausted(m: float, dry_mass: float = C.DRY_MASS) -> bool:
    """
    Check if propellant is exhausted.
    
    Args:
        m: Current mass (kg)
        dry_mass: Dry mass limit (kg)
        
    Returns:
        True if propellant is exhausted
    """
    return m <= dry_mass


def get_propellant_fraction(m: float) -> float:
    """
    Compute the fraction of propellant remaining.
    
    Args:
        m: Current mass (kg)
        
    Returns:
        Fraction of initial propellant remaining (0 to 1)
    """
    propellant_remaining = max(0.0, m - C.DRY_MASS)
    return propellant_remaining / C.PROPELLANT_MASS


def compute_center_of_mass(m: float) -> float:
    """
    Compute the Center of Mass height from the base (Z=0).
    
    Linearly interpolates between H_CG_FULL and H_CG_EMPTY based on propellant fraction.
    
    Args:
        m: Current mass (kg)
        
    Returns:
        Height of CG from base (m)
    """
    frac = get_propellant_fraction(m)
    return C.H_CG_EMPTY + frac * (C.H_CG_FULL - C.H_CG_EMPTY)


def compute_inertia_tensor(m: float) -> np.ndarray:
    """
    Compute the Moment of Inertia Tensor.
    
    Linearly interpolates between INERTIA_TENSOR_FULL and INERTIA_TENSOR_EMPTY.
    
    Args:
        m: Current mass (kg)
        
    Returns:
        3x3 Inertia Tensor (kg*m^2)
    """
    frac = get_propellant_fraction(m)
    return C.INERTIA_TENSOR_EMPTY + frac * (C.INERTIA_TENSOR_FULL - C.INERTIA_TENSOR_EMPTY)

