"""
RLV Phase-I Ascent Simulation - Mass Flow Computations

This module implements mass flow rate calculations for the rocket propulsion system.
"""

import numpy as np

from . import constants as C


def compute_mass_flow_rate(thrust_on: bool = True, throttle: float = 1.0, stage: int = 1) -> float:
    """
    Compute the mass flow rate.

    ṁ = -T / (Isp * g₀)

    The negative sign indicates mass is decreasing.

    Args:
        thrust_on: Whether thrust is active
        throttle: Throttle setting (0.0 to 1.0)
        stage: Engine stage (1 = S1 engines, 2 = S2 engine)

    Returns:
        Mass flow rate (kg/s), negative for mass decrease
    """
    if not thrust_on or throttle <= 0.0:
        return 0.0

    mdot = C.STAGE2_MASS_FLOW_RATE if stage == 2 else C.MASS_FLOW_RATE
    return -mdot * float(np.clip(throttle, 0.0, 1.0))


def compute_mass_derivative(m: float, thrust_on: bool = True, throttle: float = 1.0,
                            dry_mass: float = None, stage: int = 1) -> float:
    """
    Compute the time derivative of mass.

    Args:
        m: Current mass (kg)
        thrust_on: Whether thrust is active
        throttle: Throttle setting (0.0 to 1.0)
        dry_mass: Dry mass limit (kg). Defaults to C.DRY_MASS (stacked vehicle).
        stage: Engine stage (1 = S1 engines, 2 = S2 engine)

    Returns:
        dm/dt (kg/s)
    """
    if dry_mass is None:
        dry_mass = C.DRY_MASS

    # Check if propellant is exhausted
    propellant_remaining = m - dry_mass

    if propellant_remaining <= 0 or not thrust_on or throttle <= 0.0:
        return 0.0

    mdot = C.STAGE2_MASS_FLOW_RATE if stage == 2 else C.MASS_FLOW_RATE
    return -mdot * float(np.clip(throttle, 0.0, 1.0))


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

    For stacked vehicle (m > DRY_MASS): interpolates S1 propellant fraction.
    For S2 vehicle (m < DRY_MASS): returns 0 (S1 propellant exhausted).

    Args:
        m: Current mass (kg)

    Returns:
        Fraction of S1 propellant remaining (0 to 1), clamped.
    """
    propellant_remaining = max(0.0, m - C.DRY_MASS)
    return min(1.0, propellant_remaining / C.PROPELLANT_MASS)


def compute_center_of_mass(m: float) -> float:
    """
    Compute the Center of Mass height from the base (Z=0).

    Linearly interpolates between H_CG_FULL and H_CG_EMPTY based on propellant fraction.
    For S2 vehicle (post-separation), uses a simplified CG model.

    Args:
        m: Current mass (kg)

    Returns:
        Height of CG from base (m)
    """
    if m < C.DRY_MASS:
        # Post-separation S2 vehicle: CG is near middle of S2 stage
        # S2 is the upper 20m of the stack (40m to 60m from base)
        # CG shifts from middle (50m) down as propellant depletes
        s2_frac = max(0.0, min(1.0, (m - C.STAGE2_DRY_MASS) / C.STAGE2_PROPELLANT_MASS))
        return 45.0 + s2_frac * 5.0  # CG between 45m (empty) and 50m (full)
    frac = get_propellant_fraction(m)
    return C.H_CG_EMPTY + frac * (C.H_CG_FULL - C.H_CG_EMPTY)


def compute_inertia_tensor(m: float) -> np.ndarray:
    """
    Compute the Moment of Inertia Tensor.

    For stacked vehicle: interpolates between FULL and EMPTY tensors.
    For S2 vehicle (post-separation): uses simplified S2 inertia model.

    Args:
        m: Current mass (kg)

    Returns:
        3x3 Inertia Tensor (kg*m^2)
    """
    if m < C.DRY_MASS:
        # Post-separation S2 vehicle (doc §A.3.2)
        # Doc specifies I2 = diag(2.5e6, 2.5e6, 4.0e5) at full S2 mass (120,000 kg)
        # S2 is ~20m tall, ~3.7m diameter, mass 8000-120000 kg
        # Ixx scales roughly linearly with mass for uniform cylinder
        # For m=8t (dry):  Ixx ≈ 2.5e6 * (8000/120000) ≈ 1.67e5 kg·m²
        s2_frac = max(0.0, min(1.0, (m - C.STAGE2_DRY_MASS) / C.STAGE2_PROPELLANT_MASS))
        ixx_s2_full = 2.5e6   # kg·m² (S2 wet, doc §A.3.2)
        ixx_s2_empty = 1.67e5 # kg·m² (S2 dry, scaled from doc value)
        izz_s2_full = 4.0e5   # kg·m² (roll axis, doc §A.3.2)
        izz_s2_empty = 1.0e5  # kg·m² (scaled)
        ixx = ixx_s2_empty + s2_frac * (ixx_s2_full - ixx_s2_empty)
        izz = izz_s2_empty + s2_frac * (izz_s2_full - izz_s2_empty)
        return np.diag([ixx, ixx, izz])

    frac = get_propellant_fraction(m)
    return C.INERTIA_TENSOR_EMPTY + frac * (C.INERTIA_TENSOR_FULL - C.INERTIA_TENSOR_EMPTY)

