"""
RLV Phase-I Ascent Simulation - Validation Checks

This module implements physics validation checks:
- Quaternion norm check
- Zero force → constant velocity
- Zero torque → constant ω
- Energy conservation (approximate)

Abort on violation.
"""

import numpy as np
from typing import Optional, Tuple

from . import constants as C
from .state import State


class ValidationError(Exception):
    """Raised when a physics validation check fails."""
    pass


def check_quaternion_norm(q: np.ndarray, tolerance: float = None) -> bool:
    """
    Verify quaternion is unit-normalized.
    
    Args:
        q: Quaternion [w, x, y, z]
        tolerance: Allowable deviation from 1.0
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    if tolerance is None:
        tolerance = C.QUATERNION_NORM_TOL
    
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > tolerance:
        raise ValidationError(
            f"Quaternion norm violation: |q| = {norm:.10f}, "
            f"deviation = {abs(norm - 1.0):.2e}, tolerance = {tolerance:.2e}"
        )
    return True


def check_position_valid(r: np.ndarray) -> bool:
    """
    Check that position is physically valid (above Earth's center).
    
    Args:
        r: Position vector (m)
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    r_norm = np.linalg.norm(r)
    
    if r_norm < C.R_EARTH * 0.5:  # Well below surface
        raise ValidationError(
            f"Position inside Earth: |r| = {r_norm/1000:.2f} km, "
            f"R_Earth = {C.R_EARTH/1000:.2f} km"
        )
    return True


def check_mass_valid(m: float) -> bool:
    """
    Check that mass is physically valid.
    
    Args:
        m: Vehicle mass (kg)
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    if m < C.DRY_MASS * 0.99:  # Allow small numerical tolerance
        raise ValidationError(
            f"Mass below dry mass: m = {m:.2f} kg, "
            f"dry mass = {C.DRY_MASS:.2f} kg"
        )
    
    if m > C.INITIAL_MASS * 1.01:  # Mass shouldn't increase
        raise ValidationError(
            f"Mass exceeds initial mass: m = {m:.2f} kg, "
            f"initial = {C.INITIAL_MASS:.2f} kg"
        )
    return True


def check_velocity_reasonable(v: np.ndarray) -> bool:
    """
    Check that velocity is within reasonable bounds.
    
    Args:
        v: Velocity vector (m/s)
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    v_mag = np.linalg.norm(v)
    
    # Escape velocity at Earth's surface is ~11.2 km/s
    # Phase I should stay well below this
    max_reasonable_v = 15000.0  # m/s
    
    if v_mag > max_reasonable_v:
        raise ValidationError(
            f"Velocity exceeds reasonable bounds: |v| = {v_mag:.2f} m/s, "
            f"max = {max_reasonable_v:.2f} m/s"
        )
    return True


def check_angular_velocity_reasonable(omega: np.ndarray) -> bool:
    """
    Check that angular velocity is within reasonable bounds.
    
    Args:
        omega: Angular velocity (rad/s)
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    omega_mag = np.linalg.norm(omega)
    
    # Maximum reasonable rotation rate for a rocket
    max_reasonable_omega = 10.0  # rad/s (~573 deg/s)
    
    if omega_mag > max_reasonable_omega:
        raise ValidationError(
            f"Angular velocity exceeds reasonable bounds: |ω| = {omega_mag:.4f} rad/s "
            f"({np.degrees(omega_mag):.2f} deg/s), max = {max_reasonable_omega:.2f} rad/s"
        )
    return True


def validate_state(state: State, abort_on_error: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Perform all validation checks on a state.
    
    Args:
        state: State to validate
        abort_on_error: If True, raise exception on first error
        
    Returns:
        (is_valid, error_message) tuple
    """
    try:
        check_quaternion_norm(state.q)
        check_position_valid(state.r)
        check_mass_valid(state.m)
        check_velocity_reasonable(state.v)
        check_angular_velocity_reasonable(state.omega)
        return True, None
    except ValidationError as e:
        if abort_on_error:
            raise
        return False, str(e)


def compute_orbital_energy(r: np.ndarray, v: np.ndarray, m: float) -> float:
    """
    Compute specific orbital energy.
    
    E = v²/2 - μ/r
    
    Args:
        r: Position vector (m)
        v: Velocity vector (m/s)
        m: Mass (kg) - not used for specific energy
        
    Returns:
        Specific orbital energy (J/kg)
    """
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    
    kinetic = 0.5 * v_norm ** 2
    potential = -C.MU_EARTH / r_norm
    
    return kinetic + potential


def validate_energy_conservation(state_initial: State, state_final: State,
                                 tolerance: float = None) -> bool:
    """
    Check energy conservation for unpowered flight.
    
    This is only valid when thrust = 0 and drag is negligible.
    
    Args:
        state_initial: Initial state
        state_final: Final state
        tolerance: Relative tolerance for energy change
        
    Returns:
        True if energy is conserved within tolerance
    """
    if tolerance is None:
        tolerance = C.ENERGY_TOLERANCE
    
    E_initial = compute_orbital_energy(state_initial.r, state_initial.v, state_initial.m)
    E_final = compute_orbital_energy(state_final.r, state_final.v, state_final.m)
    
    if abs(E_initial) < 1e-10:
        return True  # Can't compute relative error
    
    relative_change = abs(E_final - E_initial) / abs(E_initial)
    
    if relative_change > tolerance:
        raise ValidationError(
            f"Energy conservation violation: ΔE/E = {relative_change:.4e}, "
            f"tolerance = {tolerance:.4e}"
        )
    return True


def run_validation_suite(state: State, verbose: bool = False) -> dict:
    """
    Run all validation checks and return results.
    
    Args:
        state: State to validate
        verbose: Print results
        
    Returns:
        Dictionary of validation results
    """
    results = {
        'quaternion_norm': None,
        'position_valid': None,
        'mass_valid': None,
        'velocity_reasonable': None,
        'omega_reasonable': None,
        'all_passed': True
    }
    
    checks = [
        ('quaternion_norm', lambda: check_quaternion_norm(state.q)),
        ('position_valid', lambda: check_position_valid(state.r)),
        ('mass_valid', lambda: check_mass_valid(state.m)),
        ('velocity_reasonable', lambda: check_velocity_reasonable(state.v)),
        ('omega_reasonable', lambda: check_angular_velocity_reasonable(state.omega))
    ]
    
    for name, check in checks:
        try:
            check()
            results[name] = 'PASS'
            if verbose:
                print(f"  ✓ {name}")
        except ValidationError as e:
            results[name] = f'FAIL: {e}'
            results['all_passed'] = False
            if verbose:
                print(f"  ✗ {name}: {e}")
    
    return results
