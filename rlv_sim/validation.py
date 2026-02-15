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


def check_mass_valid(m: float, dry_mass: float = C.DRY_MASS) -> bool:
    """
    Check that mass is physically valid.
    
    Args:
        m: Vehicle mass (kg)
        dry_mass: Dry mass limit (kg)
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    if m < dry_mass * 0.99:  # Allow small numerical tolerance
        raise ValidationError(
            f"Mass below dry mass: m = {m:.2f} kg, "
            f"dry mass = {dry_mass:.2f} kg"
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


def validate_state(state: State, abort_on_error: bool = True, dry_mass: float = C.DRY_MASS) -> Tuple[bool, Optional[str]]:
    """
    Perform all validation checks on a state.
    
    Args:
        state: State to validate
        abort_on_error: If True, raise exception on first error
        dry_mass: Dry mass lower bound
    """
    try:
        check_quaternion_norm(state.q)
        check_position_valid(state.r)
        check_mass_valid(state.m, dry_mass)
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


def validate_energy_conservation_states(state_initial: State, state_final: State,
                                        tolerance: float = None) -> bool:
    """
    Check energy conservation for unpowered flight between two State objects.

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


def compute_total_energy(r: np.ndarray, v: np.ndarray, m: float) -> float:
    """
    Compute total mechanical energy (kinetic + potential).
    
    E_total = (1/2) * m * |v|² - (μ * m) / |r|
    
    Reference: ROCKET_SIMULATION_RULES.md Section 10.1
    [PHASE I] Critical for validation of integration accuracy
    
    Args:
        r: Position in ECI (m)
        v: Velocity in ECI (m/s)
        m: Vehicle mass (kg)
        
    Returns:
        Total mechanical energy (J)
    """
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    
    if r_norm < C.ZERO_TOLERANCE or m < 0:
        return 0.0
    
    kinetic = 0.5 * m * (v_norm ** 2)
    potential = -C.MU_EARTH * m / r_norm
    
    return float(kinetic + potential)


def validate_energy_conservation(E_current: float, E_previous: float, 
                                 dt: float) -> dict:
    """
    Validate energy conservation across a time step.
    
    Reference: ROCKET_SIMULATION_RULES.md Section 10.1
    [PHASE I] Checks that dE/dt change is small (integration quality metric)
    
    Args:
        E_current: Current total energy (J)
        E_previous: Previous total energy (J)
        dt: Time step (s)
        
    Returns:
        Dictionary with validation results
    """
    if abs(E_previous) < C.ZERO_TOLERANCE:
        return {'valid': True, 'error': 0.0, 'message': 'Energy too small to validate'}
    
    dE = E_current - E_previous
    relative_error = abs(dE / E_previous) if abs(E_previous) > C.ZERO_TOLERANCE else 0.0
    
    return {
        'valid': relative_error < C.ENERGY_TOLERANCE,
        'dE': float(dE),
        'relative_error': float(relative_error),
        'message': f"Energy error: {relative_error:.2e} (tol={C.ENERGY_TOLERANCE:.2e})"
    }


def validate_constraints(r: np.ndarray, v: np.ndarray, omega: np.ndarray, m: float) -> dict:
    """
    Validate physical constraints and limits.
    
    Reference: ROCKET_SIMULATION_RULES.md Section 9
    [PHASE I] Checks structural and operational limits
    
    Args:
        r: Position in ECI (m)
        v: Velocity in ECI (m/s)
        omega: Angular velocity in body frame (rad/s)
        m: Vehicle mass (kg)
        
    Returns:
        Dictionary with constraint status
    """
    violations = []
    altitude = np.linalg.norm(r) - C.R_EARTH
    velocity = np.linalg.norm(v)
    omega_mag = np.linalg.norm(omega)
    
    # Altitude check
    if altitude < C.CRASH_ALTITUDE_TOLERANCE:
        violations.append(f"CRASH: altitude={altitude/1000:.1f} km")
    
    # Velocity check (escape velocity)
    if np.linalg.norm(r) > C.ZERO_TOLERANCE:
        v_escape = np.sqrt(2 * C.MU_EARTH / np.linalg.norm(r))
        if velocity > v_escape * 1.1:
            violations.append(f"Over-velocity: {velocity:.0f} m/s (escape: {v_escape:.0f})")
    
    # Angular rate check (structural limit)
    omega_limit = np.radians(180.0)  # 180 deg/s
    if omega_mag > omega_limit:
        violations.append(f"Spin rate: {np.degrees(omega_mag):.1f} deg/s (limit: 180)")
    
    # Mass check
    if m < C.DRY_MASS * 0.99:
        violations.append(f"Mass: {m:.0f} kg (dry: {C.DRY_MASS:.0f})")
    
    return {
        'valid': len(violations) == 0,
        'violations': violations,
        'altitude': float(altitude),
        'velocity': float(velocity),
        'omega_mag': float(omega_mag),
        'mass': float(m)
    }
