import pytest
import numpy as np
from rlv_sim import validation
from rlv_sim import constants as C
from rlv_sim.state import State


# ============================================================================
# check_quaternion_norm tests
# ============================================================================

def test_check_quaternion_norm_valid():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert validation.check_quaternion_norm(q)

def test_check_quaternion_norm_invalid():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    with pytest.raises(validation.ValidationError):
        validation.check_quaternion_norm(q)

def test_check_quaternion_norm_custom_tolerance():
    q = np.array([1.001, 0.0, 0.0, 0.0])
    # Should fail with tight tolerance
    with pytest.raises(validation.ValidationError):
        validation.check_quaternion_norm(q, tolerance=1e-6)
    # Should pass with loose tolerance
    assert validation.check_quaternion_norm(q, tolerance=0.01)


# ============================================================================
# check_position_valid tests
# ============================================================================

def test_check_position_valid():
    r = np.array([7e6, 0.0, 0.0])
    assert validation.check_position_valid(r)

def test_check_position_inside_earth():
    r = np.array([1e6, 0.0, 0.0])
    with pytest.raises(validation.ValidationError):
        validation.check_position_valid(r)


# ============================================================================
# check_mass_valid tests
# ============================================================================

def test_check_mass_valid():
    m = C.INITIAL_MASS
    assert validation.check_mass_valid(m)

def test_check_mass_at_dry_mass():
    m = C.DRY_MASS
    assert validation.check_mass_valid(m)

def test_check_mass_below_dry_mass():
    m = C.DRY_MASS * 0.5
    with pytest.raises(validation.ValidationError):
        validation.check_mass_valid(m)

def test_check_mass_above_initial():
    m = C.INITIAL_MASS * 1.5
    with pytest.raises(validation.ValidationError):
        validation.check_mass_valid(m)


# ============================================================================
# check_velocity_reasonable tests
# ============================================================================

def test_check_velocity_reasonable():
    v = np.array([1000.0, 500.0, 0.0])
    assert validation.check_velocity_reasonable(v)

def test_check_velocity_exceeds_limit():
    v = np.array([20000.0, 0.0, 0.0])
    with pytest.raises(validation.ValidationError):
        validation.check_velocity_reasonable(v)


# ============================================================================
# check_angular_velocity_reasonable tests
# ============================================================================

def test_check_angular_velocity_reasonable():
    omega = np.array([0.1, 0.1, 0.1])
    assert validation.check_angular_velocity_reasonable(omega)

def test_check_angular_velocity_exceeds_limit():
    omega = np.array([15.0, 0.0, 0.0])  # > 10 rad/s limit
    with pytest.raises(validation.ValidationError):
        validation.check_angular_velocity_reasonable(omega)


# ============================================================================
# validate_state tests
# ============================================================================

def test_validate_state_valid():
    state = State(
        r=np.array([C.R_EARTH + 1000.0, 0.0, 0.0]),
        v=np.array([0.0, 500.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.0, 0.0, 0.0]),
        m=C.INITIAL_MASS,
        t=0.0
    )
    valid, error = validation.validate_state(state, abort_on_error=False)
    assert valid is True
    assert error is None

def test_validate_state_invalid_quaternion():
    state = State(
        r=np.array([C.R_EARTH + 1000.0, 0.0, 0.0]),
        v=np.array([0.0, 500.0, 0.0]),
        q=np.array([2.0, 0.0, 0.0, 0.0]),  # Invalid norm
        omega=np.array([0.0, 0.0, 0.0]),
        m=C.INITIAL_MASS,
        t=0.0
    )
    valid, error = validation.validate_state(state, abort_on_error=False)
    assert valid is False
    assert error is not None
    assert "Quaternion" in error

def test_validate_state_abort_on_error():
    state = State(
        r=np.array([1e6, 0.0, 0.0]),  # Inside Earth
        v=np.array([0.0, 500.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.0, 0.0, 0.0]),
        m=C.INITIAL_MASS,
        t=0.0
    )
    with pytest.raises(validation.ValidationError):
        validation.validate_state(state, abort_on_error=True)


# ============================================================================
# compute_orbital_energy tests
# ============================================================================

def test_compute_orbital_energy():
    r = np.array([C.R_EARTH + 200000.0, 0.0, 0.0])
    v = np.array([0.0, 7800.0, 0.0])
    m = 1000.0
    E = validation.compute_orbital_energy(r, v, m)
    # Specific energy should be negative for bound orbit
    assert E < 0


# ============================================================================
# compute_total_energy tests
# ============================================================================

def test_compute_total_energy():
    r = np.array([C.R_EARTH + 200000.0, 0.0, 0.0])
    v = np.array([0.0, 7800.0, 0.0])
    m = 1000.0
    E = validation.compute_total_energy(r, v, m)
    # Total energy should be negative for bound orbit
    assert E < 0

def test_compute_total_energy_zero_radius():
    r = np.array([0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    m = 1000.0
    E = validation.compute_total_energy(r, v, m)
    assert E == 0.0

def test_compute_total_energy_negative_mass():
    r = np.array([C.R_EARTH, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])
    m = -100.0
    E = validation.compute_total_energy(r, v, m)
    assert E == 0.0


# ============================================================================
# validate_energy_conservation (second overload) tests
# ============================================================================

def test_validate_energy_conservation_valid():
    E_prev = -1e9
    E_curr = E_prev * 1.0001  # Small change
    result = validation.validate_energy_conservation(E_curr, E_prev, dt=0.1)
    assert result['valid'] is True

def test_validate_energy_conservation_violation():
    E_prev = -1e9
    E_curr = E_prev * 1.1  # 10% change - too much
    result = validation.validate_energy_conservation(E_curr, E_prev, dt=0.1)
    assert result['valid'] is False
    assert 'relative_error' in result

def test_validate_energy_conservation_zero_previous():
    E_prev = 0.0
    E_curr = 1000.0
    result = validation.validate_energy_conservation(E_curr, E_prev, dt=0.1)
    assert result['valid'] is True
    assert 'Energy too small' in result['message']


# ============================================================================
# validate_constraints tests
# ============================================================================

def test_validate_constraints_valid():
    r = np.array([C.R_EARTH + 100000.0, 0.0, 0.0])
    v = np.array([0.0, 2000.0, 0.0])
    omega = np.array([0.01, 0.01, 0.01])
    m = C.INITIAL_MASS
    result = validation.validate_constraints(r, v, omega, m)
    assert result['valid'] is True
    assert len(result['violations']) == 0

def test_validate_constraints_crash():
    r = np.array([C.R_EARTH - 2000.0, 0.0, 0.0])  # Below surface
    v = np.array([0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    m = C.INITIAL_MASS
    result = validation.validate_constraints(r, v, omega, m)
    assert result['valid'] is False
    assert any('CRASH' in v for v in result['violations'])

def test_validate_constraints_over_velocity():
    r = np.array([C.R_EARTH + 100000.0, 0.0, 0.0])
    v_escape = np.sqrt(2 * C.MU_EARTH / np.linalg.norm(r))
    v = np.array([v_escape * 1.2, 0.0, 0.0])  # 20% over escape velocity
    omega = np.array([0.0, 0.0, 0.0])
    m = C.INITIAL_MASS
    result = validation.validate_constraints(r, v, omega, m)
    assert result['valid'] is False
    assert any('velocity' in v.lower() for v in result['violations'])

def test_validate_constraints_high_spin():
    r = np.array([C.R_EARTH + 100000.0, 0.0, 0.0])
    v = np.array([0.0, 2000.0, 0.0])
    omega = np.array([np.radians(200), 0.0, 0.0])  # 200 deg/s > 180 deg/s limit
    m = C.INITIAL_MASS
    result = validation.validate_constraints(r, v, omega, m)
    assert result['valid'] is False
    assert any('Spin' in v for v in result['violations'])

def test_validate_constraints_mass_violation():
    r = np.array([C.R_EARTH + 100000.0, 0.0, 0.0])
    v = np.array([0.0, 2000.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    m = C.DRY_MASS * 0.5  # Below dry mass
    result = validation.validate_constraints(r, v, omega, m)
    assert result['valid'] is False
    assert any('Mass' in v for v in result['violations'])


# ============================================================================
# run_validation_suite tests
# ============================================================================

def test_run_validation_suite_all_pass():
    state = State(
        r=np.array([C.R_EARTH + 1000.0, 0.0, 0.0]),
        v=np.array([0.0, 500.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.0, 0.0, 0.0]),
        m=C.INITIAL_MASS,
        t=0.0
    )
    results = validation.run_validation_suite(state, verbose=False)
    assert results['all_passed'] is True
    assert results['quaternion_norm'] == 'PASS'
    assert results['position_valid'] == 'PASS'
    assert results['mass_valid'] == 'PASS'

def test_run_validation_suite_verbose(capsys):
    state = State(
        r=np.array([C.R_EARTH + 1000.0, 0.0, 0.0]),
        v=np.array([0.0, 500.0, 0.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.0, 0.0, 0.0]),
        m=C.INITIAL_MASS,
        t=0.0
    )
    results = validation.run_validation_suite(state, verbose=True)
    captured = capsys.readouterr()
    assert '✓' in captured.out  # Check marks printed

def test_run_validation_suite_with_failures(capsys):
    state = State(
        r=np.array([C.R_EARTH + 1000.0, 0.0, 0.0]),
        v=np.array([20000.0, 0.0, 0.0]),  # Exceeds velocity limit
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.0, 0.0, 0.0]),
        m=C.INITIAL_MASS,
        t=0.0
    )
    results = validation.run_validation_suite(state, verbose=True)
    assert results['all_passed'] is False
    assert 'FAIL' in results['velocity_reasonable']
    captured = capsys.readouterr()
    assert '✗' in captured.out  # X marks printed for failures
