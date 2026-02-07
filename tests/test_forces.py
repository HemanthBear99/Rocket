import pytest
import numpy as np
from rlv_sim import forces, constants as C

def test_compute_atmosphere_properties_troposphere():
    T, P, rho, a = forces.compute_atmosphere_properties(1000.0)
    assert T > 0
    assert P > 0
    assert rho > 0
    assert a > 0

def test_compute_atmosphere_properties_stratosphere():
    T, P, rho, a = forces.compute_atmosphere_properties(20000.0)
    assert T > 0
    assert P > 0
    assert rho > 0
    assert a > 0

def test_compute_gravity_force():
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    m = 1000.0
    F = forces.compute_gravity_force(r, m)
    assert F.shape == (3,)
    assert F[0] < 0
    assert np.allclose(F[1:], 0.0)

def test_compute_drag_force_zero_velocity():
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    v = np.zeros(3)
    F = forces.compute_drag_force(r, v)
    assert np.allclose(F, 0.0)

def test_compute_lift_force_zero_velocity():
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    v = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    F = forces.compute_lift_force(r, v, q)
    assert np.allclose(F, 0.0)

def test_compute_thrust_force_off():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    F = forces.compute_thrust_force(q, r, thrust_on=False)
    assert np.allclose(F, 0.0)

def test_compute_thrust_force_on():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    F = forces.compute_thrust_force(q, r, thrust_on=True)
    assert F.shape == (3,)
    assert F[2] > 0

def test_compute_aerodynamic_moment_zero_velocity():
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    v = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    cg_pos_z = 20.0
    M = forces.compute_aerodynamic_moment(r, v, q, cg_pos_z)
    assert np.allclose(M, 0.0)

def test_centrifugal_correction():
    """Centrifugal correction is available for ECEF analysis (not used in ECI dynamics)."""
    r = np.array([C.R_EARTH, 0.0, 0.0])
    m = 1000.0
    F = forces.compute_centrifugal_correction(r, m)
    assert F.shape == (3,)
    assert np.issubdtype(F.dtype, np.floating)
    # Centrifugal force should point radially outward (along +X here)
    assert F[0] > 0


def test_compute_total_force_shapes():
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    v = np.array([0.0, 100.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    m = 1000.0
    F = forces.compute_total_force(r, v, q, m, thrust_on=True)
    assert F.shape == (3,)


def test_compute_total_force_no_coriolis_in_eci():
    """ECI frame should NOT include Coriolis (fictitious force only in rotating frames)."""
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    v = np.array([0.0, 100.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    m = 1000.0
    F_total = forces.compute_total_force(r, v, q, m, thrust_on=True)
    # Total should equal gravity + thrust + drag + lift (no Coriolis)
    F_grav = forces.compute_gravity_force(r, m)
    F_thrust = forces.compute_thrust_force(q, r, thrust_on=True)
    F_drag = forces.compute_drag_force(r, v)
    F_lift = forces.compute_lift_force(r, v, q)
    expected = F_grav + F_thrust + F_drag + F_lift
    np.testing.assert_allclose(F_total, expected, atol=1e-6)


def test_compute_specific_forces_keys():
    r = np.array([C.R_EARTH + 1000.0, 0.0, 0.0])
    v = np.array([0.0, 100.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    m = 1000.0
    out = forces.compute_specific_forces(r, v, q, m, thrust_on=True)
    for k in ['gravity', 'thrust', 'drag', 'lift', 'total',
              'gravity_magnitude', 'thrust_magnitude', 'drag_magnitude', 'lift_magnitude']:
        assert k in out
