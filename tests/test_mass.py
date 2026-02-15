import pytest
import numpy as np
from rlv_sim import mass, constants as C

def test_compute_mass_flow_rate_on():
    rate = mass.compute_mass_flow_rate(thrust_on=True, throttle=1.0)
    assert rate < 0
    assert rate == pytest.approx(-C.MASS_FLOW_RATE)

def test_compute_mass_flow_rate_off():
    rate = mass.compute_mass_flow_rate(thrust_on=False)
    assert rate == 0.0
    rate = mass.compute_mass_flow_rate(thrust_on=True, throttle=0.0)
    assert rate == 0.0

def test_compute_mass_derivative_exhausted():
    m = C.DRY_MASS
    dm = mass.compute_mass_derivative(m, thrust_on=True, throttle=1.0)
    assert dm == 0.0

def test_compute_mass_derivative_on():
    m = C.DRY_MASS + 100.0
    dm = mass.compute_mass_derivative(m, thrust_on=True, throttle=1.0)
    assert dm < 0
    assert dm == pytest.approx(-C.MASS_FLOW_RATE)

def test_update_mass_stops_at_dry():
    m = C.DRY_MASS + 10.0
    dt = 1000.0
    m2 = mass.update_mass(m, dt, thrust_on=True, throttle=1.0)
    assert m2 >= C.DRY_MASS

def test_is_propellant_exhausted():
    assert mass.is_propellant_exhausted(C.DRY_MASS)
    assert not mass.is_propellant_exhausted(C.DRY_MASS + 1.0)

def test_get_propellant_fraction():
    m = C.DRY_MASS + C.PROPELLANT_MASS
    assert mass.get_propellant_fraction(m) == pytest.approx(1.0)
    assert mass.get_propellant_fraction(C.DRY_MASS) == 0.0

def test_compute_center_of_mass_bounds():
    cg_full = mass.compute_center_of_mass(C.DRY_MASS + C.PROPELLANT_MASS)
    cg_empty = mass.compute_center_of_mass(C.DRY_MASS)
    assert C.H_CG_FULL - 1e-6 <= cg_full <= C.H_CG_FULL + 1e-6
    assert C.H_CG_EMPTY - 1e-6 <= cg_empty <= C.H_CG_EMPTY + 1e-6

def test_compute_inertia_tensor_shape():
    I = mass.compute_inertia_tensor(C.DRY_MASS + C.PROPELLANT_MASS)
    assert I.shape == (3, 3)
    assert np.allclose(I, I.T)
    I2 = mass.compute_inertia_tensor(C.DRY_MASS)
    assert I2.shape == (3, 3)
