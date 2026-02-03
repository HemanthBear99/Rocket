import pytest
import numpy as np
import rlv_sim.constants as C

def test_earth_parameters():
    assert C.MU_EARTH > 0
    assert C.R_EARTH > 6e6
    assert C.G0 > 9.0

def test_vehicle_masses():
    assert C.DRY_MASS > 0
    assert C.PROPELLANT_MASS > 0
    assert C.INITIAL_MASS > 0
    assert C.DRY_MASS < C.INITIAL_MASS

def test_thrust_and_isp():
    assert C.THRUST_MAGNITUDE > 0
    assert C.ISP > 0
    assert C.MASS_FLOW_RATE > 0

def test_reference_area():
    assert C.REFERENCE_AREA > 0
    assert C.REFERENCE_DIAMETER > 0

def test_inertia_tensors():
    assert np.all(np.diag(C.INERTIA_TENSOR_FULL) > 0)
    assert np.all(np.diag(C.INERTIA_TENSOR_EMPTY) > 0)
    assert C.IXX_FULL > C.IXX_EMPTY
    assert C.IZZ_FULL > C.IZZ_EMPTY

def test_simulation_parameters():
    assert C.DT > 0
    assert C.MAX_TIME > 0
    assert C.TARGET_ALTITUDE > 0
    assert C.TARGET_SPEED > 0
