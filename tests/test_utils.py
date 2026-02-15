import pytest
import numpy as np
from rlv_sim import utils

def test_compute_relative_velocity_zero():
    r = np.array([1.0, 0.0, 0.0])
    v = np.zeros(3)
    v_rel = utils.compute_relative_velocity(r, v)
    assert v_rel.shape == (3,)
    assert np.issubdtype(v_rel.dtype, np.floating)

def test_compute_relative_velocity_magnitude_zero():
    r = np.array([1.0, 0.0, 0.0])
    v = np.zeros(3)
    mag = utils.compute_relative_velocity_magnitude(r, v)
    assert mag >= 0.0
