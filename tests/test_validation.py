import pytest
import numpy as np
from rlv_sim import validation

def test_check_quaternion_norm_valid():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert validation.check_quaternion_norm(q)

def test_check_quaternion_norm_invalid():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    with pytest.raises(validation.ValidationError):
        validation.check_quaternion_norm(q)

def test_check_position_valid():
    r = np.array([7e6, 0.0, 0.0])
    assert validation.check_position_valid(r)
    r = np.array([1e6, 0.0, 0.0])
    with pytest.raises(validation.ValidationError):
        validation.check_position_valid(r)
