import pytest
import numpy as np
from rlv_sim import types

def test_guidance_output_typeddict():
    out = types.GuidanceOutput(
        thrust_direction=np.array([1.0, 0.0, 0.0]),
        phase="VERTICAL_ASCENT",
        thrust_on=True,
        pitch_angle=0.0,
        blend_alpha=0.0,
        altitude=0.0,
        velocity=0.0,
        local_vertical=np.array([1.0, 0.0, 0.0]),
        local_horizontal=np.array([0.0, 1.0, 0.0]),
        throttle=1.0
    )
    assert out["phase"] == "VERTICAL_ASCENT"
    assert out["thrust_on"] is True
    assert isinstance(out["thrust_direction"], np.ndarray)

def test_control_output_typeddict():
    out = types.ControlOutput(
        q_commanded=np.array([1.0, 0.0, 0.0, 0.0]),
        error_axis=np.array([0.0, 0.0, 1.0]),
        error_angle=0.0,
        error_degrees=0.0,
        torque=np.zeros(3),
        torque_magnitude=0.0,
        saturated=False
    )
    assert out["saturated"] is False
    assert isinstance(out["q_commanded"], np.ndarray)

def test_forcebreakdown_typeddict():
    out = types.ForceBreakdown(
        gravity=np.zeros(3),
        thrust=np.zeros(3),
        drag=np.zeros(3),
        lift=np.zeros(3),
        total=np.zeros(3),
        gravity_magnitude=0.0,
        thrust_magnitude=0.0,
        drag_magnitude=0.0,
        lift_magnitude=0.0
    )
    assert isinstance(out["gravity"], np.ndarray)
    assert out["drag_magnitude"] == 0.0
    assert out["lift_magnitude"] == 0.0
