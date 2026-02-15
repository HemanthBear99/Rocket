"""
RLV Phase-I Ascent Simulation - Type Definitions

This module provides TypedDict definitions for structured return types,
improving type safety and IDE support.
"""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class GuidanceOutput(TypedDict):
    """Return type for guidance system output."""
    thrust_direction: NDArray[np.float64]  # Unit vector in inertial frame
    phase: str  # Current guidance phase: VERTICAL_ASCENT, PITCHOVER, GRAVITY_TURN, PROGRADE
    thrust_on: bool  # Whether thrust should be active
    pitch_angle: float  # Pitch angle from vertical (radians)
    blend_alpha: float  # Blend parameter for gravity turn (0-1)
    altitude: float  # Current altitude above Earth surface (m)
    velocity: float  # Current velocity magnitude (m/s)
    local_vertical: NDArray[np.float64]  # Unit vector pointing radially outward
    local_horizontal: NDArray[np.float64]  # Unit vector in velocity direction
    throttle: float  # Throttle command (0.0 to 1.0)
    gamma_angle: float  # Flight path angle from horizontal (radians)
    gamma_command_deg: float  # Commanded flight path angle (degrees)
    gamma_measured_deg: float  # Measured flight path angle (degrees)
    velocity_tilt_deg: float  # Velocity tilt from vertical (degrees)
    prograde: NDArray[np.float64]  # Prograde direction vector
    v_rel: NDArray[np.float64]  # Air-relative velocity vector
    v_rel_mag: float  # Air-relative velocity magnitude (m/s)


class ControlOutput(TypedDict):
    """Return type for control system output."""
    q_commanded: NDArray[np.float64]  # Commanded orientation quaternion [w, x, y, z]
    error_axis: NDArray[np.float64]  # Rotation axis for attitude error
    error_angle: float  # Attitude error magnitude (radians)
    error_degrees: float  # Attitude error magnitude (degrees)
    torque: NDArray[np.float64]  # Control torque in body frame (N·m)
    torque_magnitude: float  # Torque magnitude (N·m)
    saturated: bool  # Whether torque is at saturation limit


class ForceBreakdown(TypedDict):
    """Return type for force computation details (ECI inertial frame).

    Note: No Coriolis force — this is an inertial frame formulation.
    Earth rotation is accounted for via air-relative velocity in drag/lift.
    """
    gravity: NDArray[np.float64]  # Gravity force vector (N)
    thrust: NDArray[np.float64]  # Thrust force vector (N)
    drag: NDArray[np.float64]  # Drag force vector (N)
    lift: NDArray[np.float64]  # Lift force vector (N)
    total: NDArray[np.float64]  # Total force vector (N)
    gravity_magnitude: float  # Gravity force magnitude (N)
    thrust_magnitude: float  # Thrust force magnitude (N)
    drag_magnitude: float  # Drag force magnitude (N)
    lift_magnitude: float  # Lift force magnitude (N)


class AtmosphereProperties(TypedDict):
    """Return type for atmosphere model output."""
    temperature: float  # Temperature (K)
    pressure: float  # Pressure (Pa)
    density: float  # Density (kg/m³)
    speed_of_sound: float  # Speed of sound (m/s)
