"""
RLV Phase-I Ascent Simulation - Configuration

This module provides a SimulationConfig dataclass for dependency injection,
allowing different simulation parameters to be passed without modifying
global constants.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from . import constants as C


@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable configuration for simulation parameters.
    
    Using frozen=True ensures configs cannot be accidentally modified.
    Create new configs via dataclass replace() if needed.
    
    Attributes:
        dt: Time step (s)
        max_time: Maximum simulation time (s)
        
        # Control gains
        kp_attitude: Proportional attitude control gain
        kd_attitude: Derivative attitude control gain
        max_torque: Maximum control torque (N·m)
        
        # Guidance parameters
        gravity_turn_start_altitude: Altitude to begin gravity turn (m)
        gravity_turn_transition_range: Range over which to transition (m)
        min_velocity_for_turn: Minimum velocity before turning (m/s)
        pitchover_start_altitude: Altitude to start pitchover (m)
        pitchover_end_altitude: Altitude to end pitchover (m)
        pitchover_angle: Pitchover kick angle (rad)
        
        # Physics tolerances
        quaternion_norm_tolerance: Allowable deviation from unit norm
        zero_tolerance: General tolerance for near-zero checks
    """
    
    # Simulation parameters
    dt: float = C.DT
    max_time: float = C.MAX_TIME
    
    # Control gains
    kp_attitude: float = C.KP_ATTITUDE
    kd_attitude: float = C.KD_ATTITUDE
    max_torque: float = C.MAX_TORQUE
    
    # Guidance parameters
    gravity_turn_start_altitude: float = C.GRAVITY_TURN_START_ALTITUDE
    gravity_turn_transition_range: float = C.GRAVITY_TURN_TRANSITION_RANGE
    min_velocity_for_turn: float = C.MIN_VELOCITY_FOR_TURN
    pitchover_start_altitude: float = C.PITCHOVER_START_ALTITUDE
    pitchover_end_altitude: float = C.PITCHOVER_END_ALTITUDE
    pitchover_angle: float = C.PITCHOVER_ANGLE
    
    # Physics tolerances
    quaternion_norm_tolerance: float = C.QUATERNION_NORM_TOL
    zero_tolerance: float = 1e-10
    
    # Verbose output
    verbose: bool = True


def create_default_config() -> SimulationConfig:
    """Create a SimulationConfig with default values from constants."""
    return SimulationConfig()


def create_test_config(dt: float = 0.1, max_time: float = 10.0) -> SimulationConfig:
    """Create a fast config suitable for testing."""
    return SimulationConfig(dt=dt, max_time=max_time, verbose=False)
