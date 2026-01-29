"""
RLV Phase-I Ascent Simulation - Attitude Control

This module implements the attitude control system:
- Direction to quaternion conversion
- Quaternion error computation
- PD control law
- Torque saturation
"""

import numpy as np

from . import constants as C
from .types import ControlOutput
from .frames import (
    direction_to_quaternion,
    quaternion_error,
    quaternion_to_axis_angle,
    quaternion_normalize
)


def compute_commanded_quaternion(desired_direction: np.ndarray) -> np.ndarray:
    """
    Convert desired thrust direction to commanded orientation quaternion.
    
    The body +Z axis should align with the desired thrust direction.
    
    Args:
        desired_direction: Desired thrust direction (inertial frame)
        
    Returns:
        Commanded quaternion [w, x, y, z]
    """
    # Body +Z must point along desired direction
    # direction_to_quaternion rotates BODY_Z to the target direction
    return direction_to_quaternion(desired_direction, C.BODY_Z_AXIS)


def compute_attitude_error(q_current: np.ndarray, 
                          q_commanded: np.ndarray) -> tuple:
    """
    Compute the attitude error between current and commanded orientation.
    
    Returns both the error quaternion and the error as an axis-angle
    representation for the control law.
    
    Args:
        q_current: Current orientation quaternion
        q_commanded: Commanded orientation quaternion
        
    Returns:
        (error_axis, error_angle) in body frame
    """
    q_current = quaternion_normalize(q_current)
    q_commanded = quaternion_normalize(q_commanded)
    
    q_err = quaternion_error(q_current, q_commanded)
    axis, angle = quaternion_to_axis_angle(q_err)
    
    return axis, angle


def saturate_torque(torque: np.ndarray) -> np.ndarray:
    """
    Apply torque saturation limits.
    
    Args:
        torque: Commanded torque vector (N*m)
        
    Returns:
        Saturated torque vector (N*m)
    """
    torque_magnitude = np.linalg.norm(torque)
    
    if torque_magnitude > C.MAX_TORQUE:
        return torque * (C.MAX_TORQUE / torque_magnitude)
    
    return torque


def limit_gimbal_rate(omega: np.ndarray) -> np.ndarray:
    """
    Apply gimbal rate limits to angular velocity.
    
    Constrains angular velocity components to MAX_GIMBAL_RATE to respect
    physical engine gimbal actuator capabilities.
    
    Args:
        omega: Current angular velocity in body frame (rad/s)
        
    Returns:
        Rate-limited angular velocity in body frame (rad/s)
    """
    # Limit each component to MAX_GIMBAL_RATE
    # For a rocket, typically limit pitch and roll, while yaw is less critical
    omega_limited = np.array([
        np.clip(omega[0], -C.MAX_GIMBAL_RATE, C.MAX_GIMBAL_RATE),  # Roll
        np.clip(omega[1], -C.MAX_GIMBAL_RATE, C.MAX_GIMBAL_RATE),  # Pitch
        np.clip(omega[2], -C.MAX_GIMBAL_RATE, C.MAX_GIMBAL_RATE)   # Yaw
    ])
    return omega_limited


def pd_control_law(error_axis: np.ndarray, error_angle: float,
                   omega: np.ndarray) -> np.ndarray:
    """
    Implement PD attitude control law with gimbal rate limiting.
    
    τ = Kp * θ * axis - Kd * ω_limited
    
    where θ is the error angle, axis is the rotation axis, and ω_limited
    respects physical gimbal rate constraints.
    
    Args:
        error_axis: Unit axis of rotation error
        error_angle: Magnitude of rotation error (rad)
        omega: Current angular velocity in body frame (rad/s)
        
    Returns:
        Control torque in body frame (N*m)
    """
    # Apply gimbal rate limits to angular velocity [FIX #4]
    omega_limited = limit_gimbal_rate(omega)
    
    # Proportional term (Kp * θ * axis)
    tau_p = C.KP_ATTITUDE * (error_angle * error_axis)
    
    # Derivative term (-Kd * ω_limited)
    tau_d = -C.KD_ATTITUDE * omega_limited
    
    return saturate_torque(tau_p + tau_d)


def compute_control_torque(q_current: np.ndarray, omega: np.ndarray,
                          desired_direction: np.ndarray) -> np.ndarray:
    """
    Compute the control torque to track desired thrust direction.
    
    This is the main control function that:
    1. Converts desired direction to commanded quaternion
    2. Computes attitude error
    3. Applies PD control law
    4. Saturates torque
    
    Args:
        q_current: Current orientation quaternion [w, x, y, z]
        omega: Current angular velocity in body frame (rad/s)
        desired_direction: Desired thrust direction (inertial frame)
        
    Returns:
        Control torque in body frame (N*m)
    """
    # Step 1: Get commanded quaternion
    q_commanded = compute_commanded_quaternion(desired_direction)
    
    # Step 2: Compute error and apply PD control
    error_axis, error_angle = compute_attitude_error(q_current, q_commanded)
    return pd_control_law(error_axis, error_angle, omega)


def compute_control_output(q_current: np.ndarray, omega: np.ndarray,
                          desired_direction: np.ndarray) -> ControlOutput:
    """
    Compute full control output for logging and analysis.
    
    Args:
        q_current: Current orientation quaternion [w, x, y, z]
        omega: Current angular velocity in body frame (rad/s)
        desired_direction: Desired thrust direction (inertial frame)
        
    Returns:
        Dictionary containing control state and commands
    """
    q_commanded = compute_commanded_quaternion(desired_direction)
    error_axis, error_angle = compute_attitude_error(q_current, q_commanded)
    torque = pd_control_law(error_axis, error_angle, omega)
    
    return {
        'q_commanded': q_commanded,
        'error_axis': error_axis,
        'error_angle': error_angle,
        'error_degrees': np.degrees(error_angle),
        'torque': torque,
        'torque_magnitude': np.linalg.norm(torque),
        'saturated': np.linalg.norm(torque) >= C.MAX_TORQUE * 0.999  # 99.9% of max
    }
