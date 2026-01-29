"""
Engine gimbal actuator dynamics (second-order rate-limited slewing).
"""

from dataclasses import dataclass
import numpy as np

from . import constants as C


@dataclass
class ActuatorState:
    """Tracks current thrust direction for gimbal dynamics."""
    thrust_dir: np.ndarray = None

    def __post_init__(self):
        if self.thrust_dir is None:
            self.thrust_dir = np.array([1.0, 0.0, 0.0], dtype=float)  # radial up at launch
        else:
            self.thrust_dir = np.asarray(self.thrust_dir, dtype=float)


def _limit_rotation(current: np.ndarray, desired: np.ndarray, max_rate: float, dt: float) -> np.ndarray:
    """Limit change in direction to respect max gimbal rate (rad/s)."""
    cur_n = current / (np.linalg.norm(current) + 1e-12)
    des_n = desired / (np.linalg.norm(desired) + 1e-12)
    dot = np.clip(np.dot(cur_n, des_n), -1.0, 1.0)
    angle = np.arccos(dot)
    if angle <= max_rate * dt:
        return des_n
    # rotate cur_n toward des_n by capped angle
    axis = np.cross(cur_n, des_n)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return cur_n  # parallel or anti-parallel; nothing to do
    axis /= axis_norm
    theta = max_rate * dt
    # Rodrigues rotation
    return (
        cur_n * np.cos(theta)
        + np.cross(axis, cur_n) * np.sin(theta)
        + axis * np.dot(axis, cur_n) * (1 - np.cos(theta))
    )


def update_actuator(state: ActuatorState, desired_dir: np.ndarray, dt: float) -> ActuatorState:
    """
    Advance gimbal actuator toward desired thrust direction with rate limit.
    """
    limited = _limit_rotation(state.thrust_dir, desired_dir, C.MAX_GIMBAL_RATE, dt)
    return ActuatorState(thrust_dir=limited)
