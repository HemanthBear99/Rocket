"""
RLV Ascent Simulation - Navigation Sensor Models

IMU error model with bias, noise and quantization for
navigation sensor simulation.

Enabled via config.enable_nav_sensors = True.
"""

import numpy as np


class IMUState:
    """Inertial Measurement Unit state with error model."""

    def __init__(self, accel_bias_sigma: float = 0.001,
                 accel_noise_density: float = 0.01,
                 gyro_bias_sigma: float = 1e-5,
                 gyro_noise_density: float = 1e-4,
                 seed: int = None):
        """
        Args:
            accel_bias_sigma: Accelerometer bias 1-sigma (m/s^2)
            accel_noise_density: Accel noise spectral density (m/s^2/sqrt(Hz))
            gyro_bias_sigma: Gyro bias 1-sigma (rad/s)
            gyro_noise_density: Gyro noise spectral density (rad/s/sqrt(Hz))
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

        # Draw constant biases (fixed for lifetime of sensor)
        self.accel_bias = self.rng.normal(0, accel_bias_sigma, size=3)
        self.gyro_bias = self.rng.normal(0, gyro_bias_sigma, size=3)

        self.accel_noise_density = accel_noise_density
        self.gyro_noise_density = gyro_noise_density

    def measure_acceleration(self, true_accel: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply IMU error model to true specific acceleration.

        Args:
            true_accel: True specific acceleration in body frame (m/s^2)
            dt: Sampling interval (s)

        Returns:
            Measured acceleration with bias + noise (m/s^2)
        """
        noise_sigma = self.accel_noise_density / np.sqrt(max(dt, 1e-6))
        noise = self.rng.normal(0, noise_sigma, size=3)
        return true_accel + self.accel_bias + noise

    def measure_angular_rate(self, true_omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply IMU error model to true angular velocity.

        Args:
            true_omega: True angular rate in body frame (rad/s)
            dt: Sampling interval (s)

        Returns:
            Measured angular rate with bias + noise (rad/s)
        """
        noise_sigma = self.gyro_noise_density / np.sqrt(max(dt, 1e-6))
        noise = self.rng.normal(0, noise_sigma, size=3)
        return true_omega + self.gyro_bias + noise
