"""
Unit tests for plot generation functionality.

Tests that plots are created correctly and files are generated
using the clean plot_generator module.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rlv_sim.plotting import (
    generate_all_plots,
    extract_log_data,
    compute_gravity_turn_start,
    TrajectoryData,
)


class MockLog:
    """Mock simulation log for testing plots without running full simulation.
    
    Provides realistic trajectory data for unit testing.
    """
    
    def __init__(self, n_points: int = 100):
        """Initialize mock log with synthetic trajectory data.
        
        Args:
            n_points: Number of time points in the trajectory
        """
        t = np.linspace(0, 100, n_points)
        
        self.time = list(t)
        self.altitude = list(np.linspace(0, 100, n_points))  # km
        self.velocity = list(np.linspace(465, 3000, n_points))  # m/s
        self.mass = list(np.linspace(540000, 200000, n_points))  # kg
        self.pitch_angle = list(np.linspace(0, 45, n_points))  # deg from vertical
        self.actual_pitch_angle = list(np.linspace(0, 40, n_points))  # deg
        self.attitude_error = list(np.abs(np.sin(t/10)) * 0.5)  # deg
        self.torque_magnitude = list(np.abs(np.sin(t/5)) * 1e6)  # N·m
        self.gamma_command_deg = list(np.linspace(90, 10, n_points))
        self.gamma_actual_deg = list(np.linspace(90, 15, n_points))
        
        # Position in ECI (simple radial trajectory)
        r_start = 6.371e6
        self.position_x = list(np.linspace(r_start, r_start + 100e3, n_points))
        self.position_y = list(np.linspace(0, 50e3, n_points))
        self.position_z = list(np.linspace(0, 30e3, n_points))
        
        # Velocity components in ECI
        self.velocity_x = list(np.linspace(0, 100, n_points))
        self.velocity_y = list(np.linspace(465, 600, n_points))
        self.velocity_z = list(np.linspace(0, 2800, n_points))

        # Inertial thrust vector (N) — simulates thrust decaying as pitch increases
        thrust_mag = np.where(np.linspace(0, 100, n_points) < 80,
                              7.6e6, 0.0)  # Thrust on for 80% of flight
        pitch_rad = np.radians(self.pitch_angle)
        self.inertial_thrust_x = list(thrust_mag * np.cos(pitch_rad))
        self.inertial_thrust_y = list(np.zeros(n_points))
        self.inertial_thrust_z = list(thrust_mag * np.sin(pitch_rad))


class TestPlotGeneration(unittest.TestCase):
    """Test suite for plot generation functionality."""
    
    def setUp(self):
        """Create mock log and temporary directory for test outputs."""
        self.mock_log = MockLog()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_all_plots_creates_files(self):
        """Test that generate_all_plots creates all expected plot files."""
        saved = generate_all_plots(self.mock_log, self.temp_dir)
        
        # Check that files were created
        self.assertGreater(len(saved), 0, "No plot files were generated")
        
        # Verify each file exists
        for path in saved:
            self.assertTrue(
                os.path.exists(path),
                f"Plot file not found: {path}"
            )
    
    def test_generate_all_plots_returns_valid_paths(self):
        """Test that generated plot paths are valid."""
        saved = generate_all_plots(self.mock_log, self.temp_dir)
        
        for path in saved:
            # Check path is string
            self.assertIsInstance(path, str)
            # Check path is absolute or relative to output dir
            self.assertTrue(
                path.startswith(self.temp_dir) or os.path.isabs(path),
                f"Path {path} is not in expected directory"
            )
    
    def test_output_directory_created(self):
        """Test that output directory is created if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, 'new_subdir', 'nested')
        
        saved = generate_all_plots(self.mock_log, new_dir)
        
        self.assertTrue(os.path.isdir(new_dir), "Output directory was not created")
        self.assertGreater(len(saved), 0, "No plots generated in new directory")


class TestDataExtraction(unittest.TestCase):
    """Test suite for data extraction and processing functions."""
    
    def setUp(self):
        """Create mock log for testing."""
        self.mock_log = MockLog(n_points=50)
    
    def test_extract_log_data_returns_trajectory_data(self):
        """Test that extract_log_data returns valid TrajectoryData."""
        data = extract_log_data(self.mock_log)
        
        self.assertIsInstance(data, TrajectoryData)
        
        # Check all arrays have correct length
        self.assertEqual(len(data.time), 50)
        self.assertEqual(len(data.altitude), 50)
        self.assertEqual(len(data.velocity), 50)
        self.assertEqual(len(data.position), 50)
    
    def test_extract_log_data_arrays_have_correct_shape(self):
        """Test that extracted arrays have expected shapes."""
        data = extract_log_data(self.mock_log)
        
        # Position and velocity vectors should be 2D
        self.assertEqual(data.position.shape, (50, 3))
        self.assertEqual(data.velocity_vec.shape, (50, 3))
        self.assertEqual(data.velocity_rel_vec.shape, (50, 3))
        
        # Scalar quantities should be 1D
        self.assertEqual(data.altitude.shape, (50,))
        self.assertEqual(data.velocity.shape, (50,))
    
    def test_extract_log_data_physical_constraints(self):
        """Test that extracted data satisfies physical constraints."""
        data = extract_log_data(self.mock_log)
        
        # Velocities should be non-negative
        self.assertTrue(np.all(data.velocity >= 0))
        self.assertTrue(np.all(data.velocity_rel >= 0))
        
        # Altitude should be non-negative
        self.assertTrue(np.all(data.altitude >= 0))
        
        # Mass should decrease over time (fuel consumption)
        self.assertGreater(data.mass[0], data.mass[-1])


class TestGravityTurnDetection(unittest.TestCase):
    """Test suite for gravity turn detection."""
    
    def test_compute_gravity_turn_start_with_constant_pitch(self):
        """Test gravity turn detection with constant pitch (no turn)."""
        # Create data with constant pitch (no change exceeds threshold)
        data = TrajectoryData(
            time=np.array([0, 10, 20, 30]),
            altitude=np.array([0, 10, 20, 30]),
            velocity=np.array([100, 200, 300, 400]),
            velocity_rel=np.array([100, 200, 300, 400]),
            mass=np.array([500000, 400000, 300000, 200000]),
            pitch_angle=np.array([0, 0, 0, 0]),  # Constant
            actual_pitch=np.array([0, 0, 0, 0]),
            attitude_error=np.array([0, 0, 0, 0]),
            torque=np.array([0, 0, 0, 0]),
            gamma_cmd=np.array([90, 90, 90, 90]),
            gamma_actual=np.array([90, 90, 90, 90]),
            gamma_rel=np.array([90, 90, 90, 90]),
            position=np.zeros((4, 3)),
            velocity_vec=np.zeros((4, 3)),
            velocity_rel_vec=np.zeros((4, 3)),
            downrange=np.array([0, 10, 20, 30]),
            dynamic_pressure=np.array([0, 10, 20, 30]),
            thrust_force=np.array([7.6e6, 7.6e6, 7.6e6, 0.0]),
        )

        turn_time = compute_gravity_turn_start(data)
        # With constant pitch, no pitch rate exceeds threshold, so returns
        # time at index min(20, n-1) = min(20, 3) = 3, which is time=30.0
        self.assertEqual(turn_time, 30.0)
    
    def test_compute_gravity_turn_start_with_pitch_change(self):
        """Test gravity turn detection with clear pitch change."""
        data = TrajectoryData(
            time=np.array([0, 10, 20, 30, 40]),
            altitude=np.array([0, 10, 20, 30, 40]),
            velocity=np.array([100, 200, 300, 400, 500]),
            velocity_rel=np.array([100, 200, 300, 400, 500]),
            mass=np.array([500000, 450000, 400000, 350000, 300000]),
            # Pitch changes: 0->0.05 (rate 0.05), 0.05->5.0 (rate 4.95 > 0.1 threshold)
            # The large change happens between indices 1 and 2, so detection at index 1
            pitch_angle=np.array([0, 0.05, 5.0, 10.0, 15.0]),
            actual_pitch=np.array([0, 0, 5, 10, 15]),
            attitude_error=np.array([0, 0, 0, 0, 0]),
            torque=np.array([0, 0, 0, 0, 0]),
            gamma_cmd=np.array([90, 90, 80, 70, 60]),
            gamma_actual=np.array([90, 90, 80, 70, 60]),
            gamma_rel=np.array([90, 90, 80, 70, 60]),
            position=np.zeros((5, 3)),
            velocity_vec=np.zeros((5, 3)),
            velocity_rel_vec=np.zeros((5, 3)),
            downrange=np.array([0, 5, 15, 30, 50]),
            dynamic_pressure=np.array([0, 10, 30, 50, 70]),
            thrust_force=np.array([7.6e6, 7.6e6, 7.6e6, 7.6e6, 0.0]),
        )
        
        turn_time = compute_gravity_turn_start(data)
        # Should detect turn at index 1 (time = 10) where rate first exceeds 0.1
        self.assertEqual(turn_time, 10.0)


class TestMockLog(unittest.TestCase):
    """Test suite for the MockLog class itself."""
    
    def test_mock_log_has_required_attributes(self):
        """Test that MockLog has all required attributes."""
        log = MockLog()
        
        required = [
            'time', 'altitude', 'velocity', 'mass', 'pitch_angle',
            'attitude_error', 'torque_magnitude', 'position_x',
            'position_y', 'position_z', 'velocity_x', 'velocity_y', 'velocity_z'
        ]
        
        for attr in required:
            self.assertTrue(hasattr(log, attr), f"Missing attribute: {attr}")
            self.assertGreater(len(getattr(log, attr)), 0)
    
    def test_mock_log_consistent_length(self):
        """Test that all arrays in MockLog have consistent length."""
        log = MockLog(n_points=50)
        
        lengths = [
            len(log.time),
            len(log.altitude),
            len(log.velocity),
            len(log.mass),
            len(log.pitch_angle),
        ]
        
        self.assertEqual(len(set(lengths)), 1, "Inconsistent array lengths in MockLog")


if __name__ == '__main__':
    unittest.main()
