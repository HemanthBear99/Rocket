"""
Tests for plot generation functionality.

Tests that plots are created correctly and files are generated.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from rlv_sim.main import SimulationLog


class MockLog:
    """Mock simulation log for testing plots without running simulation."""
    
    def __init__(self, n_points: int = 100):
        t = np.linspace(0, 100, n_points)
        
        self.time = list(t)
        self.altitude = list(np.linspace(0, 100, n_points))  # km
        self.velocity = list(np.linspace(465, 3000, n_points))  # m/s
        self.mass = list(np.linspace(540000, 200000, n_points))  # kg
        self.pitch_angle = list(np.linspace(0, 45, n_points))  # degrees
        self.attitude_error = list(np.abs(np.sin(t/10)) * 0.5)  # degrees
        self.torque_magnitude = list(np.abs(np.sin(t/5)) * 1e6)  # N*m
        
        # Position in ECI (simple radial trajectory)
        r_start = 6.371e6
        self.position_x = list(np.linspace(r_start, r_start + 100e3, n_points))
        self.position_y = list(np.linspace(0, 50e3, n_points))
        self.position_z = list(np.linspace(0, 30e3, n_points))
        
        self.quaternion_norm = list(np.ones(n_points))


class TestPlotGeneration(unittest.TestCase):
    """Tests for plot generation."""
    
    def setUp(self):
        """Create mock log and temp directory."""
        self.mock_log = MockLog()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_all_plots_creates_files(self):
        """generate_all_plots should create plot files."""
        from generate_plots import generate_all_plots
        
        # Create mock final state
        mock_state = MagicMock()
        mock_state.altitude = 100000
        mock_state.speed = 3000
        mock_state.m = 200000
        
        # Generate plots
        saved = generate_all_plots(self.mock_log, mock_state, self.temp_dir)
        
        # Check files were created
        self.assertGreater(len(saved), 0)
        for path in saved:
            self.assertTrue(os.path.exists(path), f"Plot file not found: {path}")
    
    def test_plot_directory_created(self):
        """Plot directory should be created if it doesn't exist."""
        from generate_plots import generate_all_plots
        
        mock_state = MagicMock()
        new_dir = os.path.join(self.temp_dir, 'new_subdir')
        
        generate_all_plots(self.mock_log, mock_state, new_dir)
        
        self.assertTrue(os.path.isdir(new_dir))


class TestMockLog(unittest.TestCase):
    """Tests for the mock log class itself."""
    
    def test_mock_log_has_required_attributes(self):
        """MockLog should have all required attributes."""
        log = MockLog()
        
        required = [
            'time', 'altitude', 'velocity', 'mass', 'pitch_angle',
            'attitude_error', 'torque_magnitude', 'position_x',
            'position_y', 'position_z', 'quaternion_norm'
        ]
        
        for attr in required:
            self.assertTrue(hasattr(log, attr), f"Missing attribute: {attr}")
            self.assertGreater(len(getattr(log, attr)), 0)


if __name__ == '__main__':
    unittest.main()
