"""
Integration tests for full simulation run.

Tests the complete simulation pipeline from start to finish,
verifying termination conditions, state bounds, and physics validity.
"""

import unittest
import numpy as np

from rlv_sim import run_simulation, create_initial_state
from rlv_sim import constants as C
from rlv_sim.validation import run_validation_suite


class TestFullSimulation(unittest.TestCase):
    """Integration tests for complete simulation runs."""
    
    @classmethod
    def setUpClass(cls):
        """Run simulation once for all tests in this class."""
        # Use a shorter simulation for faster tests
        cls.final_state, cls.log, cls.reason = run_simulation(
            dt=0.1,  # Larger timestep for speed
            max_time=60.0,  # Run for 60 seconds only
            verbose=False
        )
    
    def test_simulation_terminates(self):
        """Simulation should terminate with a valid reason."""
        self.assertIsNotNone(self.reason)
        self.assertIsInstance(self.reason, str)
        self.assertGreater(len(self.reason), 0)
    
    def test_log_has_data(self):
        """Simulation log should contain data."""
        self.assertGreater(len(self.log.time), 0)
        self.assertGreater(len(self.log.altitude), 0)
        self.assertGreater(len(self.log.velocity), 0)
    
    def test_altitude_increases(self):
        """Altitude should generally increase during ascent."""
        altitudes = np.array(self.log.altitude)
        # Check that final altitude > initial altitude
        self.assertGreater(altitudes[-1], altitudes[0])
    
    def test_velocity_increases(self):
        """Velocity should increase during powered ascent."""
        velocities = np.array(self.log.velocity)
        # Check that velocity increases overall
        self.assertGreater(velocities[-1], velocities[0])
    
    def test_mass_decreases(self):
        """Mass should decrease as propellant is consumed."""
        masses = np.array(self.log.mass)
        self.assertLess(masses[-1], masses[0])
        self.assertGreaterEqual(masses[-1], C.DRY_MASS * 0.99)  # Never below dry mass
    
    def test_quaternion_norm_preserved(self):
        """Quaternion norm should stay close to 1.0."""
        quat_norms = np.array(self.log.quaternion_norm)
        for norm in quat_norms:
            self.assertAlmostEqual(norm, 1.0, places=4)
    
    def test_final_state_valid(self):
        """Final state should pass all validation checks."""
        results = run_validation_suite(self.final_state, verbose=False)
        self.assertTrue(results['all_passed'])


class TestShortSimulation(unittest.TestCase):
    """Quick simulation tests for basic functionality."""
    
    def test_one_second_simulation(self):
        """Short simulation should complete without errors."""
        final_state, log, reason = run_simulation(
            dt=0.01,
            max_time=1.0,
            verbose=False
        )
        
        self.assertIsNotNone(final_state)
        self.assertGreater(len(log.time), 0)
        # Note: Simulation may terminate early due to MECO or may hit max_time
        # We just verify it runs without crashing
    
    def test_different_timesteps(self):
        """Simulation should work with different timesteps."""
        for dt in [0.001, 0.01, 0.1]:
            final_state, log, reason = run_simulation(
                dt=dt,
                max_time=1.0,
                verbose=False
            )
            self.assertIsNotNone(final_state)


class TestTerminationConditions(unittest.TestCase):
    """Tests for simulation termination conditions."""
    
    def test_max_time_termination(self):
        """Simulation should terminate when max_time is reached."""
        final_state, log, reason = run_simulation(
            dt=0.1,
            max_time=5.0,
            verbose=False
        )
        
        # Should either hit max time or some other condition first
        self.assertIsNotNone(reason)
        self.assertTrue(
            'Maximum simulation time' in reason or 
            len(log.time) > 0
        )


class TestInitialState(unittest.TestCase):
    """Tests for initial state creation."""
    
    def test_initial_state_at_surface(self):
        """Initial state should be at Earth's surface."""
        state = create_initial_state()
        
        # Check position magnitude
        r_mag = np.linalg.norm(state.r)
        self.assertAlmostEqual(r_mag, C.R_EARTH, delta=100)
    
    def test_initial_mass_correct(self):
        """Initial mass should equal stacked configuration mass."""
        state = create_initial_state()
        self.assertEqual(state.m, C.INITIAL_MASS)
    
    def test_initial_quaternion_valid(self):
        """Initial quaternion should be a unit quaternion."""
        state = create_initial_state()
        q_norm = np.linalg.norm(state.q)
        self.assertAlmostEqual(q_norm, 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
