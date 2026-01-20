
import unittest
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlv_sim import guidance
from rlv_sim import constants as C

class TestGuidanceLogic(unittest.TestCase):
    
    def test_local_vertical(self):
        """Verify local vertical computation"""
        r = np.array([C.R_EARTH, 0, 0])
        vertical = guidance.compute_local_vertical(r)
        
        # Should be [1, 0, 0]
        np.testing.assert_array_almost_equal(vertical, np.array([1.0, 0.0, 0.0]))
        
    def test_pitchover_trigger(self):
        """Verify Pitchover output direction"""
        # Conditions for Pitchover:
        # 1. Low Altitude (e.g. 500m)
        # 2. Low Alpha (Gravity turn hasn't started dominated)
        
        r = np.array([C.R_EARTH + 500.0, 0, 0]) # 500m altitude
        v = np.array([10.0, 465.0, 0]) # Low vertical velocity, ~Earth rotation tangential
        
        # Time doesn't matter for this function
        thrust_dir = guidance.compute_desired_thrust_direction(r, v, 0.0)
        
        # Expected behavior:
        # Vertical is [1, 0, 0]
        # East is [0, 1, 0]
        # Pitchover Angle is 2 degrees (0.0349 rad)
        # Result should be tilted towards +Y
        
        # Check Y component is positive (Eastward kick)
        self.assertTrue(thrust_dir[1] > 0.01, "Thrust vector should have positive Y component (East kick)")
        
        # Check Angle
        vertical = np.array([1.0, 0.0, 0.0])
        cos_theta = np.dot(thrust_dir, vertical)
        angle_rad = np.arccos(cos_theta)
        
        # Should be close to PITCHOVER_ANGLE
        self.assertAlmostEqual(angle_rad, C.PITCHOVER_ANGLE, places=4, msg="Pitchover angle incorrect")

    def test_gravity_turn_prograde(self):
        """Verify Gravity Turn follows velocity"""
        # High altitude (e.g. 50km), High velocity
        r = np.array([C.R_EARTH + 50000.0, 0, 0]) 
        v = np.array([1000.0, 1000.0, 0.0]) # 45 degree flight path roughly
        
        thrust_dir = guidance.compute_desired_thrust_direction(r, v, 0.0)
        
        # In gravity turn, thrust should align with v_rel (roughly v here since v >> v_wind)
        # Let's just check it's not purely vertical
        self.assertTrue(thrust_dir[1] > 0.1, "Gravity turn should steer downrange")

if __name__ == '__main__':
    unittest.main()
