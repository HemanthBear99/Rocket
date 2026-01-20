"""Tests for physics upgrades including atmosphere model and Mach-dependent drag."""

import unittest

import numpy as np

from rlv_sim import forces
from rlv_sim import constants as C

class TestPhysicsUpgrades(unittest.TestCase):
    
    def test_atmosphere_sea_level(self):
        """Verify US Standard Atmosphere at Sea Level"""
        T, P, rho, a = forces.compute_atmosphere_properties(0.0)
        
        self.assertAlmostEqual(T, 288.15, places=1, msg="Sea Level Temp wrong")
        self.assertAlmostEqual(P, 101325.0, places=0, msg="Sea Level Pressure wrong")
        self.assertAlmostEqual(rho, 1.225, places=3, msg="Sea Level Density wrong")
        
    def test_atmosphere_tropopause(self):
        """Verify Atmosphere at 11km (Tropopause)"""
        T, P, rho, a = forces.compute_atmosphere_properties(11000.0)
        
        # Standard values at 11km
        self.assertAlmostEqual(T, 216.65, places=1, msg="Tropopause Temp wrong")
        self.assertTrue(P < 23000, "Pressure at 11km should be approx 22.6kPa")
        
    def test_stratosphere_isothermal(self):
        """Verify Stratosphere (11km-20km) is Isothermal"""
        T11, _, _, _ = forces.compute_atmosphere_properties(11000.0)
        T20, _, _, _ = forces.compute_atmosphere_properties(20000.0)
        
        self.assertAlmostEqual(T11, T20, places=4, msg="Lower Stratosphere should be isothermal")
        
    def test_drag_coefficient_mach(self):
        """Verify Mach-dependent Drag Coefficient"""
        # Test Subsonic
        mach_sub = 0.5
        cd_sub = np.interp(mach_sub, C.MACH_BREAKPOINTS, C.CD_VALUES)
        self.assertEqual(cd_sub, 0.42, "Subsonic Cd should be constant 0.42")
        
        # Test Transonic Spike
        mach_trans = 1.05
        cd_trans = np.interp(mach_trans, C.MACH_BREAKPOINTS, C.CD_VALUES)
        self.assertEqual(cd_trans, 0.75, "Transonic Cd spike missing")
        
        # Test Supersonic Decay
        mach_super = 5.0
        cd_super = np.interp(mach_super, C.MACH_BREAKPOINTS, C.CD_VALUES)
        self.assertEqual(cd_super, 0.35, "Supersonic Cd decay wrong")
        
    def test_vacuum_thrust(self):
        """Verify Thrust scaling with pressure"""
        # Sea Level
        F_sl = forces.compute_thrust_force(np.array([1,0,0,0]), np.array([C.R_EARTH,0,0]), thrust_on=True)
        F_sl_mag = np.linalg.norm(F_sl)
        # Should be close to constants.THRUST_MAGNITUDE (7.6 MN)
        # Note: The calculation in forces.py might have slight deviations due to mdot/Isp/g0 consistency
        # but let's check it's within 1%
        self.assertAlmostEqual(F_sl_mag/1e6, 7.6, delta=0.1, msg="Sea Level Thrust magnitude mismatch")
        
        # Vacuum (Deep Space)
        r_vac = np.array([C.R_EARTH + 200000.0, 0, 0]) # 200km up
        F_vac = forces.compute_thrust_force(np.array([1,0,0,0]), r_vac, thrust_on=True)
        F_vac_mag = np.linalg.norm(F_vac)
        
        # Expected Vacuum Thrust: T_vac = mdot * Isp_vac * g0
        # mdot = 7.6e6 / (282 * 9.80665) = 2748.5 kg/s
        # T_vac = 2748.5 * 311 * 9.80665 = 8.38 MN
        expected_vac_thrust = (C.MASS_FLOW_RATE * C.ISP_VAC * C.G0)
        
        self.assertAlmostEqual(F_vac_mag, expected_vac_thrust, delta=1000.0, msg="Vacuum Thrust calculation incorrect")
        self.assertTrue(F_vac_mag > F_sl_mag, "Vacuum thrust should be higher than sea level thrust")

if __name__ == '__main__':
    unittest.main()
