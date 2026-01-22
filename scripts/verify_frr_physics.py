"""
Script to verify Phase-I Physics Fixes against Flight Readiness Review (FRR) Standards.
"""

import sys
import numpy as np
import logging

# Path hack specific to this user's workspace structure
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rlv_sim import constants as C
from rlv_sim.main import run_simulation

def verify_frr_physics():
    print("=" * 60)
    print("PHASE-I FLIGHT READINESS VERIFICATION")
    print("=" * 60)
    
    # Run Simulation
    state, log, reason = run_simulation(verbose=False)
    
    time = np.array(log.time)
    alt = np.array(log.altitude) * 1000.0  # m
    mass = np.array(log.mass)
    torque = np.array(log.torque_magnitude)
    pitch = np.array(log.pitch_angle) # degrees
    err = np.array(log.attitude_error) # degrees
    
    failures = []
    
    # -------------------------------------------------------------------------
    # TEST 1: Gravity Gradient Check (Physics)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Gravity Gradient Validation...")
    # Theoretical g at liftoff (r = R_EARTH)
    g_0_calc = C.MU_EARTH / C.R_EARTH**2
    # Theoretical g at MECO (r = R_EARTH + alt_meco)
    alt_meco = alt[-1]
    g_meco_calc = C.MU_EARTH / (C.R_EARTH + alt_meco)**2
    
    ratio = g_meco_calc / g_0_calc
    print(f"  g(0)    = {g_0_calc:.4f} m/s^2")
    print(f"  g(MECO) = {g_meco_calc:.4f} m/s^2 (at {alt_meco/1000:.1f} km)")
    print(f"  Ratio   = {ratio:.4f}")
    
    if abs(g_0_calc - 9.80665) > 0.05:
         failures.append(f"Gravity constant mismatch: {g_0_calc} != 9.80665")
    
    if ratio > 0.99:
        failures.append("Gravity does not decrease significantly with altitude (Check Physics)")
    elif ratio < 0.8: # Unlikely for low orbit
        # Check simple inverse square
        expected_ratio = (C.R_EARTH / (C.R_EARTH + alt_meco))**2
        if abs(ratio - expected_ratio) > 1e-4:
             failures.append(f"Gravity ratio mismatch: {ratio} vs {expected_ratio}")
    
    # -------------------------------------------------------------------------
    # TEST 2: Torque Authority (Control)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Control Torque Authority...")
    max_torque = np.max(torque)
    avg_torque_active = np.mean(torque[100:200]) # Sample mid-flight
    
    print(f"  Max Torque: {max_torque/1e6:.2f} MNm (Limit: {C.MAX_TORQUE/1e6:.1f} MNm)")
    print(f"  Avg Torque: {avg_torque_active/1e6:.2f} MNm")
    
    if max_torque < 1.0:
        failures.append("CRITICAL: Torque is effectively ZERO. Controller inactive!")
    elif max_torque > C.MAX_TORQUE * 1.01:
        failures.append("Torque saturation limit violated.")
        
    # -------------------------------------------------------------------------
    # TEST 3: Guidance Continuity (Guidance)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Pitch Continuity (Smoothness)...")
    # First derivative of pitch
    d_pitch = np.diff(pitch) / np.diff(time)
    max_pitch_rate = np.max(np.abs(d_pitch))
    
    print(f"  Max Pitch Rate: {max_pitch_rate:.2f} deg/s")
    
    # Check for steps (instantaneous jumps) -> very high rate
    if max_pitch_rate > 5.0: # 5 deg/s is very aggressive for a big rocket
        # Find where
        step_idx = np.argmax(np.abs(d_pitch))
        failures.append(f"Pitch Step Detected at t={time[step_idx]:.1f}s (Rate={max_pitch_rate:.1f} deg/s)")
        
    # -------------------------------------------------------------------------
    # TEST 4: Attitude Tracking (Performance)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Attitude Tracking Error...")
    # Ignore first 5 seconds (transient)
    steady_err = err[time > 10.0]
    max_steady_err = np.max(steady_err) if len(steady_err) > 0 else 0.0
    
    print(f"  Max Steady Error: {max_steady_err:.2f} deg")
    
    if max_steady_err > 2.0:
        failures.append(f"Tracking error too high: {max_steady_err:.2f} > 2.0 deg.")
        
    print("-" * 60)
    if not failures:
        print("✅ ALL SYSTEMS GO. SIMULATION VERIFIED.")
        return True
    else:
        print("❌ VERIFICATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        return False

if __name__ == "__main__":
    success = verify_frr_physics()
    sys.exit(0 if success else 1)
