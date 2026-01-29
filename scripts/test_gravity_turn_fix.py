#!/usr/bin/env python3
"""
Quick test of gravity turn fix.
Runs simulation and analyzes key metrics.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rlv_sim.main import run_simulation
import warnings
warnings.filterwarnings('ignore')

def test_gravity_turn():
    """Run simulation and check if gravity turn fix is working."""
    
    print("\n" + "="*80)
    print("TESTING GRAVITY TURN FIX")
    print("="*80)
    print("\nRunning Phase-I simulation...")
    
    try:
        state, log, reason = run_simulation(verbose=False)
    except Exception as e:
        print(f"ERROR in simulation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Extract data
    time = np.array(log.time)
    altitude = np.array(log.altitude)  # already in km
    velocity = np.array(log.velocity)
    pitch = np.array(log.pitch_angle)
    vx = np.array(log.velocity_x)
    vy = np.array(log.velocity_y)
    vz = np.array(log.velocity_z)
    
    # Find key velocity milestones
    print("\n" + "-"*80)
    print("KEY MILESTONES - Velocity Profile")
    print("-"*80)
    print(f"{'Velocity':<12} {'Time':<8} {'Altitude':<12} {'Pitch':<10} {'Horiz Vel':<12} {'Status'}")
    print("-"*80)
    
    milestones = {
        100: "Should start pitching over",
        300: "Pitchover in progress",
        500: "Mid-turn (should be ~35-40°)",
        800: "Aggressive turn continues (should be ~40-50°)",
        1000: "Further turn (should be ~50-60°)",
        1500: "Late turn (should be ~65-75°)",
        2000: "Near final (should be ~70-80°)"
    }
    
    all_pass = True
    
    for v_target in sorted(milestones.keys()):
        # Find index closest to target velocity
        idx = np.argmin(np.abs(velocity - v_target))
        
        # Skip if we haven't reached this velocity yet or it's too far off
        if velocity[idx] < v_target - 50 or (velocity[idx] > velocity[-1] and velocity[idx] != v_target):
            continue
        
        t_val = time[idx]
        h_val = altitude[idx]
        p_val = pitch[idx]
        vh = np.sqrt(vx[idx]**2 + vy[idx]**2)
        
        # Assess
        if v_target < 200:
            expected_range = "0-10 deg"
            target_range = (0, 10)
        elif v_target < 500:
            expected_range = "15-35 deg"
            target_range = (15, 35)
        elif v_target < 800:
            expected_range = "35-50 deg"
            target_range = (35, 50)
        elif v_target < 1200:
            expected_range = "50-65 deg"
            target_range = (50, 65)
        else:
            expected_range = "65-80 deg"
            target_range = (65, 80)
        
        status = "[OK]" if target_range[0] <= p_val <= target_range[1] else "[FAIL]"
        if status == "[FAIL]":
            all_pass = False
        
        print(f"{velocity[idx]:<12.1f} {t_val:<8.2f} {h_val:<12.2f} {p_val:<10.1f} {vh:<12.1f} {status} {expected_range}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL STATE AT MECO")
    print("="*80)
    
    final_pitch = pitch[-1]
    final_alt = altitude[-1]
    final_vel = velocity[-1]
    final_vh = np.sqrt(vx[-1]**2 + vy[-1]**2)
    final_vz = vz[-1]
    
    print(f"Altitude:       {final_alt:.2f} km")
    print(f"Velocity:       {final_vel:.2f} m/s")
    print(f"Horizontal V:   {final_vh:.2f} m/s")
    print(f"Vertical V:     {final_vz:.2f} m/s")
    print(f"Pitch Angle:    {final_pitch:.2f}°")
    print(f"Trajectory Tilt: {np.degrees(np.arctan2(final_vh, final_vz)):.2f}°")
    
    # Final checks
    print("\n" + "="*80)
    print("VERIFICATION CHECKS")
    print("="*80)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Pitch increases significantly
    total_checks += 1
    if final_pitch > 50:
        print("[OK] CHECK 1 PASS: Final pitch >50 deg (good gravity turn)")
        checks_passed += 1
    else:
        print(f"[FAIL] CHECK 1 FAIL: Final pitch {final_pitch:.1f} deg should be >50 deg")
    
    # Check 2: Horizontal velocity is substantial
    total_checks += 1
    if final_vh > 1000:
        print(f"[OK] CHECK 2 PASS: Horizontal velocity {final_vh:.0f} m/s (good)")
        checks_passed += 1
    else:
        print(f"[FAIL] CHECK 2 FAIL: Horizontal velocity {final_vh:.0f} m/s should be >1000 m/s")
    
    # Check 3: Altitude is reasonable
    total_checks += 1
    if final_alt > 90:
        print(f"[OK] CHECK 3 PASS: Altitude {final_alt:.1f} km (good trajectory)")
        checks_passed += 1
    else:
        print(f"[FAIL] CHECK 3 FAIL: Altitude {final_alt:.1f} km seems low")
    
    # Check 4: Pitch is increasing during ascent (sample 50% mark)
    total_checks += 1
    mid_idx = len(pitch) // 2
    pitch_at_50pct = pitch[mid_idx]
    if pitch_at_50pct > pitch[100]:  # Compare to early pitch
        print(f"[OK] CHECK 4 PASS: Pitch increases during ascent ({pitch[100]:.1f} deg -> {pitch_at_50pct:.1f} deg)")
        checks_passed += 1
    else:
        print(f"[FAIL] CHECK 4 FAIL: Pitch should increase during ascent")
    
    print("\n" + "="*80)
    print(f"RESULT: {checks_passed}/{total_checks} checks passed")
    print("="*80)
    
    return checks_passed == total_checks

if __name__ == '__main__':
    success = test_gravity_turn()
    sys.exit(0 if success else 1)
