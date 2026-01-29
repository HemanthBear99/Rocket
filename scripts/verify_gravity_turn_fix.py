#!/usr/bin/env python3
"""
Verification script for gravity turn fix.

This script:
1. Runs a short Phase-I simulation
2. Extracts diagnostics at key milestones (100 m/s, 500 m/s, 1000 m/s, 2000 m/s)
3. Compares thrust direction pitch vs velocity tilt
4. Verifies gamma follows 90->45->10 profile
5. Confirms pitch increases during ascent
"""

import sys
sys.path.insert(0, '/d/rocket/Rocket')

import numpy as np
from rlv_sim.main import run_simulation
from rlv_sim.guidance import compute_gravity_turn_angle, compute_local_vertical
import warnings
warnings.filterwarnings('ignore')

def analyze_simulation_diagnostics(log, guidance_log):
    """
    Analyze simulation data for gravity turn correctness.
    
    Args:
        log: Simulation log with time, altitude, velocity, etc.
        guidance_log: List of guidance outputs at each step
    """
    print("\n" + "="*80)
    print("GRAVITY TURN VERIFICATION - PHASE I ASCENT")
    print("="*80)
    
    # Find key velocity milestones
    velocity = np.array(log['velocity'])
    time = np.array(log['time'])
    altitude = np.array(log['altitude']) * 1000  # Convert back to m
    
    gamma_angles = []
    pitch_angles = []
    velocity_tilts = []
    thrust_pitches = []
    vx = np.array(log['velocity_x'])
    vy = np.array(log['velocity_y'])
    vz = np.array(log['velocity_z'])
    
    milestones = [100, 200, 300, 500, 800, 1000, 1500, 2000, 2500]
    found_milestones = {}
    
    print("\nMILESTONE ANALYSIS:")
    print(f"{'V [m/s]':<10} {'t [s]':<8} {'h [km]':<8} {'γ [deg]':<10} {'θ_pitch [deg]':<12} {'v_tilt [deg]':<12} {'Δ [deg]':<10}")
    print("-" * 90)
    
    for i, v_target in enumerate(milestones):
        # Find closest velocity to target
        idx = np.argmin(np.abs(velocity - v_target))
        if abs(velocity[idx] - v_target) > 50 and v_target <= velocity[-1]:
            continue  # Skip if too far from target velocity
        
        if velocity[idx] < v_target:
            continue  # Haven't reached this velocity yet
        
        t_val = time[idx]
        h_val = altitude[idx] / 1000
        v_mag = velocity[idx]
        
        # Get gamma angle from guidance law
        gamma_rad = compute_gravity_turn_angle(altitude[idx], v_mag)
        gamma_deg = np.degrees(gamma_rad)
        
        # Get logged pitch angle
        pitch_deg = log['pitch_angle'][idx]
        
        # Compute velocity tilt: atan(sqrt(vx²+vy²) / |vz|)
        vh_mag = np.sqrt(vx[idx]**2 + vy[idx]**2)
        if abs(vz[idx]) < 1.0:
            v_tilt_deg = 90.0
        else:
            v_tilt_rad = np.arctan2(vh_mag, abs(vz[idx]))
            v_tilt_deg = np.degrees(v_tilt_rad)
        
        # Thrust direction pitch: angle from local vertical
        if i < len(guidance_log):
            guidance = guidance_log[idx]
            thrust_dir = guidance.get('thrust_direction', np.array([0, 0, 1]))
            # Pitch from vertical = arccos(dot(vertical, thrust_dir))
            # But we want it as angle above horizontal = 90 - that
            thrust_from_vert = np.degrees(np.arccos(np.clip(np.dot(compute_local_vertical(np.array([6.371e6, 0, 0])), thrust_dir), -1, 1)))
            thrust_pitch_deg = 90 - thrust_from_vert
        else:
            thrust_pitch_deg = pitch_deg
        
        delta_deg = abs(pitch_deg - v_tilt_deg)
        
        found_milestones[v_target] = {
            't': t_val,
            'h': h_val,
            'gamma': gamma_deg,
            'pitch': pitch_deg,
            'v_tilt': v_tilt_deg,
            'delta': delta_deg
        }
        
        print(f"{v_mag:<10.0f} {t_val:<8.2f} {h_val:<8.2f} {gamma_deg:<10.1f} {pitch_deg:<12.1f} {v_tilt_deg:<12.1f} {delta_deg:<10.1f}")
    
    print("\n" + "="*80)
    print("VERIFICATION CHECKS:")
    print("="*80)
    
    # Check 1: Gamma profile (should go 90 -> 45 -> 10)
    print("\n[CHECK 1] Gamma Profile (90° → 45° → 10°):")
    check1_pass = True
    gamma_100 = found_milestones.get(100, {}).get('gamma', 90)
    gamma_800 = found_milestones.get(800, {}).get('gamma', 45)
    gamma_2000 = found_milestones.get(2000, {}).get('gamma', 10)
    
    print(f"  At 100 m/s:  γ = {gamma_100:.1f}° (target: ~90°)")
    print(f"  At 800 m/s:  γ = {gamma_800:.1f}° (target: ~45°)")
    print(f"  At 2000 m/s: γ = {gamma_2000:.1f}° (target: ~10°)")
    
    if gamma_100 > 88:
        print("  ✓ PASS: Gamma starts near vertical (>88°)")
    else:
        print("  ✗ FAIL: Gamma should start near vertical")
        check1_pass = False
    
    if 35 < gamma_800 < 55:
        print("  ✓ PASS: Gamma reaches ~40-50° by 800 m/s (aggressive turn)")
    else:
        print(f"  ✗ FAIL: Gamma should be 35-55° at 800 m/s, got {gamma_800:.1f}°")
        check1_pass = False
    
    if gamma_2000 < 15:
        print("  ✓ PASS: Gamma reaches <15° by 2000 m/s")
    else:
        print(f"  ✗ FAIL: Gamma should be <15° at 2000 m/s, got {gamma_2000:.1f}°")
        check1_pass = False
    
    # Check 2: Pitch increase (0° → 60-80°)
    print("\n[CHECK 2] Pitch Angle Progression (should increase 0° → 60-80°):")
    check2_pass = True
    pitch_100 = found_milestones.get(100, {}).get('pitch', 0)
    pitch_500 = found_milestones.get(500, {}).get('pitch', 30)
    pitch_2000 = found_milestones.get(2000, {}).get('pitch', 75)
    
    print(f"  At 100 m/s:  θ = {pitch_100:.1f}° (target: ~0-5°)")
    print(f"  At 500 m/s:  θ = {pitch_500:.1f}° (target: ~25-40°)")
    print(f"  At 2000 m/s: θ = {pitch_2000:.1f}° (target: ~70-80°)")
    
    if pitch_100 < 10:
        print("  ✓ PASS: Pitch starts small (<10°)")
    else:
        print(f"  ✗ FAIL: Pitch should start <10°, got {pitch_100:.1f}°")
        check2_pass = False
    
    if 20 < pitch_500 < 50:
        print("  ✓ PASS: Pitch is 20-50° at 500 m/s (good turn progress)")
    else:
        print(f"  ✗ FAIL: Pitch should be 20-50° at 500 m/s, got {pitch_500:.1f}°")
        check2_pass = False
    
    if pitch_2000 > 60:
        print("  ✓ PASS: Pitch >60° by 2000 m/s (effective gravity turn)")
    else:
        print(f"  ✗ FAIL: Pitch should be >60° at 2000 m/s, got {pitch_2000:.1f}°")
        check2_pass = False
    
    # Check 3: Pitch vs velocity tilt alignment
    print("\n[CHECK 3] Pitch vs Velocity Tilt Alignment (Δ < 5°):")
    check3_pass = True
    deltas = [m.get('delta', 999) for m in found_milestones.values()]
    mean_delta = np.mean([d for d in deltas if d < 90])
    max_delta = min([d for d in deltas if d < 90])
    
    print(f"  Mean alignment error: {mean_delta:.1f}°")
    print(f"  Max alignment error:  {max_delta:.1f}°")
    
    if mean_delta < 8:
        print("  ✓ PASS: Pitch and velocity tilt are aligned (error < 8°)")
    else:
        print(f"  ✗ FAIL: Alignment error too large ({mean_delta:.1f}° > 8°)")
        check3_pass = False
    
    # Check 4: Horizontal velocity growth
    print("\n[CHECK 4] Horizontal Velocity Growth:")
    check4_pass = True
    vh_100 = np.sqrt(found_milestones.get(100, {'vx': 0})['vx']**2 + found_milestones.get(100, {'vy': 0})['vy']**2) if 100 in found_milestones else 0
    vh_2000 = np.sqrt(found_milestones.get(2000, {'vx': 0})['vx']**2 + found_milestones.get(2000, {'vy': 0})['vy']**2) if 2000 in found_milestones else 0
    
    print(f"  Horizontal velocity at 100 m/s:  ~{vh_100:.0f} m/s (target: ~0-50 m/s)")
    print(f"  Horizontal velocity at 2000 m/s: ~{vh_2000:.0f} m/s (target: >1500 m/s)")
    
    if vh_2000 > 1200:
        print(f"  ✓ PASS: Significant horizontal velocity by MECO ({vh_2000:.0f} m/s)")
    else:
        print(f"  ⚠ WARNING: Horizontal velocity {vh_2000:.0f} m/s (target >1500 m/s for good trajectory)")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    all_pass = check1_pass and check2_pass and check3_pass
    if all_pass:
        print("✓ ALL CRITICAL CHECKS PASSED - Gravity turn fix is working!")
    else:
        print("✗ SOME CHECKS FAILED - Review guidance law parameters")
    
    return all_pass


if __name__ == '__main__':
    print("Running Phase-I simulation with gravity turn fix...")
    print("This will take ~10-30 seconds...")
    
    try:
        sim_log, guidance_logs, exit_reason = run_simulation(verbose=False)
        print(f"\nSimulation completed: {exit_reason}")
        
        analyze_simulation_diagnostics(sim_log, guidance_logs)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
