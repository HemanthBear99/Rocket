"""
RLV Phase-I Ascent Simulation - Trajectory Verification Script
"""

import numpy as np

from rlv_sim import run_simulation, constants as C

def analyze_trajectory():
    print("Running simulation for trajectory analysis...")
    state, log, reason = run_simulation(verbose=False)
    
    # Analyze Final State
    r_final = state.r
    v_final = state.v
    
    print(f"\nFinal Position (Inertial): {r_final}")
    print(f"Final Velocity (Inertial): {v_final}")
    
    # Calculate Wind and Relative Velocity
    omega_earth = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    v_wind = np.cross(omega_earth, r_final)
    v_rel = v_final - v_wind
    
    print(f"Wind Velocity at Final Alt: {v_wind}")
    print(f"Relative Velocity (Air):    {v_rel}")
    
    # Check Direction (East vs West)
    # Launch site is +X. Tangential East is +Y.
    # We want +Y relative velocity (flying East faster than Earth spins).
    
    if v_rel[1] < 0:
        print("\n[CRITICAL] Trajectory is WESTWARD (Retrograde relative to Air)!")
        print("  Reason: Lack of pitchover maneuver allowed Coriolis drift to dominate.")
    else:
        print("\n[SUCCESS] Trajectory is EASTWARD (Prograde).")
        
    # Check angle
    angle_from_vertical = np.arccos(np.dot(r_final, v_rel) / (np.linalg.norm(r_final) * np.linalg.norm(v_rel)))
    print(f"Flight Path Angle (rel): {np.degrees(angle_from_vertical):.2f} deg")

if __name__ == "__main__":
    analyze_trajectory()
