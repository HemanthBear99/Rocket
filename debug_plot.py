
import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.getcwd())

from rlv_sim.main import run_simulation
from scripts.plot_generator import extract_log_data

def debug_plot_data():
    print("Running simulation explicitly for data inspection...")
    state, log, reason = run_simulation(verbose=False)
    data = extract_log_data(log)
    
    print("\nXXX DATA INSPECTION XXX")
    print(f"{'Time':<10} | {'PitchCmd':<10} | {'VelAngle':<10} | {'Diff(AoA)':<10} | {'Phase':<10}")
    print("-" * 60)
    
    # Calculate Velocity Angle as done in plot_generator
    v_rel_mag = np.linalg.norm(data.velocity_rel_vec, axis=1)
    r_mag = np.linalg.norm(data.position, axis=1)
    r_hat = data.position / r_mag[:, np.newaxis]
    v_rel_radial = np.sum(data.velocity_rel_vec * r_hat, axis=1)
    cos_theta = np.clip(v_rel_radial / np.maximum(v_rel_mag, 1.0), -1.0, 1.0)
    velocity_angle = np.degrees(np.arccos(cos_theta))
    
    # Calculate AoA
    aoa = np.abs(data.pitch_angle - velocity_angle)
    
    # Print every 50th step (approx every 0.25s if dt=0.005) -> No, print one row per 5 simulation seconds
    # dt = 0.005. 5s = 1000 steps.
    step = 1000
    
    for i in range(0, len(data.time), step):
        t = data.time[i]
        p = data.pitch_angle[i]
        v = velocity_angle[i]
        a = aoa[i]
        # Infer phase from altitude? rough guess or just print t
        print(f"{t:<10.2f} | {p:<10.3f} | {v:<10.3f} | {a:<10.3f}")
        
    print("-" * 60)
    
    # Check max AoA after t=45s (when we expect Prograde Lock)
    # Find index for t > 50s
    indices = np.where(data.time > 50.0)[0]
    if len(indices) > 0:
        max_aoa_late = np.max(aoa[indices])
        print(f"Max AoA after t=50s: {max_aoa_late:.4f} deg")
    else:
        print("Simulation ended before t=50s")

if __name__ == "__main__":
    debug_plot_data()
