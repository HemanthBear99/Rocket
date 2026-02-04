
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())
from rlv_sim.main import run_simulation
from scripts.plot_generator import extract_log_data

def debug_plot_data():
    print("RUNNING SIMULATION...")
    state, log, reason = run_simulation(verbose=False)
    data = extract_log_data(log)
    
    # Calculate AoA
    v_rel_mag = np.linalg.norm(data.velocity_rel_vec, axis=1)
    r_mag = np.linalg.norm(data.position, axis=1)
    r_hat = data.position / r_mag[:, np.newaxis]
    v_rel_radial = np.sum(data.velocity_rel_vec * r_hat, axis=1)
    cos_theta = np.clip(v_rel_radial / np.maximum(v_rel_mag, 1.0), -1.0, 1.0)
    velocity_angle = np.degrees(np.arccos(cos_theta))
    aoa = np.abs(data.pitch_angle - velocity_angle)
    
    print("\nXXX PROGRADE LOCK CHECK (t > 70s) XXX")
    indices = np.where(data.time > 70.0)[0]
    
    if len(indices) == 0:
        print("Simulation ended before 70s!")
        return

    # Check average and max AoA in locked phase
    aoa_locked = aoa[indices]
    avg_aoa = np.mean(aoa_locked)
    max_aoa = np.max(aoa_locked)
    
    print(f"Time Range: {data.time[indices[0]]:.1f}s to {data.time[indices[-1]]:.1f}s")
    print(f"Altitude Range: {data.altitude[indices[0]]:.1f}km to {data.altitude[indices[-1]]:.1f}km")
    print(f"Average AoA: {avg_aoa:.6f} deg")
    print(f"Max AoA:     {max_aoa:.6f} deg")
    
    # Print a few samples
    print("\nSAMPLES:")
    for idx in indices[::1000]: # Every ~5s
        print(f"t={data.time[idx]:.1f}s | Alt={data.altitude[idx]:.1f}km | AoA={aoa[idx]:.6f} deg")

    if max_aoa < 0.1:
        print("\nSUCCESS: Prograde Lock verified (AoA < 0.1 deg)")
    else:
        print("\nFAILURE: Prograde Lock not active!")

if __name__ == "__main__":
    debug_plot_data()
