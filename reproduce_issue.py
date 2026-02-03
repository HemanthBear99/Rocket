
import numpy as np
import matplotlib.pyplot as plt
import os
from rlv_sim.main import run_simulation
from rlv_sim import constants as C

def calculate_ground_track_distance(log):
    # Re-calculate downrange accounting for Earth rotation (Ground Track)
    downrange_ground = []
    
    r0 = C.INITIAL_POSITION
    omega_earth = C.EARTH_ROTATION_RATE
    
    for i, t in enumerate(log.time):
        r_inertial = np.array([log.position_x[i], log.position_y[i], log.position_z[i]])
        
        # Rotate ECI to ECEF (assuming they align at t=0)
        theta = omega_earth * t
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Rotation matrix Z-axis (Active rotation of frame? No, passive rotation of vector)
        # r_ecef = R_z(theta) * r_eci
        # R_z(theta) = [[cos t, sin t, 0], [-sin t, cos t, 0], [0, 0, 1]]
        
        x_new = r_inertial[0] * cos_t + r_inertial[1] * sin_t
        y_new = -r_inertial[0] * sin_t + r_inertial[1] * cos_t
        z_new = r_inertial[2]
        
        r_ecef = np.array([x_new, y_new, z_new])
        
        # Angle between r_ecef and r0 (r0 is fixed on surface in ECEF)
        r_ecef_hat = r_ecef / np.linalg.norm(r_ecef)
        r0_hat = r0 / np.linalg.norm(r0)
        
        dot = np.clip(np.dot(r_ecef_hat, r0_hat), -1.0, 1.0)
        angle = np.arccos(dot)
        dist = C.R_EARTH * angle
        downrange_ground.append(dist / 1000.0) # km
        
    return downrange_ground

def main():
    print("Running Simulation...")
    final_state, log, reason = run_simulation(verbose=False)
    
    # Calculate corrected downrange
    downrange_ground_km = calculate_ground_track_distance(log)
    
    print(f"\nSimulation Results (Reason: {reason})")
    print(f"Final Time: {final_state.t:.2f} s")
    print(f"Final Alt: {final_state.altitude/1000:.2f} km")
    print(f"Final Inertial Downrange: {log.downrange[-1]:.2f} km")
    print(f"Final Ground Track Downrange: {downrange_ground_km[-1]:.2f} km")
    
    # Check diagnostics at MECO (or end)
    idx = -1
    print("\nDiagnostics at Final Step:")
    print(f"  Pitch Angle (from Vert): {log.actual_pitch_angle[idx]:.2f} deg")
    print(f"  Velocity Tilt (from Vert): {log.velocity_tilt_deg[idx]:.2f} deg")
    print(f"  Gamma (Relative, from Horiz): {log.gamma_actual_deg[idx]:.2f} deg")
    
    # Plotting
    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # 1. Pitch vs Velocity Tilt vs Gamma
        ax = axes[0]
        ax.plot(log.time, log.actual_pitch_angle, label='Pitch Angle (deg from Vert)', color='blue')
        ax.plot(log.time, log.velocity_tilt_deg, label='Velocity Tilt (deg from Vert)', color='green', linestyle='--')
        ax.plot(log.time, 90.0 - np.array(log.gamma_actual_deg), label='90 - Gamma (deg from Vert)', color='red', linestyle=':')
        ax.set_ylabel('Angle from Vertical (deg)')
        ax.set_title('Diagnostic: Pitch vs Velocity Alignment')
        ax.legend()
        ax.grid(True)
        
        # 2. Downrange comparison
        ax = axes[1]
        ax.plot(log.time, log.downrange, label='Downrange (Inertial)', color='black')
        ax.plot(log.time, downrange_ground_km, label='Downrange (Ground Track)', color='magenta')
        ax.set_ylabel('Distance (km)')
        ax.set_title('Downrange Distance')
        ax.legend()
        ax.grid(True)
        
        # 3. Velocity
        ax = axes[2]
        ax.plot(log.time, log.velocity, label='Inertial Speed', color='blue')
        ax.plot(log.time, log.velocity_rel, label='Relative Speed', color='green')
        ax.set_ylabel('Speed (m/s)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)
        
        os.makedirs('diagnostics', exist_ok=True)
        plt.savefig('diagnostics/physics_check.png')
        print("\nSaved diagnostics/physics_check.png")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plot.")

if __name__ == "__main__":
    main()
