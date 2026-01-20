import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure we can import rlv_sim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlv_sim.main import run_simulation
import rlv_sim.constants as C

def generate_plots():
    # 1. Run Simulation
    print("Running simulation for plot generation...")
    state, log, reason = run_simulation(verbose=False)
    
    # Print summary for README
    print("\n" + "="*60)
    print("SIMULATION PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Final Altitude: {state.altitude/1000:.2f} km")
    print(f"Final Speed:    {state.speed:.2f} m/s")
    print(f"Flight Time:    {state.t:.2f} s")
    print(f"Propellant:     {state.m - C.DRY_MASS:.2f} kg remaining")
    print(f"Termination:    {reason}")
    print("="*60 + "\n")

    # 2. Setup plotting
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a nice style if available
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass

    # 3. Create Plots
    
    # Plot 1: Altitude & Velocity
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (km)', color='tab:blue')
    ax1.plot(log.time, log.altitude, color='tab:blue', linewidth=2, label='Altitude')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Velocity (m/s)', color='tab:orange')
    ax2.plot(log.time, log.velocity, color='tab:orange', linewidth=2, label='Velocity')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.title('RLV Phase-I: Ascent Profile')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ascent_profile.png'), dpi=150)
    plt.close()
    
    # Plot 2: Trajectory (X vs Z) - "The Arc"
    plt.figure(figsize=(8, 8))
    # Convert position to km
    x_km = np.array(log.position_x) / 1000.0
    z_km = np.array(log.position_y) / 1000.0 # Y is East/Downrange in this sim logic (see guidance.py)
    # Actually checking constants: 
    # Initial pos: [R, 0, 0]. Radial is X. 
    # Guidance kicks to East (+Y).
    # So Altitude is roughly r_norm - R.
    # Downrange is Y.
    
    # Let's verify coordinate mapping:
    # r[0] is roughly R_earth + alt
    # r[1] is East (tangential)
    # r[2] is North (tangential) - usually 0 for equatorial launch
    
    r_mag = np.sqrt(np.array(log.position_x)**2 + np.array(log.position_y)**2)
    alt = (r_mag - C.R_EARTH) / 1000.0
    downrange = np.array(log.position_y) / 1000.0
    
    plt.plot(downrange, alt, linewidth=2, color='purple')
    plt.fill_between(downrange, 0, alt, color='purple', alpha=0.1)
    plt.xlabel('Downrange Distance (km)')
    plt.ylabel('Altitude (km)')
    plt.title('Ascent Trajectory (Downrange vs Altitude)')
    plt.grid(True, linestyle='--')
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'trajectory.png'), dpi=150)
    plt.close()
    
    # Plot 3: Attitude Control (Pitch & Error)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(log.time, log.pitch_angle, color='green', label='Pitch Angle (deg)')
    ax1.set_ylabel('Pitch (deg) from Vertical')
    ax1.set_title('Attitude Dynamics')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(log.time, log.attitude_error, color='red', label='Error')
    ax2.set_ylabel('Control Error (deg)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    ax2.legend()
    
    plt.savefig(os.path.join(output_dir, 'control_dynamics.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    generate_plots()
