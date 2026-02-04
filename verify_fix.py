import numpy as np
import matplotlib.pyplot as plt
import os
from rlv_sim.main import run_simulation
from rlv_sim import constants as C

def main():
    print("Running Verification Simulation (Verbose)...")
    # Run with verbose=True to see progress
    final_state, log, reason = run_simulation(verbose=True)
    
    print(f"\nSimulation Completed. Reason: {reason}")
    print("Generating Validation Plots...")
    
    os.makedirs('diagnostics', exist_ok=True)
    
    # 1. Plot AoA and Pitch Alignment
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Upper: Angles
        ax1.plot(log.time, log.actual_pitch_angle, label='Pitch Actual', color='blue')
        ax1.plot(log.time, log.pitch_angle, label='Pitch Command', color='cyan', linestyle=':')
        ax1.plot(log.time, log.velocity_tilt_deg, label='Velocity Tilt', color='green', linestyle='--')
        ax1.set_title('Pitch & Gamma Alignment')
        ax1.legend()
        ax1.grid(True)
        
        # Lower: AoA (Actual vs Velocity) AND (Command vs Velocity)
        # AoA Actual = |Actual - Velocity|
        aoa_actual = np.abs(np.array(log.actual_pitch_angle) - np.array(log.velocity_tilt_deg))
        # AoA Command = |Command - Velocity|
        aoa_cmd = np.abs(np.array(log.pitch_angle) - np.array(log.velocity_tilt_deg))
        
        ax2.plot(log.time, aoa_actual, label='AoA Actual (deg)', color='red')
        ax2.plot(log.time, aoa_cmd, label='AoA Command (deg)', color='orange', linestyle='--')
        
        # Add dynamic pressure on secondary axis
        ax2b = ax2.twinx()
        q_kpa = [0.5 * 1.225 * np.exp(-h/8500) * v**2 / 1000 for h, v in zip(log.altitude, log.velocity_rel)]
        ax2b.plot(log.time, q_kpa, color='gray', alpha=0.3, label='q (approx kPa)')
        ax2b.set_ylabel('Dynamic Pressure (kPa)')
        
        ax2.set_ylabel('Angle of Attack (deg)')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Angle of Attack vs Time')
        ax2.axhline(6.0, color='k', linestyle='--', label='Limit Target (6 deg)')
        ax2.legend(loc='upper left')
        ax2b.legend(loc='upper right')
        ax2.grid(True)
        
        plt.savefig('diagnostics/verification_aoa.png')
        print("Saved diagnostics/verification_aoa.png")
        
    except Exception as e:
        print(f"Plotting failed: {e}")

    # 2. Plot Gamma Profile smoothness
    try:
        plt.figure()
        plt.plot(log.time, log.gamma_command_deg, label='Gamma Cmd')
        plt.plot(log.time, log.gamma_actual_deg, label='Gamma Actual')
        plt.title('Gamma Profile Smoothness')
        plt.xlabel('Time (s)')
        plt.ylabel('Gamma (deg from horiz)')
        plt.legend()
        plt.grid(True)
        plt.savefig('diagnostics/verification_gamma.png')
        print("Saved diagnostics/verification_gamma.png")
    except Exception as e:
        print(f"Gamma plot failed: {e}")

if __name__ == "__main__":
    main()
