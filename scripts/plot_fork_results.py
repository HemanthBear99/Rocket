"""
RLV Forked Mission Visualization
Generates plots for the multi-branch mission: Phase I, Orbiter, and Booster.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None
    return pd.read_csv(filename)

def plot_mission():
    phase1 = load_data('plots/telemetry_phase_I.csv')
    orbiter = load_data('plots/telemetry_branch_a_orbiter.csv')
    booster = load_data('plots/telemetry_branch_b_booster.csv')
    
    if booster is None:
        print("Missing booster data file. Run run_mission_fork.py first.")
        return

    # 1. Altitude Profile
    plt.figure(figsize=(12, 6))
    if phase1 is not None:
        plt.plot(phase1['time'], phase1['altitude_km'], 'k-', linewidth=2, label='Phase I (Ascent)')
    if orbiter is not None:
        plt.plot(orbiter['time'], orbiter['altitude_km'], 'b--', linewidth=2, label='Branch A (Orbiter)')
    
    plt.plot(booster['time'], booster['altitude_km'], 'r--', linewidth=2, label='Branch B (Booster)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    plt.title('RLV Mission Profile: Altitude vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/fork_altitude_profile.png')
    print("Saved plots/fork_altitude_profile.png")
    
    # 2. Trajectory (Downrange vs Altitude)
    plt.figure(figsize=(12, 6))
    if phase1 is not None:
        plt.plot(phase1['downrange_km'], phase1['altitude_km'], 'k-', linewidth=2, label='Phase I')
    if orbiter is not None:
        plt.plot(orbiter['downrange_km'], orbiter['altitude_km'], 'b--', linewidth=2, label='Branch A: To Orbit')
    
    plt.plot(booster['downrange_km'], booster['altitude_km'], 'r--', linewidth=2, label='Branch B: Return')
    
    plt.xlabel('Downrange (km)')
    plt.ylabel('Altitude (km)')
    plt.title('RLV Mission Trajectory (Vertical Profile)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('plots/fork_trajectory.png')
    print("Saved plots/fork_trajectory.png")

    # 3. Booster Velocity & Pitch (Recovery Logic Check)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(booster['time'], booster['pitch_actual_deg'], 'm-', label='Pitch (deg)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pitch (deg)', color='m')
    ax1.tick_params(axis='y', labelcolor='m')
    
    ax2 = ax1.twinx()
    ax2.plot(booster['time'], booster['downrange_km'], 'g--', label='Downrange (km)')
    ax2.set_ylabel('Downrange (km)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.title('Booster Recovery Analysis: Pitch & Downrange')
    fig.tight_layout()
    plt.savefig('plots/fork_booster_analysis.png')
    print("Saved plots/fork_booster_analysis.png")

if __name__ == "__main__":
    plot_mission()
