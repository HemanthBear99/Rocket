"""
RLV Phase-I Ascent Simulation - Run Script

Execute this script to run the complete simulation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlv_sim import run_simulation, plot_results, plot_trajectory_3d


def main():
    print("\n" + "="*60)
    print("Starting RLV Phase-I Ascent Simulation")
    print("="*60 + "\n")
    
    # Run the simulation
    final_state, log, reason = run_simulation(verbose=True)
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Termination reason: {reason}")
    print(f"Final time: {final_state.t:.2f} s")
    print(f"Final altitude: {final_state.altitude/1000:.2f} km")
    print(f"Final velocity: {final_state.speed:.2f} m/s")
    print(f"Final mass: {final_state.m:.2f} kg")
    print(f"Propellant used: {50000 - final_state.m:.2f} kg")
    print("="*60 + "\n")
    
    # Generate plots
    if len(log.time) > 0:
        try:
            plot_results(log, save_path='simulation_results.png')
        except Exception as e:
            print(f"Could not generate plots: {e}")
    
    return final_state, log, reason


if __name__ == "__main__":
    main()
