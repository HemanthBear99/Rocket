"""
RLV Phase-I Ascent Simulation - Master Run Script
Runs simulation and generates all plots in one click.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlv_sim import run_simulation
from rlv_sim import constants as C
import generate_plots


def main():
    print("\n" + "="*70)
    print("RLV PHASE-I: MASTER SIMULATION RUN")
    print("="*70 + "\n")
    
    # 1. Run the simulation
    print(">> Running Simulation Physics Engine...")
    final_state, log, reason = run_simulation(verbose=True)
    
    # 2. Print Physics Summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Termination reason: {reason}")
    print(f"Final time: {final_state.t:.2f} s")
    print(f"Final altitude: {final_state.altitude/1000:.2f} km")
    print(f"Final velocity: {final_state.speed:.2f} m/s")
    print(f"Final mass: {final_state.m:.2f} kg")
    print(f"Propellant used: {C.INITIAL_MASS - final_state.m:.2f} kg")
    print("="*60 + "\n")
    
    # 3. Generate All Plots (each saved separately)
    if len(log.time) > 0:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        print(f">> Generating Plots in: {plot_dir}")
        
        try:
            generate_plots.generate_all_plots(log, final_state, plot_dir)
            
            print("\n" + "="*70)
            print("SUCCESS: Simulation and Plotting Complete.")
            print(f"Check outputs in: {plot_dir}")
            print("="*70)
            
        except Exception as e:
            print(f"\n[ERROR] Plotting failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[WARNING] No log data generated. Skipping plots.")
    
    return final_state, log, reason


if __name__ == "__main__":
    main()
