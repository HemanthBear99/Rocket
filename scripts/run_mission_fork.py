
import sys
import logging
import copy
import numpy as np

# Add parent to path
import sys
import os
sys.path.append(os.getcwd())

from rlv_sim.main import run_simulation
from rlv_sim.state import State
from rlv_sim import constants as C
from rlv_sim.mission_manager import MissionPhase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MissionControl")

def run_multi_mission():
    print("="*60)
    print("RLV MISSION: FORK STRATEGY (Reusable Booster)")
    print("="*60)
    
    # -------------------------------------------------------------
    # PHASE I: ASCENT TO SEPARATION
    # -------------------------------------------------------------
    print("\n[PHASE I-II] Launch to Stage Separation...")
    
    # We want to stop exactly at separation.
    # Current main.py doesn't support "stop_at_phase", so we run normally 
    # and rely on the log to find the separation event, OR we modify main for custom stopping.
    # Hack: Run full coastal simulation, but we know separation is MECO + 3s.
    
    # Let's run simulation with "coast_to_apogee=True"
    final_state, log, reason = run_simulation(verbose=False, coast_to_apogee=True)
    
    # Find Separation Index (MECO + 3s)
    # 1. Find MECO time
    meco_indices = [i for i, m in enumerate(log.mass) if m <= (C.DRY_MASS + 100.0)] # Approx check
    if not meco_indices:
        print("ERROR: MECO not reached.")
        return
        
    meco_idx = meco_indices[0]
    meco_time = log.time[meco_idx]
    separation_time = meco_time + 3.0
    
    print(f"MECO Detected at t={meco_time:.2f}s")
    print(f"Planned Separation at t={separation_time:.2f}s")
    
    # Find active state at t=separation_time
    sep_idx = next((i for i, t in enumerate(log.time) if t >= separation_time), -1)
    if sep_idx == -1:
        print("ERROR: Simulation ended before separation time.")
        return

    # Extract Separation State
    sep_state_vec = [
        log.position_x[sep_idx], log.position_y[sep_idx], log.position_z[sep_idx],
        log.velocity_x[sep_idx], log.velocity_y[sep_idx], log.velocity_z[sep_idx],
        log.actual_quat_w[sep_idx], log.actual_quat_x[sep_idx], log.actual_quat_y[sep_idx], log.actual_quat_z[sep_idx],
        log.omega_x[sep_idx], log.omega_y[sep_idx], log.omega_z[sep_idx],
        log.mass[sep_idx]
    ]
    sep_state = State.from_vector(np.array(sep_state_vec), log.time[sep_idx])
    
    print(f"\n[EVENT] STAGE SEPARATION at Alt={sep_state.altitude/1000:.1f}km, V={sep_state.speed:.1f}m/s")
    
    # -------------------------------------------------------------
    # FORK A: ORBITER (Continues to Space)
    # -------------------------------------------------------------
    print("\n" + "-"*40)
    print("BRANCH A: ORBITER MISSION (Stage 2)")
    print("-"*40)
    
    orbiter_state = sep_state.copy()
    # PHYSICS: Drop Stage 1 Mass (30t)
    # Currently mass is 150t (30t S1 + 120t S2). Orbiter is 120t.
    orbiter_state.m -= C.STAGE1_DRY_MASS 
    print(f"Orbiter Mass: {orbiter_state.m/1000:.1f}t")
    
    # Run Orbiter Simulation (Coast to Apogee for now, later Insertion)
    # We pass this new light state into the sim
    orbiter_final, orbiter_log, reason = run_simulation(initial_state=orbiter_state, verbose=False, coast_to_apogee=True, vehicle_type="orbiter")
    
    print(f"Orbiter Status: {reason}")
    print(f"Orbiter Apogee: {orbiter_final.altitude/1000:.1f} km")
    
    # Save Orbiter Data
    if hasattr(orbiter_log, 'to_csv'):
        orbiter_log.to_csv('plots/telemetry_branch_a_orbiter.csv')

    # -------------------------------------------------------------
    # FORK B: BOOSTER (Returns Home)
    # -------------------------------------------------------------
    print("\n" + "-"*40)
    print("BRANCH B: BOOSTER MISSION (Stage 1)")
    print("-"*40)
    
    booster_state = sep_state.copy()
    # PHYSICS: Isolate Stage 1 Mass
    # Booster mass = Stage 1 Dry (30t) + Residual Prop (assume minimal/zero for now)
    # Actually, we dropped S1 from orbiter. Here we KEEP S1 and DROP S2 (120t).
    booster_state.m = C.STAGE1_DRY_MASS + 5000.0 # Add 5t reserve for landing
    print(f"Booster Mass: {booster_state.m/1000:.1f}t (Dry + Reserve)")
    
    # Run Booster Simulation
    # TODO: Booster needs custom guidance (Boostback, etc), currently just coasting to crash/ocean
    booster_final, booster_log, reason = run_simulation(initial_state=booster_state, verbose=False, coast_to_apogee=False, vehicle_type="booster")
    
    print(f"Booster Status: {reason}")
    print(f"Booster Final Alt: {booster_final.altitude/1000:.1f} km")
    
    # Save Booster Data
    if hasattr(booster_log, 'to_csv'):
        booster_log.to_csv('plots/telemetry_branch_b_booster.csv')
    
    print("\nMission Complete. Data saved to plots/.")

if __name__ == "__main__":
    run_multi_mission()
