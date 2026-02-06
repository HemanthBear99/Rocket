
import pandas as pd
import numpy as np

def verify():
    print("Verifying Phase II Implementation...")
    
    # Load Data
    try:
        baseline = pd.read_csv('plots/baseline_telemetry.csv')
        phase2 = pd.read_csv('plots/phase2_telemetry.csv')
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # 1. Baseline Regression Check (Phase I)
    # Compare rows where Time <= MECO of baseline (approx 200s or whenever it stopped)
    # Actually, baseline only ran to MECO/Apogee? Baseline old code stopped at MECO/Coast end.
    # Let's compare the first 1000 steps or until mass < 150000
    
    print(f"Baseline rows: {len(baseline)}")
    print(f"Phase2 rows: {len(phase2)}")
    
    # Common Time Index
    time_limit = min(baseline['time'].max(), phase2['time'].max())
    # Truncate to common time for direct comparison if needed, 
    # but Phase 1 should be identical step-for-step if DT is constant.
    
    limit_step = min(len(baseline), len(phase2))
    diff_cols = ['altitude_km', 'velocity_inertial', 'mass']
    
    # Check absolute difference for first N steps (Ascent)
    # We expect perfect match until MECO detection logic potentially shifts by 1 step
    ascent_steps = 1000 # Parametric check
    if len(baseline) > ascent_steps and len(phase2) > ascent_steps:
        diff = np.abs(baseline.iloc[:ascent_steps][diff_cols] - phase2.iloc[:ascent_steps][diff_cols])
        max_diff = diff.max().max()
        print(f"Regresion Check (First {ascent_steps} steps): Max Diff = {max_diff}")
        if max_diff > 1e-6:
            print("WARNING: Significant regression in Ascent Phase!")
        else:
            print("PASS: Ascent Phase matches baseline.")

    # 2. Phase II Physics Check
    # Check steps where thrust_on == 0 (Coast)
    coast_data = phase2[phase2['thrust_on'] == 0]
    
    if len(coast_data) == 0:
        print("FAIL: No Coast Phase data found (thrust_on never 0).")
    else:
        print(f"PASS: Found {len(coast_data)} steps of Coast Phase.")
        
        # Verify Thrust is Zero
        max_thrust = coast_data[['inertial_thrust_x', 'inertial_thrust_y', 'inertial_thrust_z']].abs().max().max()
        if max_thrust > 1e-3:
             print(f"FAIL: Non-zero inertial thrust detected in Coast Phase: {max_thrust} N")
        else:
             print("PASS: Inertial Thrust is zero during Coast.")
             
        # Verify Mass Constant
        mass_std = coast_data['mass'].std()
        if mass_std > 1e-6:
             print(f"FAIL: Mass is changing during Coast! StdDev={mass_std}")
        else:
             print("PASS: Mass is constant during Coast.")
             
        # Verify Apogee Detection
        # Check if last point has vertical velocity ~ 0 or negative
        last_row = phase2.iloc[-1]
        v_vert = last_row['vel_vert']
        print(f"Termination Vertical Velocity: {v_vert:.2f} m/s")
        if v_vert <= 0.5:
             print("PASS: Simulation terminated near Apogee (v_vert <= 0).")
        else:
             print("WARNING: Simulation terminated with positive vertical velocity (did not reach apogee?)")

if __name__ == "__main__":
    verify()
