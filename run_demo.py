"""Demo script: run full mission and show booster + orbiter tracking."""
from rlv_sim.main import run_full_mission
import numpy as np

result = run_full_mission(dt=0.05, verbose=True)

print("\n\n===== BOOSTER TRACKING DETAILS =====")
blog = result.booster_log
if len(blog.time) > 0:
    times = np.array(blog.time)
    alts = np.array(blog.altitude)
    vels = np.array(blog.velocity)
    masses = np.array(blog.mass)
    phases = blog.phase_name
    print(f"Booster log entries: {len(blog.time)}")
    print(f"Time range: {times[0]:.1f}s - {times[-1]:.1f}s")
    print(f"Peak altitude: {np.max(alts):.1f} km")
    print(f"Final altitude: {alts[-1]:.1f} km")
    print(f"Final velocity: {vels[-1]:.1f} m/s")
    print(f"Final mass: {masses[-1]:.1f} kg")
    print()
    print("Booster Phase Timeline:")
    prev_phase = None
    for i in range(len(phases)):
        if phases[i] != prev_phase:
            print(f"  t={times[i]:8.1f}s | Alt={alts[i]:8.1f} km | "
                  f"V={vels[i]:8.1f} m/s | Phase: {phases[i]}")
            prev_phase = phases[i]

print()
print("===== ORBITER TRACKING DETAILS =====")
olog = result.orbiter_log
if len(olog.time) > 0:
    times = np.array(olog.time)
    alts = np.array(olog.altitude)
    vels = np.array(olog.velocity)
    phases = olog.phase_name
    print(f"Orbiter log entries: {len(olog.time)}")
    print(f"Time range: {times[0]:.1f}s - {times[-1]:.1f}s")
    print(f"Peak altitude: {np.max(alts):.1f} km")
    print(f"Final velocity: {vels[-1]:.1f} m/s")
    print()
    print("Orbiter Phase Timeline:")
    prev_phase = None
    for i in range(len(phases)):
        if phases[i] != prev_phase:
            print(f"  t={times[i]:8.1f}s | Alt={alts[i]:8.1f} km | "
                  f"V={vels[i]:8.1f} m/s | Phase: {phases[i]}")
            prev_phase = phases[i]
