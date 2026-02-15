"""
Mission audit report for full RLV mission (ascent + orbiter + booster).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlv_sim import constants as C
from rlv_sim.config import create_default_config
from rlv_sim.main import run_full_mission
from rlv_sim.utils import compute_relative_velocity
from rlv_sim.recovery import target_landing_site_eci, great_circle_distance_m


def _phase_transitions(times, phases, alts, vels):
    out = []
    prev = None
    for t, p, h, v in zip(times, phases, alts, vels):
        if p != prev:
            out.append((float(t), str(p), float(h), float(v)))
            prev = p
    return out


def _phase_propellant_usage(times, phases, masses):
    usage = OrderedDict()
    if len(times) < 2:
        return usage

    start_idx = 0
    for i in range(1, len(times)):
        if phases[i] != phases[start_idx]:
            p = str(phases[start_idx])
            burned = max(0.0, float(masses[start_idx] - masses[i - 1]))
            usage[p] = usage.get(p, 0.0) + burned
            start_idx = i

    p = str(phases[start_idx])
    burned = max(0.0, float(masses[start_idx] - masses[-1]))
    usage[p] = usage.get(p, 0.0) + burned
    return usage


def _orbital_elements(state):
    r = np.asarray(state.r, dtype=float)
    v = np.asarray(state.v, dtype=float)
    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))
    energy = 0.5 * v_mag * v_mag - C.MU_EARTH / r_mag
    a = -C.MU_EARTH / (2.0 * energy) if abs(energy) > 1e-9 else float("inf")
    h = np.cross(r, v)
    h_mag = float(np.linalg.norm(h))
    if a > 0 and np.isfinite(a):
        ecc = np.sqrt(max(0.0, 1.0 - h_mag * h_mag / (C.MU_EARTH * a)))
        perigee = a * (1.0 - ecc) - C.R_EARTH
        apogee = a * (1.0 + ecc) - C.R_EARTH
    else:
        ecc = float("nan")
        perigee = float("nan")
        apogee = float("nan")
    return {
        "a_km": a / 1000.0,
        "ecc": ecc,
        "perigee_km": perigee / 1000.0,
        "apogee_km": apogee / 1000.0,
        "speed_mps": v_mag,
    }


def main():
    parser = argparse.ArgumentParser(description="Run full mission and print physics audit summary.")
    parser.add_argument("--dt", type=float, default=0.05, help="Integration timestep (s)")
    parser.add_argument("--max-time", type=float, default=1200.0, help="Maximum mission time (s)")
    args = parser.parse_args()

    cfg = create_default_config()
    result = run_full_mission(dt=args.dt, max_time=args.max_time, verbose=False, config=cfg)

    print("=" * 88)
    print("RLV MISSION AUDIT")
    print("=" * 88)
    print(f"Separation time: {result.separation_time:.2f} s")
    print(f"Ascent reason:   {result.ascent_reason}")
    print(f"Orbiter reason:  {result.orbiter_reason}")
    print(f"Booster reason:  {result.booster_reason}")
    print("-" * 88)

    # Booster transitions and prop usage
    bt = np.array(result.booster_log.time)
    bp = np.array(result.booster_log.phase_name)
    bh = np.array(result.booster_log.altitude)
    bv = np.array(result.booster_log.velocity)
    bm = np.array(result.booster_log.mass)

    if len(bt) > 0:
        print("Booster Phase Transitions:")
        for t, p, h, v in _phase_transitions(bt, bp, bh, bv):
            print(f"  t={t:8.2f}s | {p:16s} | alt={h:8.2f} km | v={v:8.1f} m/s")
        print("-" * 88)
        print("Booster Propellant Usage by Phase:")
        usage = _phase_propellant_usage(bt, bp, bm)
        for phase, burned in usage.items():
            print(f"  {phase:16s} : {burned:9.1f} kg")
        print("-" * 88)
        peak_idx = int(np.argmax(bh))
        print(f"Booster apogee:      {bh[peak_idx]:.2f} km at t={bt[peak_idx]:.2f}s")
        v_rel = compute_relative_velocity(result.booster_final_state.r, result.booster_final_state.v)
        target_site = target_landing_site_eci(
            result.booster_final_state.t,
            cfg.booster_landing_target_downrange_km,
        )
        miss = great_circle_distance_m(result.booster_final_state.r, target_site)
        print(f"Booster final speed (ECI): {result.booster_final_state.speed:.1f} m/s")
        print(f"Booster touchdown speed:   {np.linalg.norm(v_rel):.2f} m/s (air-relative)")
        print(f"Booster site miss distance:{miss/1000.0:8.2f} km")
        print(f"Booster final mass:  {result.booster_final_state.m:.1f} kg")
    else:
        print("Booster telemetry unavailable (no separation).")

    print("-" * 88)
    orb = _orbital_elements(result.orbiter_final_state)
    print("Final Orbiter Elements:")
    print(f"  a      = {orb['a_km']:.2f} km")
    print(f"  e      = {orb['ecc']:.5f}")
    print(f"  perigee= {orb['perigee_km']:.2f} km")
    print(f"  apogee = {orb['apogee_km']:.2f} km")
    print(f"  speed  = {orb['speed_mps']:.1f} m/s")
    print("=" * 88)


if __name__ == "__main__":
    main()
