"""Regression coverage for the booster/orbiter physics-corrective plan."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from rlv_sim import constants as C
from rlv_sim.config import create_default_config, create_test_config
from rlv_sim.forces import compute_atmosphere_properties
from rlv_sim.main import run_full_mission, check_termination
from rlv_sim.mass import compute_center_of_mass, compute_inertia_tensor
from rlv_sim.mission_manager import MissionManager, MissionPhase
from rlv_sim.recovery import (
    booster_min_propellant_after_boostback,
    estimate_suicide_burn,
    rotating_launch_site_eci,
    target_landing_site_eci,
    great_circle_distance_m,
)
from rlv_sim.state import State
from rlv_sim.utils import compute_relative_velocity


def _orbital_elements(state):
    r = np.asarray(state.r, dtype=float)
    v = np.asarray(state.v, dtype=float)
    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))
    energy = 0.5 * v_mag * v_mag - C.MU_EARTH / r_mag
    a = -C.MU_EARTH / (2.0 * energy)
    h = np.cross(r, v)
    h_mag = float(np.linalg.norm(h))
    ecc = float(np.sqrt(max(0.0, 1.0 - h_mag * h_mag / (C.MU_EARTH * a))))
    perigee = float(a * (1.0 - ecc) - C.R_EARTH)
    apogee = float(a * (1.0 + ecc) - C.R_EARTH)
    return perigee, apogee, ecc


@lru_cache(maxsize=None)
def _mission(dt: float):
    return run_full_mission(dt=dt, verbose=False)


def test_booster_mass_properties_do_not_alias_orbiter_model():
    mass = C.STAGE1_DRY_MASS + 5000.0
    cg_booster = compute_center_of_mass(mass, vehicle_model="booster")
    cg_orbiter = compute_center_of_mass(mass, vehicle_model="orbiter")
    i_booster = compute_inertia_tensor(mass, vehicle_model="booster")
    i_orbiter = compute_inertia_tensor(mass, vehicle_model="orbiter")

    assert cg_booster < C.H_STAGE1
    assert cg_orbiter > C.H_STAGE1
    assert not np.allclose(i_booster, i_orbiter)


def test_ignition_model_is_consistent_between_manager_and_guidance_path():
    cfg = create_default_config()
    state = State(
        r=np.array([C.R_EARTH + 1200.0, 0.0, 0.0], dtype=float),
        v=np.array([-140.0, 220.0, 0.0], dtype=float),
        q=C.INITIAL_QUATERNION.copy(),
        omega=np.zeros(3),
        m=C.STAGE1_DRY_MASS + 9000.0,
        t=500.0,
    )
    burn = estimate_suicide_burn(
        state.r,
        state.v,
        state.m,
        C.THRUST_MAGNITUDE,
        safety_factor=cfg.booster_landing_ignition_safety_factor,
    )

    mgr = MissionManager(
        vehicle_type="booster",
        initial_mass=C.STAGE1_DRY_MASS + C.STAGE1_LANDING_FUEL_RESERVE,
        config=cfg,
    )
    mgr.current_phase = MissionPhase.BOOSTER_ENTRY
    mgr.update(state, dt=0.1)

    transitioned = mgr.get_phase() == MissionPhase.BOOSTER_LANDING

    # The manager now has two triggers:
    #   (a) energy-based ignition estimate (original behaviour)
    #   (b) altitude-floor override so ZEM/ZEV has enough time to divert to pad
    # Both should cause a transition; neither alone is sufficient to *prevent* one.
    r_norm = float(np.linalg.norm(state.r))
    radial_vel = float(np.dot(state.r, state.v) / r_norm)
    altitude = r_norm - C.R_EARTH
    altitude_floor_trigger = (
        altitude < cfg.booster_landing_min_altitude_m and radial_vel < 0.0
    )
    expected = bool(burn["ignite"]) or altitude_floor_trigger
    assert transitioned == expected


def test_boostback_reserve_guard_protects_entry_and_landing_budget():
    cfg = create_default_config()
    mgr = MissionManager(vehicle_type="booster", initial_mass=60000.0, config=cfg)
    mgr.current_phase = MissionPhase.BOOSTER_BOOSTBACK

    min_after_boostback = booster_min_propellant_after_boostback(cfg)
    state = State(
        r=np.array([C.R_EARTH + 90000.0, 0.0, 0.0], dtype=float),
        v=np.array([200.0, 1400.0, 0.0], dtype=float),
        q=C.INITIAL_QUATERNION.copy(),
        omega=np.zeros(3),
        m=C.STAGE1_DRY_MASS + min_after_boostback - 1.0,
        t=145.0,
    )
    mgr.update(state, dt=0.1)
    assert mgr.get_phase() == MissionPhase.BOOSTER_COAST


def test_nominal_full_mission_meets_soft_landing_contract():
    result = _mission(0.05)
    touchdown_v = float(
        np.linalg.norm(
            compute_relative_velocity(result.booster_final_state.r, result.booster_final_state.v)
        )
    )
    assert "CRASH" not in result.booster_reason
    assert touchdown_v < 5.0
    assert abs(result.booster_final_state.altitude) < 50.0


def test_strict_orbit_window_contract():
    cfg = create_default_config()
    result = _mission(0.05)
    perigee, apogee, ecc = _orbital_elements(result.orbiter_final_state)
    target = cfg.orbit_target_altitude_m
    tol = cfg.orbit_altitude_tolerance_m

    assert abs(perigee - target) <= tol
    assert abs(apogee - target) <= tol
    assert ecc <= cfg.orbit_ecc_max


def test_us76_layer_checkpoints():
    checkpoints = {
        0.0: (288.15, 101325.0),
        11000.0: (216.65, 22631.7),
        20000.0: (216.65, 5474.7),
        32000.0: (228.65, 867.97),
        47000.0: (270.65, 110.90),
        51000.0: (270.65, 66.93),
        71000.0: (214.65, 3.956),
        84852.0: (186.946, 0.3733),
    }
    for altitude_m, (t_ref, p_ref) in checkpoints.items():
        t, p, _, _ = compute_atmosphere_properties(altitude_m)
        assert abs(t - t_ref) <= 0.25
        # Pressure tolerance is looser at very high altitude where values are tiny.
        assert abs(p - p_ref) <= max(0.5, 0.02 * p_ref)


def test_dt_robustness_spread_is_bounded():
    dts = (0.1, 0.05, 0.02)
    perigees = []
    apogees = []
    touchdown_speeds = []

    for dt in dts:
        result = _mission(dt)
        perigee, apogee, _ = _orbital_elements(result.orbiter_final_state)
        perigees.append(perigee / 1000.0)
        apogees.append(apogee / 1000.0)
        touchdown_speeds.append(
            float(np.linalg.norm(compute_relative_velocity(result.booster_final_state.r, result.booster_final_state.v)))
        )

    assert max(perigees) - min(perigees) < 25.0
    assert max(apogees) - min(apogees) < 25.0
    assert max(touchdown_speeds) <= 10.0


def test_target_landing_site_geometry_matches_offset():
    t = 400.0
    launch = rotating_launch_site_eci(t)
    target = target_landing_site_eci(t, downrange_km=9.65)
    dist = great_circle_distance_m(launch, target)
    assert abs(dist - 9650.0) < 2.0


def test_pad_enforcement_marks_offsite_touchdown():
    cfg = create_test_config(
        dt=0.1,
        max_time=10.0,
        booster_enforce_pad_landing=True,
        booster_pad_tolerance_m=100.0,
        booster_landing_target_downrange_km=9.65,
    )
    mgr = MissionManager(vehicle_type="booster", initial_mass=60000.0, config=cfg)
    mgr.current_phase = MissionPhase.BOOSTER_LANDING

    t = 100.0
    target = target_landing_site_eci(t, cfg.booster_landing_target_downrange_km)
    # Place vehicle 2 km offsite along local-east arc while keeping zero air-relative speed.
    r_hat = target / np.linalg.norm(target)
    k_axis = np.array([0.0, 0.0, 1.0])
    east = np.cross(k_axis, r_hat)
    east /= np.linalg.norm(east)
    dtheta = 2000.0 / C.R_EARTH
    r_off = C.R_EARTH * (np.cos(dtheta) * r_hat + np.sin(dtheta) * east)
    omega_e = np.array([0.0, 0.0, C.EARTH_ROTATION_RATE])
    v_ground = np.cross(omega_e, r_off)

    state = State(
        r=r_off,
        v=v_ground,
        q=C.INITIAL_QUATERNION.copy(),
        omega=np.zeros(3),
        m=C.STAGE1_DRY_MASS + 5000.0,
        t=t,
    )
    done, reason = check_termination(state, max_time=1000.0, mission_mgr=mgr)
    assert done is True
    assert "OFFSITE LANDING" in reason
