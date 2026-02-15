"""Tests for all new physics modules: thermal, earth, rcs, flex, slosh, sensors, abort, montecarlo."""

import numpy as np
import pytest

from rlv_sim import constants as C


# ═══════════════════════════════════════════════════════════
# FORCES: J2 Gravity and Engine Transients
# ═══════════════════════════════════════════════════════════

class TestJ2Gravity:
    def test_central_gravity_unchanged_without_j2(self):
        from rlv_sim.forces import compute_gravity_force
        r = np.array([C.R_EARTH, 0.0, 0.0])
        m = 1000.0
        F_central = compute_gravity_force(r, m, enable_j2=False)
        F_no_flag = compute_gravity_force(r, m)
        np.testing.assert_array_almost_equal(F_central, F_no_flag)

    def test_j2_changes_force(self):
        from rlv_sim.forces import compute_gravity_force
        r = np.array([C.R_EARTH, 0.0, 0.0])
        m = 1000.0
        F_central = compute_gravity_force(r, m, enable_j2=False)
        F_j2 = compute_gravity_force(r, m, enable_j2=True)
        # J2 should change the force slightly
        diff = np.linalg.norm(F_j2 - F_central)
        assert diff > 0.1  # Measurable difference
        # But not by more than ~0.2%
        assert diff / np.linalg.norm(F_central) < 0.01

    def test_j2_equatorial_vs_polar(self):
        from rlv_sim.forces import compute_gravity_force
        m = 1000.0
        r_eq = np.array([C.R_EARTH, 0.0, 0.0])
        r_pole = np.array([0.0, 0.0, C.R_EARTH])
        F_eq = compute_gravity_force(r_eq, m, enable_j2=True)
        F_pole = compute_gravity_force(r_pole, m, enable_j2=True)
        # At the same geocentric radius, J2 increases equatorial gravity
        # (5*z^2/r^2 - 1 = -1 at equator → additional inward force)
        # and decreases polar gravity (5*z^2/r^2 - 3 = 2 at pole → outward correction)
        # So |F_eq| > |F_pole| at same radius
        assert np.linalg.norm(F_eq) > np.linalg.norm(F_pole)

    def test_j2_zero_position(self):
        from rlv_sim.forces import compute_gravity_force
        r = np.zeros(3)
        F = compute_gravity_force(r, 1000.0, enable_j2=True)
        np.testing.assert_array_equal(F, [0, 0, 0])


class TestEngineTransients:
    def test_spool_up_from_zero(self):
        from rlv_sim.forces import apply_engine_transient
        # Starting at 0%, commanding 100%, with 1.5s spool time
        actual = apply_engine_transient(1.0, 0.0, 0.1, spool_up_time=1.5)
        # In 0.1s with 1.5s spool: max delta = 0.1/1.5 = 0.0667
        assert 0.05 < actual < 0.08

    def test_spool_down_from_full(self):
        from rlv_sim.forces import apply_engine_transient
        actual = apply_engine_transient(0.0, 1.0, 0.1, spool_down_time=0.8)
        # In 0.1s with 0.8s spool: max delta = 0.1/0.8 = 0.125
        assert 0.87 < actual < 0.96

    def test_instant_at_steady_state(self):
        from rlv_sim.forces import apply_engine_transient
        actual = apply_engine_transient(0.5, 0.5, 0.1)
        assert actual == pytest.approx(0.5)

    def test_clamps_to_bounds(self):
        from rlv_sim.forces import apply_engine_transient
        actual = apply_engine_transient(1.5, 0.99, 10.0)  # Large dt
        assert actual <= 1.0
        actual = apply_engine_transient(-0.5, 0.01, 10.0)
        assert actual >= 0.0


# ═══════════════════════════════════════════════════════════
# THERMAL MODEL
# ═══════════════════════════════════════════════════════════

class TestThermal:
    def test_sutton_graves_zero_in_vacuum(self):
        from rlv_sim.thermal import sutton_graves_heating
        assert sutton_graves_heating(0.0, 7000.0, 1.0) == 0.0

    def test_sutton_graves_increases_with_velocity(self):
        from rlv_sim.thermal import sutton_graves_heating
        q1 = sutton_graves_heating(1.0, 1000.0, 1.0)
        q2 = sutton_graves_heating(1.0, 2000.0, 1.0)
        assert q2 > q1

    def test_radiative_equilibrium(self):
        from rlv_sim.thermal import radiative_equilibrium_temperature
        T = radiative_equilibrium_temperature(1e6, 0.85)
        assert 1000 < T < 5000  # Reasonable range

    def test_thermal_state_update(self):
        from rlv_sim.thermal import ThermalState
        ts = ThermalState()
        result = ts.update(30000.0, 3000.0, 0.01)
        assert 'q_dot' in result
        assert 'T_wall' in result
        assert result['q_dot'] >= 0


# ═══════════════════════════════════════════════════════════
# EARTH MODEL (WGS-84)
# ═══════════════════════════════════════════════════════════

class TestEarth:
    def test_eci_ecef_roundtrip(self):
        from rlv_sim.earth import eci_to_ecef, ecef_to_eci
        r = np.array([7e6, 1e6, 0.5e6])
        t = 1000.0
        r_ecef = eci_to_ecef(r, t)
        r_back = ecef_to_eci(r_ecef, t)
        np.testing.assert_array_almost_equal(r, r_back, decimal=3)

    def test_geodetic_ecef_roundtrip(self):
        from rlv_sim.earth import ecef_to_geodetic, geodetic_to_ecef
        r_ecef = np.array([6378137.0, 0.0, 0.0])  # On equator
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        assert abs(lat) < 1e-6
        assert abs(alt) < 1.0  # Should be ~0 m

        r_back = geodetic_to_ecef(lat, lon, alt)
        np.testing.assert_array_almost_equal(r_ecef, r_back, decimal=1)

    def test_wgs84_altitude_at_equator(self):
        from rlv_sim.earth import wgs84_surface_altitude
        r_eci = np.array([6378137.0 + 100000.0, 0.0, 0.0])
        alt = wgs84_surface_altitude(r_eci, t=0.0)
        assert 99000 < alt < 101000  # ~100 km

    def test_geodetic_at_pole(self):
        from rlv_sim.earth import ecef_to_geodetic
        # North pole
        r_pole = np.array([0.0, 0.0, 6356752.0])  # Approx semi-minor axis
        lat, lon, alt = ecef_to_geodetic(r_pole)
        assert abs(lat - np.pi/2) < 0.01  # Should be ~90 deg


# ═══════════════════════════════════════════════════════════
# RCS
# ═══════════════════════════════════════════════════════════

class TestRCS:
    def test_no_fire_in_deadband(self):
        from rlv_sim.rcs import RCSState, compute_rcs_torque
        rcs = RCSState(propellant_mass=100.0)
        omega_err = np.array([0.0005, 0.0, 0.0])  # Below deadband
        torque, rcs = compute_rcs_torque(omega_err, rcs, dt=0.1, deadband=0.001)
        np.testing.assert_array_equal(torque, [0, 0, 0])

    def test_fires_above_deadband(self):
        from rlv_sim.rcs import RCSState, compute_rcs_torque
        rcs = RCSState(propellant_mass=100.0)
        omega_err = np.array([0.1, 0.0, 0.0])
        torque, rcs = compute_rcs_torque(omega_err, rcs, dt=0.1)
        assert torque[0] < 0  # Opposes positive rate
        assert abs(torque[0]) > 100  # Meaningful torque

    def test_propellant_consumed(self):
        from rlv_sim.rcs import RCSState, compute_rcs_torque
        rcs = RCSState(propellant_mass=100.0)
        initial = rcs.propellant_mass
        omega_err = np.array([0.1, 0.1, 0.1])
        _, rcs = compute_rcs_torque(omega_err, rcs, dt=1.0)
        assert rcs.propellant_mass < initial

    def test_no_fire_when_empty(self):
        from rlv_sim.rcs import RCSState, compute_rcs_torque
        rcs = RCSState(propellant_mass=0.0)
        omega_err = np.array([1.0, 1.0, 1.0])
        torque, _ = compute_rcs_torque(omega_err, rcs, dt=0.1)
        np.testing.assert_array_equal(torque, [0, 0, 0])


# ═══════════════════════════════════════════════════════════
# FLEX BODY
# ═══════════════════════════════════════════════════════════

class TestFlexBody:
    def test_zero_forcing_stays_zero(self):
        from rlv_sim.flex import FlexState
        fs = FlexState()
        result = fs.update(gimbal_angle=0.0, thrust_force=0.0,
                          generalised_mass=1000.0, dt=0.01)
        assert abs(result['eta']) < 1e-10

    def test_forcing_creates_displacement(self):
        from rlv_sim.flex import FlexState
        fs = FlexState()
        # Apply constant forcing for multiple steps
        for _ in range(100):
            result = fs.update(gimbal_angle=0.01, thrust_force=1e6,
                              generalised_mass=5000.0, dt=0.01)
        assert abs(result['eta']) > 1e-6

    def test_parasitic_torque_shape(self):
        from rlv_sim.flex import FlexState
        fs = FlexState()
        fs.eta = 0.01  # Manually set displacement
        torque = fs.get_parasitic_torque(inertia=1e7)
        assert torque.shape == (3,)
        assert abs(torque[1]) > 0  # Y-axis (pitch)


# ═══════════════════════════════════════════════════════════
# SLOSH
# ═══════════════════════════════════════════════════════════

class TestSlosh:
    def test_no_slosh_with_no_propellant(self):
        from rlv_sim.slosh import SloshState
        ss = SloshState()
        result = ss.update(lateral_accel=10.0, propellant_mass=0.0, dt=0.01)
        assert result['slosh_mass'] == 0.0
        assert abs(result['x_slosh']) < 1e-10

    def test_lateral_accel_causes_displacement(self):
        from rlv_sim.slosh import SloshState
        ss = SloshState()
        for _ in range(100):
            result = ss.update(lateral_accel=5.0, propellant_mass=100000.0, dt=0.01)
        assert abs(result['x_slosh']) > 0.001

    def test_slosh_torque_shape(self):
        from rlv_sim.slosh import SloshState
        ss = SloshState()
        result = ss.update(lateral_accel=5.0, propellant_mass=100000.0, dt=0.01)
        assert result['slosh_torque'].shape == (3,)


# ═══════════════════════════════════════════════════════════
# SENSORS
# ═══════════════════════════════════════════════════════════

class TestSensors:
    def test_imu_bias_present(self):
        from rlv_sim.sensors import IMUState
        imu = IMUState(seed=42, accel_bias_sigma=0.1)
        assert np.linalg.norm(imu.accel_bias) > 0

    def test_acceleration_measurement_noisy(self):
        from rlv_sim.sensors import IMUState
        imu = IMUState(seed=42)
        true_accel = np.array([0.0, 0.0, 9.81])
        meas = imu.measure_acceleration(true_accel, dt=0.01)
        # Should be close but not exactly equal
        assert not np.allclose(meas, true_accel)
        # Should be within reasonable bounds
        assert np.linalg.norm(meas - true_accel) < 1.0

    def test_gyro_measurement_noisy(self):
        from rlv_sim.sensors import IMUState
        imu = IMUState(seed=42)
        true_omega = np.zeros(3)
        meas = imu.measure_angular_rate(true_omega, dt=0.01)
        assert not np.allclose(meas, true_omega)
        assert np.linalg.norm(meas) < 0.1  # Should be small


# ═══════════════════════════════════════════════════════════
# ABORT MONITOR
# ═══════════════════════════════════════════════════════════

class TestAbortMonitor:
    def test_no_abort_in_nominal(self):
        from rlv_sim.abort import AbortMonitor
        am = AbortMonitor()
        result = am.check(
            r=np.array([C.R_EARTH + 50000.0, 0, 0]),
            v=np.array([0, 1000.0, 0]),
            q=np.array([1, 0, 0, 0]),
            omega=np.zeros(3),
            t=60.0
        )
        assert result['abort'] is False

    def test_tumble_triggers_abort(self):
        from rlv_sim.abort import AbortMonitor
        am = AbortMonitor(max_tumble_rate=0.5)
        result = am.check(
            r=np.array([C.R_EARTH + 20000.0, 0, 0]),
            v=np.array([0, 500.0, 0]),
            q=np.array([1, 0, 0, 0]),
            omega=np.array([1.0, 1.0, 0.0]),  # High tumble
            t=60.0
        )
        assert result['abort'] is True
        assert 'Tumble' in result['reason']

    def test_abort_mode_rtls_low_altitude(self):
        from rlv_sim.abort import AbortMonitor
        am = AbortMonitor(max_tumble_rate=0.1)
        result = am.check(
            r=np.array([C.R_EARTH + 20000.0, 0, 0]),
            v=np.array([0, 500.0, 0]),
            q=np.array([1, 0, 0, 0]),
            omega=np.array([0.5, 0.0, 0.0]),
            t=60.0
        )
        assert result['mode'] == 'RTLS'


# ═══════════════════════════════════════════════════════════
# MONTE CARLO (minimal test — runs 2 iterations)
# ═══════════════════════════════════════════════════════════

class TestMonteCarlo:
    def test_mc_runs_and_returns_results(self):
        from rlv_sim.montecarlo import run_monte_carlo
        from rlv_sim.config import create_test_config
        cfg = create_test_config(dt=0.1, max_time=5.0)
        results = run_monte_carlo(base_config=cfg, n_runs=2, verbose=False)
        assert results.n_runs == 2
        assert results.wall_time_s > 0

    def test_mc_summary_string(self):
        from rlv_sim.montecarlo import run_monte_carlo
        from rlv_sim.config import create_test_config
        cfg = create_test_config(dt=0.1, max_time=5.0)
        results = run_monte_carlo(base_config=cfg, n_runs=2, verbose=False)
        summary = results.summary()
        assert 'Monte Carlo' in summary
        assert '2 runs' in summary
