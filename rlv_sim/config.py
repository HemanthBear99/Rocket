"""
RLV Phase-I Ascent Simulation - Configuration

This module provides a SimulationConfig dataclass for dependency injection,
allowing different simulation parameters to be passed without modifying
global constants.

All new physics features default to OFF (False) so existing behaviour is
unchanged until the caller explicitly enables them.
"""

from dataclasses import dataclass, field

from . import constants as C


@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable configuration for simulation parameters.

    Using frozen=True ensures configs cannot be accidentally modified.
    Create new configs via dataclass replace() if needed.

    Section grouping:
      1. Simulation timing
      2. Control gains
      3. Guidance parameters
      4. Physics feature toggles (J2, engine transients, Q-alpha, ...)
      5. Atmosphere / wind
      6. Thermal
      7. Earth model
      8. RCS / flex / slosh
      9. Sensor / nav / abort
     10. Monte Carlo
     11. Tolerances
     12. Misc
    """

    # ── 1. Simulation timing ─────────────────────────────────────────────
    dt: float = C.DT
    max_time: float = C.MAX_TIME

    # ── 2. Control gains ─────────────────────────────────────────────────
    kp_attitude: float = C.KP_ATTITUDE
    kd_attitude: float = C.KD_ATTITUDE
    max_torque: float = C.MAX_TORQUE

    # ── 3. Guidance parameters ───────────────────────────────────────────
    gravity_turn_start_altitude: float = C.GRAVITY_TURN_START_ALTITUDE
    gravity_turn_transition_range: float = C.GRAVITY_TURN_TRANSITION_RANGE
    min_velocity_for_turn: float = C.MIN_VELOCITY_FOR_TURN
    pitchover_start_altitude: float = C.PITCHOVER_START_ALTITUDE
    pitchover_end_altitude: float = C.PITCHOVER_END_ALTITUDE
    pitchover_angle: float = C.PITCHOVER_ANGLE
    booster_boostback_budget_kg: float = 12000.0
    booster_entry_budget_kg: float = 8000.0
    booster_landing_reserve_kg: float = 10000.0
    booster_apogee_target_km: float = 160.0
    booster_entry_interface_altitude_m: float = 70000.0
    booster_entry_burn_min_altitude_m: float = 35000.0
    booster_entry_burn_min_speed_mps: float = 800.0
    booster_landing_ignition_safety_factor: float = 1.5
    # RTLS target site measured from launch site along local-east surface arc.
    # Cape Canaveral references:
    #   SLC-40 -> LZ-1/2 ~= 9.65 km
    #   LC-39A -> LZ-1/2 ~= 14.5 km
    #   Direct pad landing = 0.0 km
    booster_landing_target_downrange_km: float = 9.65
    booster_pad_tolerance_m: float = 100.0
    booster_enforce_pad_landing: bool = False
    orbit_target_altitude_m: float = C.TARGET_ORBIT_ALTITUDE
    orbit_altitude_tolerance_m: float = 25000.0
    orbit_ecc_max: float = 0.01

    # ── 4. Physics feature toggles ───────────────────────────────────────
    # J2 oblateness perturbation (default OFF → central gravity only)
    enable_j2: bool = False
    j2_coefficient: float = 1.08263e-3  # J2 for Earth

    # Engine transient modelling (spool-up / spool-down ramps)
    enable_engine_transients: bool = False
    engine_spool_up_time: float = 1.5    # seconds to reach 100 % thrust
    engine_spool_down_time: float = 0.8  # seconds to reach 0 % thrust

    # Q-alpha structural load monitoring / limiting
    enable_q_alpha_limit: bool = False
    q_alpha_max: float = 50000.0  # Pa·rad (structural limit)

    # Separation dynamics (delta-V kick, tumble rate)
    enable_separation_dynamics: bool = False
    separation_delta_v: float = 1.5           # m/s separation impulse
    separation_tumble_rate: float = 0.02      # rad/s residual tumble

    # Fairing separation
    enable_fairing_separation: bool = False
    fairing_mass: float = 2000.0              # kg (total both halves)
    fairing_separation_altitude: float = 110000.0  # m

    # ── 5. Atmosphere / wind ─────────────────────────────────────────────
    # Upper atmosphere model (NRLMSISE-00 above 86 km, default OFF → US76)
    atmosphere_model: str = "us76_deterministic"
    enable_upper_atmosphere: bool = False

    # Stochastic wind (turbulence / gusts layered on mean profile)
    enable_stochastic_wind: bool = False
    wind_turbulence_intensity: float = 2.0  # m/s RMS
    wind_gust_magnitude: float = 15.0       # m/s peak discrete gust

    # ── 6. Thermal ───────────────────────────────────────────────────────
    enable_thermal: bool = False
    sutton_graves_k: float = 1.7415e-4  # Sutton-Graves constant for N2/O2
    emissivity: float = 0.85            # surface emissivity
    wall_thickness: float = 0.005       # m (TPS / skin)
    material_cp: float = 900.0          # J/(kg·K) specific heat (Al-alloy)
    material_density: float = 2700.0    # kg/m³

    # ── 7. Earth model ───────────────────────────────────────────────────
    enable_wgs84: bool = False
    # WGS-84 semi-major axis & flattening (used only if enable_wgs84=True)
    wgs84_a: float = 6378137.0          # m
    wgs84_f: float = 1.0 / 298.257223563

    # ── 8. Structural & propellant dynamics ──────────────────────────────
    # RCS (reaction control system)
    enable_rcs: bool = False
    rcs_thrust: float = 500.0           # N per thruster
    rcs_isp: float = 220.0              # s
    rcs_num_thrusters: int = 8

    # Flexible body first bending mode
    enable_flex_body: bool = False
    flex_frequency_hz: float = 2.5      # Hz (first bending mode)
    flex_damping_ratio: float = 0.02    # structural damping ratio
    flex_mode_slope: float = 0.1        # rad/m (mode shape slope at gimbal)

    # Propellant slosh (pendulum model)
    enable_slosh: bool = False
    slosh_mass_fraction: float = 0.20   # fraction of propellant that sloshes
    slosh_frequency_hz: float = 0.3     # Hz (pendulum natural frequency)
    slosh_damping_ratio: float = 0.01   # baffle damping

    # ── 9. Navigation sensors & abort ────────────────────────────────────
    enable_nav_sensors: bool = False
    imu_accel_bias: float = 0.001       # m/s² (1-sigma)
    imu_accel_noise: float = 0.01       # m/s²/√Hz
    imu_gyro_bias: float = 1e-5         # rad/s (1-sigma)
    imu_gyro_noise: float = 1e-4        # rad/s/√Hz

    enable_abort_modes: bool = False
    abort_q_alpha_threshold: float = 60000.0  # Pa·rad
    abort_attitude_threshold: float = 30.0     # degrees

    # ── 10. Monte Carlo ──────────────────────────────────────────────────
    enable_monte_carlo: bool = False
    monte_carlo_runs: int = 100
    monte_carlo_seed: int = 42
    # Dispersion magnitudes (1-sigma)
    mc_thrust_dispersion: float = 0.02        # fraction
    mc_isp_dispersion: float = 0.01           # fraction
    mc_mass_dispersion: float = 0.005         # fraction
    mc_wind_dispersion: float = 5.0           # m/s
    mc_cg_offset: float = 0.01               # m
    mc_thrust_misalignment: float = 0.001    # rad

    # ── 11. Tolerances ───────────────────────────────────────────────────
    quaternion_norm_tolerance: float = C.QUATERNION_NORM_TOL
    zero_tolerance: float = 1e-10

    # ── 12. Misc ─────────────────────────────────────────────────────────
    verbose: bool = True


def create_default_config() -> SimulationConfig:
    """Create a SimulationConfig with default values from constants."""
    return SimulationConfig()


def create_test_config(dt: float = 0.1, max_time: float = 10.0,
                       **overrides) -> SimulationConfig:
    """Create a fast config suitable for testing.

    Any keyword arg accepted by SimulationConfig can be passed as an override.
    """
    defaults = dict(dt=dt, max_time=max_time, verbose=False)
    defaults.update(overrides)
    return SimulationConfig(**defaults)
