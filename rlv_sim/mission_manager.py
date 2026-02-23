"""
RLV Mission Manager

This module handles the high-level state machine for the mission.
It defines discrete mission phases and handles the transition logic between them.

All transitions are physics-based:
  - MECO:             propellant exhaustion (m <= m_dry)
  - Stage separation: 3 s coast after MECO (pyrotechnic delay)
  - Apogee:           radial velocity sign change (r_dot: + -> -)
  - Booster flip:     attitude alignment with retrograde (dot product check)
  - Boostback:        horizontal velocity toward site cancelled
  - Entry interface:  Knudsen number / altitude threshold (70 km)
  - Landing burn:     energy-based ignition (Tsiolkovsky work integral)
"""

from enum import Enum, auto
import logging

import numpy as np

from . import constants as C
from .config import SimulationConfig, create_default_config
from .recovery import (
    booster_min_propellant_after_boostback,
    booster_propellant_remaining,
    estimate_suicide_burn,
)
from .state import State
from .mass import is_propellant_exhausted
from .frames import quaternion_to_rotation_matrix

logger = logging.getLogger(__name__)


class MissionPhase(Enum):
    PRELAUNCH = auto()
    ASCENT = auto()
    COAST = auto()                  # Post-MECO coast (before separation)
    STAGE_SEPARATION = auto()       # S1/S2 separate (mass event)
    S2_COAST_TO_APOGEE = auto()     # S2 coasts to apogee
    ORBIT_INSERTION = auto()        # S2 circularization burn at/near apogee
    ORBIT_ACHIEVED = auto()         # Circular orbit reached
    ORBIT_FAILED = auto()           # Orbit insertion failed
    APOGEE_REACHED = auto()         # Legacy: for ascent-only sims
    # Booster Recovery Phases
    BOOSTER_FLIP = auto()
    BOOSTER_BOOSTBACK = auto()
    BOOSTER_COAST = auto()
    BOOSTER_ENTRY = auto()
    BOOSTER_LANDING = auto()


class MissionManager:
    """
    Manages the current mission phase and transitions using physics-based criteria.
    """

    def __init__(self, vehicle_type: str = "ascent", initial_mass: float = None,
                 config: SimulationConfig = None):
        self.vehicle_type = vehicle_type
        self.config = config or create_default_config()
        self.current_phase = MissionPhase.ASCENT

        if vehicle_type == "orbiter":
            self.current_phase = MissionPhase.S2_COAST_TO_APOGEE
        elif vehicle_type == "booster":
            self.current_phase = MissionPhase.BOOSTER_FLIP

        self.meco_time = None
        self.separation_done = False      # True after mass event applied
        self.apogee_time = None
        self.orbit_insertion_time = None
        self.max_altitude_reached = 0.0
        self._last_radial_velocity = 0.0
        self._phase_entry_time = 0.0      # Time at which current phase began
        self._launch_site = C.INITIAL_POSITION.copy()
        self.orbit_failure_reason = None
        # Track initial propellant for fuel budgeting (booster recovery)
        if vehicle_type == "booster" and initial_mass is not None:
            self._initial_propellant = initial_mass - C.STAGE1_DRY_MASS
        else:
            self._initial_propellant = C.PROPELLANT_MASS

    def _compute_radial_velocity(self, state: State) -> float:
        """Compute radial velocity: r_dot = (r . v) / |r|."""
        r_norm = np.linalg.norm(state.r)
        if r_norm > 1.0:
            return float(np.dot(state.r, state.v) / r_norm)
        return 0.0

    def _compute_horizontal_velocity(self, state: State) -> float:
        """Compute horizontal speed (perpendicular to radial direction)."""
        r_hat = state.r / max(np.linalg.norm(state.r), 1.0)
        v_radial = np.dot(state.v, r_hat) * r_hat
        v_horiz = state.v - v_radial
        return float(np.linalg.norm(v_horiz))

    def _check_attitude_aligned(self, state: State, target_dir: np.ndarray,
                                 threshold_deg: float = 15.0) -> bool:
        """
        Check if body +Z axis is aligned within threshold of target direction.

        Physics: The rotation matrix R(q) transforms body -> inertial.
        Body +Z in inertial frame = R @ [0,0,1] = 3rd column of R.
        """
        R = quaternion_to_rotation_matrix(state.q)
        body_z_inertial = R[:, 2]
        target_norm = np.linalg.norm(target_dir)
        if target_norm < 1e-6:
            return False
        target_hat = target_dir / target_norm
        cos_angle = np.clip(np.dot(body_z_inertial, target_hat), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        return angle_deg < threshold_deg

    def _compute_downrange_distance(self, state: State) -> float:
        """
        Compute horizontal (great-circle) distance from launch site.

        Uses the central angle: d = R_earth * arccos(r_hat . r0_hat)
        """
        r_hat = state.r / max(np.linalg.norm(state.r), 1.0)
        r0_hat = self._launch_site / max(np.linalg.norm(self._launch_site), 1.0)
        cos_angle = np.clip(np.dot(r_hat, r0_hat), -1.0, 1.0)
        return float(C.R_EARTH * np.arccos(cos_angle))

    def _compute_suicide_burn_altitude(self, state: State, dry_mass: float) -> float:
        """
        Compute suicide-burn ignition altitude using shared recovery estimator.
        """
        burn = estimate_suicide_burn(
            state.r,
            state.v,
            state.m,
            C.THRUST_MAGNITUDE,
            safety_factor=self.config.booster_landing_ignition_safety_factor,
        )
        return float(burn['burn_altitude'])

    def update(self, state: State, dt: float):
        """
        Check for phase transitions based on current vehicle state.

        All transitions use physics-based criteria rather than hardcoded times.

        Args:
            state: Current vehicle state
            dt: Time step
        """
        self.max_altitude_reached = max(self.max_altitude_reached, state.altitude)

        radial_velocity = self._compute_radial_velocity(state)

        # =====================================================================
        # ASCENT VEHICLE TRANSITIONS
        # =====================================================================

        # ASCENT -> COAST (MECO: Main Engine Cut-Off)
        # Physics: S1 ascent propellant exhausted, reserving fuel for booster landing.
        # MECO mass = S1_dry + S2_wet + landing_reserve = 30000 + 120000 + 42000 = 192000 kg
        if self.current_phase == MissionPhase.ASCENT:
            meco_mass = C.DRY_MASS + C.STAGE1_LANDING_FUEL_RESERVE  # 192,000 kg
            if state.m <= meco_mass:
                s1_prop_burned = C.INITIAL_MASS - state.m
                logger.info(f"MECO detected at t={state.t:.2f}s, "
                           f"Alt={state.altitude/1000:.1f}km, "
                           f"V={state.speed:.0f}m/s, "
                           f"S1 prop burned={s1_prop_burned:.0f}kg, "
                           f"Fuel reserved={C.STAGE1_LANDING_FUEL_RESERVE:.0f}kg")
                self.current_phase = MissionPhase.COAST
                self.meco_time = state.t
                self._phase_entry_time = state.t

        # COAST -> STAGE_SEPARATION
        # Physics: 3-second pyrotechnic delay after MECO (industry standard)
        # During this interval, residual thrust tails off and springs push stages apart
        elif self.current_phase == MissionPhase.COAST:
            if self.meco_time and (state.t - self.meco_time) > 3.0:
                logger.info(f"Stage Separation at t={state.t:.2f}s, "
                           f"Alt={state.altitude/1000:.1f}km")
                self.current_phase = MissionPhase.STAGE_SEPARATION
                self._phase_entry_time = state.t

        # STAGE_SEPARATION -> S2_COAST_TO_APOGEE
        # After 2s of separation clearance, S2 begins coast to apogee
        elif self.current_phase == MissionPhase.STAGE_SEPARATION:
            time_since_sep = state.t - self._phase_entry_time
            if time_since_sep > 2.0:
                logger.info(f"S2 Coast-to-Apogee begins at t={state.t:.2f}s, "
                           f"Alt={state.altitude/1000:.1f}km, "
                           f"V={state.speed:.0f}m/s")
                self.current_phase = MissionPhase.S2_COAST_TO_APOGEE
                self._phase_entry_time = state.t

        # S2_COAST_TO_APOGEE -> ORBIT_INSERTION
        # Physics: Begin S2 burn as soon as separation clearance is complete
        # and altitude is above the dense atmosphere (> 80 km). The S2 TWR < 1
        # means it cannot hover, so we must start the burn while still ascending
        # to gain both altitude and velocity. Waiting until apogee would cause
        # the vehicle to re-enter during the long burn (~548s).
        #
        # This is standard for real upper stages (e.g., Falcon 9 S2 ignites
        # ~3-5s after separation, not at apogee).
        if self.current_phase == MissionPhase.S2_COAST_TO_APOGEE:
            # Start burn immediately: above 80 km (well out of atmosphere)
            # and after brief coast (already guaranteed by STAGE_SEPARATION -> S2_COAST transition)
            above_atmosphere = state.altitude > 80000.0
            # Also trigger if altitude is dropping (safety — don't wait too long)
            altitude_falling = radial_velocity < -20.0 and state.altitude > 70000.0

            if above_atmosphere or altitude_falling:
                logger.info(f"ORBIT INSERTION BURN START at t={state.t:.2f}s, "
                           f"Alt={state.altitude/1000:.1f}km, "
                           f"V={state.speed:.0f}m/s, "
                           f"V_radial={radial_velocity:.1f}m/s")
                self.current_phase = MissionPhase.ORBIT_INSERTION
                self.orbit_insertion_time = state.t
                self._phase_entry_time = state.t

        # ORBIT_INSERTION -> ORBIT_ACHIEVED
        # Physics: Check if orbit is circular (eccentricity < 0.01)
        # or S2 propellant exhausted
        if self.current_phase == MissionPhase.ORBIT_INSERTION:
            # Compute orbital elements to check if circular
            r_mag = np.linalg.norm(state.r)
            v_mag = np.linalg.norm(state.v)
            # Specific orbital energy
            energy = 0.5 * v_mag**2 - C.MU_EARTH / r_mag
            # Semi-major axis
            if abs(energy) > 1.0:
                a_sma = -C.MU_EARTH / (2.0 * energy)
            else:
                a_sma = r_mag  # fallback
            # Eccentricity from vis-viva: e = sqrt(1 - h^2/(mu*a))
            h_vec = np.cross(state.r, state.v)
            h_mag = np.linalg.norm(h_vec)
            if a_sma > 0 and C.MU_EARTH * a_sma > 0:
                ecc_sq = max(0.0, 1.0 - h_mag**2 / (C.MU_EARTH * a_sma))
                ecc = np.sqrt(ecc_sq)
            else:
                ecc = 1.0

            # Circular velocity at current altitude
            v_circular = np.sqrt(C.MU_EARTH / r_mag)
            v_deficit = v_circular - v_mag

            # S2 propellant check
            s2_prop_remaining = state.m - C.STAGE2_DRY_MASS
            s2_prop_exhausted = s2_prop_remaining <= 0.0

            # Compute perigee/apogee for strict target-orbit validation
            perigee_alt = a_sma * (1 - ecc) - C.R_EARTH if a_sma > 0 else -C.R_EARTH
            apogee_alt = a_sma * (1 + ecc) - C.R_EARTH if a_sma > 0 else -C.R_EARTH
            target_alt = self.config.orbit_target_altitude_m
            alt_tol = self.config.orbit_altitude_tolerance_m
            ecc_max = self.config.orbit_ecc_max
            orbit_ok = (
                energy < 0 and
                abs(perigee_alt - target_alt) <= alt_tol and
                abs(apogee_alt - target_alt) <= alt_tol and
                ecc <= ecc_max
            )

            if orbit_ok:
                perigee = a_sma * (1 - ecc) - C.R_EARTH
                logger.info(f"ORBIT ACHIEVED at t={state.t:.2f}s! "
                           f"Alt={state.altitude/1000:.1f}km, V={v_mag:.0f}m/s, "
                           f"e={ecc:.4f}, a={a_sma/1000:.0f}km, "
                           f"Perigee={perigee/1000:.0f}km, Apogee={apogee_alt/1000:.0f}km")
                self.current_phase = MissionPhase.ORBIT_ACHIEVED
                self._phase_entry_time = state.t

            elif s2_prop_exhausted:
                self.orbit_failure_reason = (
                    f"target miss: perigee={perigee_alt/1000:.1f}km, "
                    f"apogee={apogee_alt/1000:.1f}km, e={ecc:.4f}"
                )
                logger.error(
                    f"S2 PROPELLANT EXHAUSTED at t={state.t:.2f}s "
                    f"(ORBIT INSERTION FAILED), {self.orbit_failure_reason}"
                )
                self.current_phase = MissionPhase.ORBIT_FAILED
                self._phase_entry_time = state.t

        # =====================================================================
        # BOOSTER RECOVERY TRANSITIONS (Physics-based)
        # =====================================================================

        if self.vehicle_type == "booster":

            # FLIP -> BOOSTBACK
            # Physics: Flip complete when body +Z axis aligns within 15 deg
            # of retrograde direction (velocity reversed). This replaces the
            # hardcoded timer and depends on actual attitude dynamics.
            if self.current_phase == MissionPhase.BOOSTER_FLIP:
                v_norm = np.linalg.norm(state.v)
                if v_norm > 1.0:
                    retrograde = -state.v / v_norm
                    aligned = self._check_attitude_aligned(state, retrograde, 15.0)
                else:
                    aligned = True  # Vehicle nearly stationary — flip doesn't matter

                # Safety: also allow transition after 30s regardless (tumble recovery)
                time_in_phase = state.t - self._phase_entry_time
                if (aligned and time_in_phase > 3.0) or time_in_phase > 30.0:
                    logger.info(f"Booster Flip Complete at t={state.t:.2f}s "
                               f"(attitude aligned: {aligned})")
                    self.current_phase = MissionPhase.BOOSTER_BOOSTBACK
                    self._phase_entry_time = state.t

            # BOOSTBACK -> COAST
            # Physics: Boostback complete when horizontal velocity component
            # toward launch site is small enough that ballistic trajectory
            # will return near the pad. We check:
            #   1. Downrange distance is decreasing (heading home), AND
            #   2. Horizontal speed < threshold, OR propellant low
            elif self.current_phase == MissionPhase.BOOSTER_BOOSTBACK:
                # Project velocity onto horizontal plane
                r_hat = state.r / max(np.linalg.norm(state.r), 1.0)
                v_horiz = state.v - np.dot(state.v, r_hat) * r_hat
                v_horiz_mag = np.linalg.norm(v_horiz)

                # Exit boostback only when horizontal velocity is largely cancelled
                # and the booster is no longer climbing.
                horiz_cancelled = v_horiz_mag < 120.0
                ascent_arrested = radial_velocity <= 0.0
                boostback_complete = horiz_cancelled and ascent_arrested

                # Fuel budget: Keep enough for landing burn (terminal velocity + margin)
                # Terminal velocity from 95km fall ~= sqrt(2*g*h) ~= 1365 m/s
                # Landing dV needed: ~1400 m/s (mostly vertical after good boostback)
                # Tsiolkovsky: m_fuel = m_dry*(exp(dv/(Isp*g0))-1) = 30000*(exp(1400/2766)-1) ≈ 19,700 kg
                # Keep 20% of initial propellant as minimum reserve
                propellant_remaining = booster_propellant_remaining(state.m)
                min_after_boostback = booster_min_propellant_after_boostback(self.config)
                propellant_used = max(0.0, self._initial_propellant - propellant_remaining)
                budget_used = propellant_used >= self.config.booster_boostback_budget_kg
                fuel_guard = propellant_remaining <= min_after_boostback

                if boostback_complete or budget_used or fuel_guard:
                    reason = ("horiz cancelled + ascent arrested"
                              if boostback_complete else
                              ("boostback budget" if budget_used else "reserve guard"))
                    logger.info(f"Boostback Complete at t={state.t:.2f}s ({reason}), "
                               f"V_horiz={v_horiz_mag:.0f}m/s, Vr={radial_velocity:.0f}m/s, "
                               f"prop={propellant_remaining:.0f}kg, "
                               f"used={propellant_used:.0f}kg")
                    self.current_phase = MissionPhase.BOOSTER_COAST
                    self._phase_entry_time = state.t

            # COAST -> ENTRY
            # Physics: Entry interface at ~70 km altitude (Karman line vicinity)
            # where atmospheric density becomes significant for deceleration.
            # Knudsen number Kn < 0.01 marks continuum flow regime.
            elif self.current_phase == MissionPhase.BOOSTER_COAST:
                # Entry interface: altitude < 70 km and descending
                entry_interface = self.config.booster_entry_interface_altitude_m
                if state.altitude < entry_interface and radial_velocity < 0.0:
                    logger.info(f"Booster Entry Interface at t={state.t:.2f}s, "
                               f"Alt={state.altitude/1000:.1f}km, "
                               f"V={state.speed:.0f}m/s")
                    self.current_phase = MissionPhase.BOOSTER_ENTRY
                    self._phase_entry_time = state.t

            # ENTRY -> LANDING
            # Physics: Transition to landing phase when EITHER:
            #   (a) The energy-based ignition estimate triggers (Tsiolkovsky
            #       integration accounting for variable mass), OR
            #   (b) Altitude drops below the minimum landing burn altitude.
            #
            # The minimum altitude floor (b) is critical for ZEM/ZEV divert
            # guidance.  Without it the energy estimator may not trigger until
            # < 100 m (after a slow entry burn), leaving only ~1 s to close a
            # 10+ km position error to the pad.  Starting at 2000 m gives the
            # ZEM/ZEV law ~25 s to both decelerate and translate to the pad
            # while keeping propellant consumption within the landing reserve.
            elif self.current_phase == MissionPhase.BOOSTER_ENTRY:
                burn = estimate_suicide_burn(
                    state.r,
                    state.v,
                    state.m,
                    C.THRUST_MAGNITUDE,
                    safety_factor=self.config.booster_landing_ignition_safety_factor,
                )
                h_ignite = float(burn['burn_altitude'])

                # (a) Energy-based ignition
                energy_trigger = burn['ignite'] and radial_velocity < 0.0

                # (b) Minimum altitude floor — ensure ZEM/ZEV has enough time
                min_land_alt = self.config.booster_landing_min_altitude_m
                altitude_trigger = (
                    state.altitude < min_land_alt and
                    radial_velocity < 0.0  # Must be descending
                )

                landing_trigger = energy_trigger or altitude_trigger
                trigger_reason = (
                    "energy_ignition" if energy_trigger else "min_altitude_floor"
                )

                if landing_trigger:
                    logger.info(
                        f"Booster Landing Phase at t={state.t:.2f}s, "
                        f"Alt={state.altitude/1000:.1f}km, "
                        f"h_ignite={h_ignite:.0f}m, "
                        f"trigger={trigger_reason}"
                    )
                    self.current_phase = MissionPhase.BOOSTER_LANDING
                    self._phase_entry_time = state.t
        self._last_radial_velocity = radial_velocity

    def get_phase(self) -> MissionPhase:
        return self.current_phase

    @property
    def separation_time(self) -> float:
        """Return the time of stage separation (MECO + 3s), or None."""
        if self.meco_time is not None:
            return self.meco_time + 3.0
        return None




