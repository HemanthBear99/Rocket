"""Tests for run_full_mission() dual-vehicle tracking."""

import pytest
import numpy as np

from rlv_sim.main import run_full_mission, FullMissionResult, SimulationLog
from rlv_sim import constants as C


class TestFullMissionBasic:
    """Basic sanity checks for full mission mode."""

    @classmethod
    def setup_class(cls):
        """Run a short full mission once for all tests in this class."""
        cls.result = run_full_mission(dt=0.1, max_time=200.0, verbose=False)

    def test_returns_full_mission_result(self):
        assert isinstance(self.result, FullMissionResult)

    def test_separation_time_positive(self):
        assert self.result.separation_time is not None
        assert self.result.separation_time > 0

    def test_ascent_log_has_data(self):
        assert len(self.result.ascent_log.time) > 0

    def test_orbiter_log_has_data(self):
        assert len(self.result.orbiter_log.time) > 0

    def test_booster_log_has_data(self):
        assert len(self.result.booster_log.time) > 0

    def test_ascent_times_before_separation(self):
        """All ascent log times should be <= separation time."""
        for t in self.result.ascent_log.time:
            assert t <= self.result.separation_time + 0.2  # Small tolerance

    def test_orbiter_times_after_separation(self):
        """All orbiter log times should be >= separation time."""
        if len(self.result.orbiter_log.time) > 0:
            assert self.result.orbiter_log.time[0] >= self.result.separation_time - 0.2

    def test_booster_times_after_separation(self):
        """All booster log times should be >= separation time."""
        if len(self.result.booster_log.time) > 0:
            assert self.result.booster_log.time[0] >= self.result.separation_time - 0.2

    def test_orbiter_mass_is_s2(self):
        """Orbiter should start at S2 wet mass."""
        if len(self.result.orbiter_log.mass) > 0:
            # First logged mass should be ~ S2 wet mass
            assert self.result.orbiter_log.mass[0] <= C.STAGE2_WET_MASS * 1.01

    def test_booster_mass_is_s1(self):
        """Booster should start at S1 dry + landing fuel."""
        expected = C.STAGE1_DRY_MASS + C.STAGE1_LANDING_FUEL_RESERVE
        if len(self.result.booster_log.mass) > 0:
            assert self.result.booster_log.mass[0] <= expected * 1.01

    def test_extended_telemetry_present(self):
        """New Phase 0.3 telemetry fields should be populated."""
        log = self.result.ascent_log
        assert len(log.mach_number) == len(log.time)
        assert len(log.dynamic_pressure) == len(log.time)
        assert len(log.angle_of_attack_deg) == len(log.time)
        assert len(log.q_alpha) == len(log.time)
        assert len(log.heating_rate) == len(log.time)
        assert len(log.latitude_deg) == len(log.time)
        assert len(log.longitude_deg) == len(log.time)
        assert len(log.phase_name) == len(log.time)

    def test_independent_guidance_states(self):
        """Orbiter and booster should have divergent trajectories."""
        if (len(self.result.orbiter_log.altitude) > 10 and
                len(self.result.booster_log.altitude) > 10):
            # After some time, altitudes should differ significantly
            orb_alt_final = self.result.orbiter_log.altitude[-1]
            bst_alt_final = self.result.booster_log.altitude[-1]
            # They diverge (orbiter goes up, booster comes back)
            assert abs(orb_alt_final - bst_alt_final) > 1.0  # > 1 km difference


class TestFullMissionReasons:
    """Test termination reasons for various scenarios."""

    def test_short_run_max_time(self):
        """Very short max_time should hit time limit."""
        result = run_full_mission(dt=0.1, max_time=50.0, verbose=False)
        # May or may not reach separation in 50s
        assert result.ascent_reason is not None
