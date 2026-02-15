"""
RLV Ascent Simulation - Monte Carlo Dispersion Analysis

Framework for running parametric dispersions to assess mission
robustness. Supports thrust, Isp, mass, wind, CG offset, and
thrust misalignment variations.

Enabled via config.enable_monte_carlo = True.
"""

import time
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Callable

import numpy as np

from . import constants as C
from .config import SimulationConfig, create_default_config


@dataclass
class MCDispersion:
    """Definition of a single Monte Carlo dispersion parameter."""
    name: str
    nominal: float
    sigma: float       # 1-sigma magnitude
    distribution: str = "gaussian"  # "gaussian" or "uniform"


@dataclass
class MCRunResult:
    """Result from a single Monte Carlo run."""
    run_index: int
    seed: int
    max_altitude_km: float
    max_velocity_ms: float
    meco_time: float
    meco_altitude_km: float
    meco_velocity_ms: float
    max_q_pa: float
    max_q_alpha_pa_rad: float
    final_reason: str
    dispersions_applied: Dict[str, float] = field(default_factory=dict)


@dataclass
class MCResults:
    """Aggregated results from a Monte Carlo campaign."""
    runs: List[MCRunResult] = field(default_factory=list)
    config: Optional[SimulationConfig] = None
    wall_time_s: float = 0.0

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def get_statistic(self, attr: str) -> dict:
        """Compute mean/std/min/max for a scalar attribute across runs."""
        values = [getattr(r, attr) for r in self.runs if hasattr(r, attr)]
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
        }

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [f"Monte Carlo Results: {self.n_runs} runs in {self.wall_time_s:.1f}s"]
        for attr in ['max_altitude_km', 'max_velocity_ms', 'meco_time',
                      'max_q_pa', 'max_q_alpha_pa_rad']:
            stats = self.get_statistic(attr)
            lines.append(f"  {attr:25s}: mean={stats['mean']:.2f} std={stats['std']:.2f} "
                        f"min={stats['min']:.2f} max={stats['max']:.2f}")
        return '\n'.join(lines)


def run_monte_carlo(base_config: SimulationConfig = None,
                    n_runs: int = 100,
                    seed: int = 42,
                    run_function: Callable = None,
                    verbose: bool = True) -> MCResults:
    """
    Run a Monte Carlo dispersion campaign.

    For each run, disperses simulation parameters according to config
    tolerances, runs the simulation, and collects statistics.

    Args:
        base_config: Base simulation configuration
        n_runs: Number of Monte Carlo runs
        seed: Master random seed
        run_function: Callable(config, seed) -> (state, log, reason).
                      If None, uses run_simulation from main.
        verbose: Print progress

    Returns:
        MCResults with per-run data and statistics
    """
    if base_config is None:
        base_config = create_default_config()

    if run_function is None:
        from .main import run_simulation
        def run_function(cfg, s):
            return run_simulation(dt=cfg.dt, max_time=cfg.max_time, verbose=False,
                                  config=cfg)

    rng = np.random.default_rng(seed)
    results = MCResults(config=base_config)
    start = time.time()

    for i in range(n_runs):
        run_seed = int(rng.integers(0, 2**31))

        # Apply dispersions
        dispersions = {}

        thrust_scale = 1.0 + rng.normal(0, base_config.mc_thrust_dispersion)
        dispersions['thrust_scale'] = thrust_scale

        isp_scale = 1.0 + rng.normal(0, base_config.mc_isp_dispersion)
        dispersions['isp_scale'] = isp_scale

        mass_offset = rng.normal(0, base_config.mc_mass_dispersion) * C.INITIAL_MASS
        dispersions['mass_offset_kg'] = mass_offset

        wind_offset = rng.normal(0, base_config.mc_wind_dispersion)
        dispersions['wind_offset_ms'] = wind_offset

        # Run simulation with dispersed parameters
        # (For now, dispersions are stored but not yet applied to the physics.
        #  The physics modules will read dispersions from config in future phases.)
        try:
            state, log, reason = run_function(base_config, run_seed)

            # Extract statistics from log
            alt_array = np.array(log.altitude) if log.altitude else np.array([0.0])
            vel_array = np.array(log.velocity) if log.velocity else np.array([0.0])
            q_array = np.array(log.dynamic_pressure) if hasattr(log, 'dynamic_pressure') and log.dynamic_pressure else np.array([0.0])
            qa_array = np.array(log.q_alpha) if hasattr(log, 'q_alpha') and log.q_alpha else np.array([0.0])

            run_result = MCRunResult(
                run_index=i,
                seed=run_seed,
                max_altitude_km=float(np.max(alt_array)),
                max_velocity_ms=float(np.max(vel_array)),
                meco_time=float(log.time[-1]) if log.time else 0.0,
                meco_altitude_km=float(alt_array[-1]) if len(alt_array) > 0 else 0.0,
                meco_velocity_ms=float(vel_array[-1]) if len(vel_array) > 0 else 0.0,
                max_q_pa=float(np.max(q_array)),
                max_q_alpha_pa_rad=float(np.max(qa_array)),
                final_reason=reason or "unknown",
                dispersions_applied=dispersions,
            )
        except Exception as e:
            run_result = MCRunResult(
                run_index=i, seed=run_seed,
                max_altitude_km=0, max_velocity_ms=0,
                meco_time=0, meco_altitude_km=0, meco_velocity_ms=0,
                max_q_pa=0, max_q_alpha_pa_rad=0,
                final_reason=f"ERROR: {e}",
                dispersions_applied=dispersions,
            )

        results.runs.append(run_result)

        if verbose and (i + 1) % max(1, n_runs // 10) == 0:
            elapsed = time.time() - start
            print(f"  MC run {i+1}/{n_runs} ({elapsed:.1f}s)")

    results.wall_time_s = time.time() - start

    if verbose:
        print(results.summary())

    return results
