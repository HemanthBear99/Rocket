import pytest
import numpy as np
from rlv_sim import main, constants as C

@pytest.mark.slow
@pytest.mark.parametrize('seed', range(10))
def test_montecarlo_dispersion(seed):
    np.random.seed(seed)
    # Perturb initial mass and thrust
    mass_perturb = 1.0 + 0.01 * np.random.randn()
    thrust_perturb = 1.0 + 0.01 * np.random.randn()
    orig_mass = C.INITIAL_MASS
    orig_thrust = C.THRUST_MAGNITUDE
    C.INITIAL_MASS = orig_mass * mass_perturb
    C.THRUST_MAGNITUDE = orig_thrust * thrust_perturb
    try:
        state, log, reason = main.run_simulation(dt=0.1, max_time=60.0, verbose=False)
        # Assert final altitude and velocity are within reasonable bounds
        assert 80e3 < state.altitude < 200e3
        assert 1000 < state.speed < 4000
    finally:
        C.INITIAL_MASS = orig_mass
        C.THRUST_MAGNITUDE = orig_thrust
