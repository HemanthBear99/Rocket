import pytest
from rlv_sim import main

def test_headless_run():
    # Ensure simulation runs without plotting or GUI
    state, log, reason = main.run_simulation(dt=0.1, max_time=1.0, verbose=False)
    assert state.t <= 1.0
    assert hasattr(log, 'time')
    assert isinstance(reason, str)
