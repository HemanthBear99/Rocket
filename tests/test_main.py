import pytest
from rlv_sim import main

def test_run_simulation_completes():
    # Run a very short simulation for test speed
    state, log, reason = main.run_simulation(dt=0.1, max_time=1.0, verbose=False)
    assert state.t <= 1.0
    assert hasattr(log, 'time')
    assert isinstance(reason, str)
    assert len(log.time) > 0

def test_check_termination_conditions():
    from rlv_sim.state import State
    s = State(m=main.C.DRY_MASS, r=main.C.INITIAL_POSITION.copy(), v=main.C.INITIAL_VELOCITY.copy(), q=main.C.INITIAL_QUATERNION.copy(), omega=main.C.INITIAL_OMEGA.copy(), t=0.0)
    # Should terminate if propellant exhausted and not coasting
    term, reason = main.check_termination(s, max_time=100.0, meco_time=None, coast_to_apogee=False)
    assert term is True
    assert 'MECO' in reason or 'Propellant' in reason
    # Should terminate if max_time exceeded
    s2 = State(m=main.C.INITIAL_MASS, r=main.C.INITIAL_POSITION.copy(), v=main.C.INITIAL_VELOCITY.copy(), q=main.C.INITIAL_QUATERNION.copy(), omega=main.C.INITIAL_OMEGA.copy(), t=200.0)
    term, reason = main.check_termination(s2, max_time=100.0)
    assert term is True
    assert 'Maximum simulation time' in reason

def test_simulation_step_pipeline():
    from rlv_sim.state import State
    from rlv_sim.actuator import ActuatorState
    s = State(r=main.C.INITIAL_POSITION.copy(), v=main.C.INITIAL_VELOCITY.copy(), q=main.C.INITIAL_QUATERNION.copy(), omega=main.C.INITIAL_OMEGA.copy(), m=main.C.INITIAL_MASS, t=0.0)
    a = ActuatorState(thrust_dir=main.C.INITIAL_POSITION.copy()/main.C.R_EARTH)
    s2, guidance, control, a2 = main.simulation_step(s, a, dt=0.1)
    assert hasattr(s2, 'r')
    assert isinstance(guidance, dict)
    assert isinstance(control, dict)
    assert hasattr(a2, 'thrust_dir')
