import pytest
import numpy as np
from rlv_sim import main
import os

GOLDEN_MECO_CHECKSUM_FILE = os.path.join(os.path.dirname(__file__), 'golden_meco_checksum.npy')

@pytest.mark.regression
def test_golden_run_meco_checksum():
    # Run a full simulation to MECO
    state, log, reason = main.run_simulation(dt=0.1, max_time=120.0, verbose=False, coast_to_apogee=False)
    # Use state vector at MECO
    vec = state.to_vector()
    checksum = np.round(np.sum(vec), 8)
    # If golden file exists, compare
    if os.path.exists(GOLDEN_MECO_CHECKSUM_FILE):
        golden = np.load(GOLDEN_MECO_CHECKSUM_FILE)
        assert np.isclose(checksum, golden, atol=1e-6), f"MECO checksum drift: {checksum} vs {golden}"
    else:
        # Store golden checksum for future runs
        np.save(GOLDEN_MECO_CHECKSUM_FILE, checksum)
        pytest.skip("Golden MECO checksum created; rerun to enable regression check.")
