import numpy as np
import pytest
from fbica import generate_panel, generate_missing


@pytest.fixture(scope="session")
def panel():
    """Small complete panel: T=40, N=15, m=3, r=2."""
    X, F, lam, nu = generate_panel(N=15, T=40, m=3, r=2, phi=0.3, seed_errors=0)
    return X, F, lam


@pytest.fixture(scope="session")
def obs_panel(panel):
    """Observed panel with ~15 % MCAR missingness."""
    X, _, _ = panel
    X_obs, miss = generate_missing(X, miss_probs=0.15, seed=0)
    return X, X_obs, miss
