import numpy as np
import pytest
from fbica import generate_panel, generate_missing


# ── generate_panel ────────────────────────────────────────────────────────────

def test_generate_panel_shape():
    X, F, lam, nu = generate_panel(N=10, T=30, m=3, r=2, seed_errors=1)
    assert X.shape == (30, 10, 3)
    assert F.shape == (30, 2)
    assert lam.shape == (10, 3, 2)
    assert nu.shape == (30, 10, 3)


def test_generate_panel_no_nans():
    X, *_ = generate_panel(N=10, T=30, m=3, r=2, seed_errors=1)
    assert not np.isnan(X).any()


def test_generate_panel_reproducible():
    X1, *_ = generate_panel(N=10, T=30, m=3, r=2, seed_errors=42)
    X2, *_ = generate_panel(N=10, T=30, m=3, r=2, seed_errors=42)
    np.testing.assert_array_equal(X1, X2)


def test_generate_panel_different_seeds():
    X1, *_ = generate_panel(N=10, T=30, m=3, r=2, seed_errors=1)
    X2, *_ = generate_panel(N=10, T=30, m=3, r=2, seed_errors=2)
    assert not np.allclose(X1, X2)


def test_generate_panel_fixed_factors_and_loadings():
    """Fixing F and lam should give identical common components across draws."""
    from fbica.dgp import draw_fixed_factors_and_loadings
    F, lam = draw_fixed_factors_and_loadings(N=10, T=30, m=3, r=2)
    X1, F1, lam1, _ = generate_panel(N=10, T=30, m=3, r=2, _fixed_F=F, _fixed_lam=lam, seed_errors=1)
    X2, F2, lam2, _ = generate_panel(N=10, T=30, m=3, r=2, _fixed_F=F, _fixed_lam=lam, seed_errors=2)
    np.testing.assert_array_equal(F1, F2)
    np.testing.assert_array_equal(lam1, lam2)
    # Common component X - nu should be identical
    common1 = np.einsum("tr,ibr->tib", F1, lam1)
    common2 = np.einsum("tr,ibr->tib", F2, lam2)
    np.testing.assert_array_almost_equal(common1, common2)


# ── generate_missing ──────────────────────────────────────────────────────────

def test_generate_missing_shape(panel):
    X, _, _ = panel
    X_obs, miss = generate_missing(X, miss_probs=0.2, seed=0)
    assert X_obs.shape == X.shape
    assert miss.shape == X.shape
    assert miss.dtype == bool


def test_generate_missing_nans_at_mask(panel):
    X, _, _ = panel
    X_obs, miss = generate_missing(X, miss_probs=0.2, seed=0)
    assert np.isnan(X_obs[miss]).all()
    assert not np.isnan(X_obs[~miss]).any()


def test_generate_missing_observed_values_unchanged(panel):
    X, _, _ = panel
    X_obs, miss = generate_missing(X, miss_probs=0.2, seed=0)
    np.testing.assert_array_equal(X[~miss], X_obs[~miss])


def test_generate_missing_miss_rate_approximate(panel):
    X, _, _ = panel
    X_obs, miss = generate_missing(X, miss_probs=0.2, seed=0)
    rate = miss.mean()
    assert 0.10 < rate < 0.30, f"Expected ~20% missing, got {rate:.2%}"


def test_generate_missing_per_variable_rates(panel):
    X, _, _ = panel
    miss_probs = [0.05, 0.15, 0.25]
    X_obs, miss = generate_missing(X, miss_probs=miss_probs, seed=0)
    for b, p in enumerate(miss_probs):
        rate = miss[:, :, b].mean()
        assert abs(rate - p) < 0.10, f"Variable {b}: expected ~{p}, got {rate:.2%}"


def test_generate_missing_forced(panel):
    X, _, _ = panel
    forced = [(0, 0, 0), (5, 3, 1), (10, 7, 2)]
    X_obs, miss = generate_missing(X, miss_probs=0.1, forced_missing=forced, seed=0)
    for (t, i, b) in forced:
        assert miss[t, i, b], f"Forced cell ({t},{i},{b}) should be missing"
        assert np.isnan(X_obs[t, i, b])


def test_generate_missing_wrong_prob_length(panel):
    X, _, _ = panel
    with pytest.raises(ValueError, match="miss_probs"):
        generate_missing(X, miss_probs=[0.1, 0.2])  # m=3 but only 2 probs
