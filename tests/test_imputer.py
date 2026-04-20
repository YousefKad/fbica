import numpy as np
import pytest
from fbica import FBICA


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_obs(obs_panel):
    X, X_obs, miss = obs_panel
    return X, X_obs, miss


# ── repr ──────────────────────────────────────────────────────────────────────

def test_repr_unfitted():
    imp = FBICA(use_loo=True)
    assert "not fitted" in repr(imp)


def test_repr_fitted(obs_panel):
    _, X_obs, _ = obs_panel
    imp = FBICA(use_loo=True)
    imp.fit_transform(X_obs)
    assert "fitted" in repr(imp)
    assert "not fitted" not in repr(imp)


# ── plain FBI-CA ──────────────────────────────────────────────────────────────

class TestPlain:
    def test_output_shape(self, obs_panel):
        _, X_obs, _ = obs_panel
        X_filled = FBICA(use_loo=False).fit_transform(X_obs)
        assert X_filled.shape == X_obs.shape

    def test_no_nans(self, obs_panel):
        _, X_obs, _ = obs_panel
        X_filled = FBICA(use_loo=False).fit_transform(X_obs)
        assert not np.isnan(X_filled).any()

    def test_observed_values_preserved(self, obs_panel):
        X, X_obs, miss = obs_panel
        X_filled = FBICA(use_loo=False).fit_transform(X_obs)
        np.testing.assert_array_equal(X_filled[~miss], X_obs[~miss])

    def test_fitted_attributes_set(self, obs_panel):
        _, X_obs, _ = obs_panel
        imp = FBICA(use_loo=False)
        imp.fit_transform(X_obs)
        T, N, m = X_obs.shape
        assert imp.fhat_.shape == (T, m)       # plain: (T, m_f)
        assert imp.lam_hat_.shape == (N, m, m) # (N, m_f, m)
        assert imp.C_fit_.shape == (T, N, m)

    def test_imputation_reduces_nan_count(self, obs_panel):
        _, X_obs, miss = obs_panel
        X_filled = FBICA(use_loo=False).fit_transform(X_obs)
        assert np.isnan(X_filled).sum() < np.isnan(X_obs).sum()


# ── LOO FBI-CA ────────────────────────────────────────────────────────────────

class TestLOO:
    def test_output_shape(self, obs_panel):
        _, X_obs, _ = obs_panel
        X_filled = FBICA(use_loo=True).fit_transform(X_obs)
        assert X_filled.shape == X_obs.shape

    def test_no_nans(self, obs_panel):
        _, X_obs, _ = obs_panel
        X_filled = FBICA(use_loo=True).fit_transform(X_obs)
        assert not np.isnan(X_filled).any()

    def test_observed_values_preserved(self, obs_panel):
        X, X_obs, miss = obs_panel
        X_filled = FBICA(use_loo=True).fit_transform(X_obs)
        np.testing.assert_array_equal(X_filled[~miss], X_obs[~miss])

    def test_fitted_attributes_set(self, obs_panel):
        _, X_obs, _ = obs_panel
        imp = FBICA(use_loo=True)
        imp.fit_transform(X_obs)
        T, N, m = X_obs.shape
        assert imp.fhat_.shape == (N, T, m)    # LOO: (N, T, m_f)
        assert imp.lam_hat_.shape == (N, m, m)
        assert imp.C_fit_.shape == (T, N, m)

    def test_loo_requires_n_ge_2(self):
        X = np.random.randn(20, 1, 3)
        X[5, 0, 0] = np.nan
        with pytest.raises(ValueError, match="N=2"):
            FBICA(use_loo=True).fit_transform(X)


# ── factor_vars ───────────────────────────────────────────────────────────────

class TestFactorVars:
    def test_subset_factor_vars(self, obs_panel):
        _, X_obs, _ = obs_panel
        imp = FBICA(use_loo=True, factor_vars=[1, 2])
        X_filled = imp.fit_transform(X_obs)
        assert not np.isnan(X_filled).any()
        T, N, m = X_obs.shape
        assert imp.fhat_.shape == (N, T, 2)   # only 2 factor vars

    def test_empty_factor_vars_raises(self, obs_panel):
        _, X_obs, _ = obs_panel
        with pytest.raises(ValueError, match="factor_vars cannot be empty"):
            FBICA(use_loo=False, factor_vars=[]).fit_transform(X_obs)

    def test_out_of_range_factor_var_raises(self, obs_panel):
        _, X_obs, _ = obs_panel
        with pytest.raises(ValueError, match="out of range"):
            FBICA(use_loo=False, factor_vars=[0, 99]).fit_transform(X_obs)

    def test_duplicate_factor_vars_raises(self, obs_panel):
        _, X_obs, _ = obs_panel
        with pytest.raises(ValueError, match="duplicate"):
            FBICA(use_loo=False, factor_vars=[0, 0]).fit_transform(X_obs)


# ── input validation ──────────────────────────────────────────────────────────

class TestInputValidation:
    def test_wrong_ndim_raises(self):
        X_2d = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="3 dimensions"):
            FBICA().fit_transform(X_2d)

    def test_inf_raises(self, obs_panel):
        _, X_obs, _ = obs_panel
        X_bad = X_obs.copy()
        X_bad[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            FBICA().fit_transform(X_bad)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            FBICA().fit_transform([[[None, "a"], [1, 2]]])


# ── get_common_component / get_residuals ──────────────────────────────────────

class TestPostFitMethods:
    def test_get_common_component_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            FBICA().get_common_component()

    def test_get_common_component_shape(self, obs_panel):
        _, X_obs, _ = obs_panel
        imp = FBICA(use_loo=True)
        imp.fit_transform(X_obs)
        C = imp.get_common_component()
        assert C.shape == X_obs.shape

    def test_get_residuals_before_fit_raises(self, obs_panel):
        _, X_obs, _ = obs_panel
        with pytest.raises(RuntimeError):
            FBICA().get_residuals(X_obs)

    def test_get_residuals_nan_at_missing(self, obs_panel):
        _, X_obs, miss = obs_panel
        imp = FBICA(use_loo=True)
        imp.fit_transform(X_obs)
        resid = imp.get_residuals(X_obs)
        assert np.isnan(resid[miss]).all()
        assert not np.isnan(resid[~miss]).any()


# ── expanding window ──────────────────────────────────────────────────────────

class TestExpandingWindow:
    def test_output_shape(self, obs_panel):
        _, X_obs, _ = obs_panel
        imp = FBICA(use_loo=True)
        X_rt = imp.fit_transform_expanding(X_obs, min_window=20, verbose=False)
        assert X_rt.shape == X_obs.shape

    def test_no_nans(self, obs_panel):
        _, X_obs, _ = obs_panel
        imp = FBICA(use_loo=True)
        X_rt = imp.fit_transform_expanding(X_obs, min_window=20, verbose=False)
        assert not np.isnan(X_rt).any()

    def test_full_sample_attributes_set(self, obs_panel):
        _, X_obs, _ = obs_panel
        T, N, m = X_obs.shape
        imp = FBICA(use_loo=True)
        imp.fit_transform_expanding(X_obs, min_window=20, verbose=False)
        assert imp.C_fit_.shape == (T, N, m)

    def test_min_window_exceeds_T_raises(self, obs_panel):
        _, X_obs, _ = obs_panel
        T = X_obs.shape[0]
        with pytest.raises(ValueError, match="min_window"):
            FBICA(use_loo=True).fit_transform_expanding(X_obs, min_window=T + 1, verbose=False)
