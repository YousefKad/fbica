import numpy as np
import pytest
from fbica import run_simulation
from fbica.simulation import compare_loo_vs_plain


_FAST = dict(N=15, T=30, m=3, r=2, phi=0.3, pi=0.0,
             miss_probs=0.15, n_sim=5, verbose=False)


def test_run_simulation_returns_expected_keys():
    res = run_simulation(**_FAST)
    expected = {"mean_imputed", "true_C", "rmse", "bias", "store",
                "fixed_points", "params"}
    assert expected.issubset(res.keys())


def test_run_simulation_rmse_nonnegative():
    res = run_simulation(**_FAST)
    assert (res["rmse"] >= 0).all()


def test_run_simulation_store_shape():
    res = run_simulation(**_FAST)
    n_pts = len(res["fixed_points"])
    assert res["store"].shape == (_FAST["n_sim"], n_pts)


def test_run_simulation_true_c_finite():
    res = run_simulation(**_FAST)
    assert np.isfinite(res["true_C"]).all()


def test_run_simulation_fixed_points_custom():
    fixed = [(4, 4, 0), (5, 5, 1)]
    res = run_simulation(**_FAST, fixed_points=fixed)
    assert len(res["true_C"]) == 2
    assert res["store"].shape == (_FAST["n_sim"], 2)


def test_run_simulation_params_recorded():
    res = run_simulation(**_FAST)
    p = res["params"]
    assert p["N"] == _FAST["N"]
    assert p["T"] == _FAST["T"]
    assert p["n_sim"] == _FAST["n_sim"]


def test_run_simulation_loo_vs_plain_rmse_finite():
    for use_loo in (True, False):
        res = run_simulation(**_FAST, use_loo=use_loo)
        assert np.isfinite(res["rmse"]).all()


def test_compare_returns_both_variants(capsys):
    comp = compare_loo_vs_plain(**_FAST)
    assert "loo" in comp and "plain" in comp


def test_compare_loo_plain_rmse_finite(capsys):
    comp = compare_loo_vs_plain(**_FAST)
    assert np.isfinite(comp["loo"]["rmse"]).all()
    assert np.isfinite(comp["plain"]["rmse"]).all()
