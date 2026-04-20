import numpy as np
import pytest
from fbica.metrics import rmse, mae, r_squared, rmse_per_variable, summary_table


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    T, N, m = 20, 10, 3
    X_true = rng.standard_normal((T, N, m))
    noise = rng.standard_normal((T, N, m)) * 0.5
    X_imp = X_true + noise
    mask = rng.random((T, N, m)) < 0.3
    return X_true, X_imp, mask


def test_rmse_perfect(arrays):
    X_true, _, mask = arrays
    assert rmse(X_true, X_true, mask) == pytest.approx(0.0)


def test_rmse_known_value():
    X_true = np.zeros((5, 5, 2))
    X_imp = np.ones((5, 5, 2))
    assert rmse(X_true, X_imp) == pytest.approx(1.0)


def test_rmse_with_mask(arrays):
    X_true, X_imp, mask = arrays
    val = rmse(X_true, X_imp, mask)
    assert val > 0
    assert np.isfinite(val)


def test_rmse_no_mask(arrays):
    X_true, X_imp, _ = arrays
    val = rmse(X_true, X_imp)
    assert val > 0
    assert np.isfinite(val)


def test_mae_perfect(arrays):
    X_true, _, mask = arrays
    assert mae(X_true, X_true, mask) == pytest.approx(0.0)


def test_mae_known_value():
    X_true = np.zeros((5, 5, 2))
    X_imp = np.ones((5, 5, 2)) * 2
    assert mae(X_true, X_imp) == pytest.approx(2.0)


def test_mae_no_mask(arrays):
    X_true, X_imp, _ = arrays
    val = mae(X_true, X_imp)
    assert val > 0 and np.isfinite(val)


def test_r_squared_perfect(arrays):
    X_true, _, mask = arrays
    assert r_squared(X_true, X_true, mask) == pytest.approx(1.0)


def test_r_squared_no_mask(arrays):
    X_true, X_imp, _ = arrays
    val = r_squared(X_true, X_imp)
    assert np.isfinite(val)


def test_r_squared_constant_true():
    X_true = np.ones((5, 5, 2))
    X_imp = np.ones((5, 5, 2)) * 2
    val = r_squared(X_true, X_imp)
    assert np.isnan(val)


def test_rmse_per_variable_shape(arrays):
    X_true, X_imp, mask = arrays
    out = rmse_per_variable(X_true, X_imp, mask)
    assert out.shape == (X_true.shape[2],)


def test_rmse_per_variable_perfect(arrays):
    X_true, _, mask = arrays
    out = rmse_per_variable(X_true, X_true, mask)
    np.testing.assert_array_almost_equal(out, 0.0)


def test_rmse_per_variable_no_mask(arrays):
    X_true, X_imp, _ = arrays
    out = rmse_per_variable(X_true, X_imp)
    assert out.shape == (X_true.shape[2],)
    assert (out > 0).all()


def test_summary_table_columns(arrays):
    X_true, X_imp, mask = arrays
    df = summary_table(X_true, X_imp, mask)
    assert set(df.columns) == {"MAE", "RMSE", "R2"}


def test_summary_table_rows(arrays):
    X_true, X_imp, mask = arrays
    m = X_true.shape[2]
    df = summary_table(X_true, X_imp, mask)
    assert len(df) == m + 1
    assert "TOTAL" in df.index


def test_summary_table_custom_names(arrays):
    X_true, X_imp, mask = arrays
    names = ["alpha", "beta", "gamma"]
    df = summary_table(X_true, X_imp, mask, var_names=names)
    for name in names:
        assert name in df.index


def test_summary_table_perfect_imputation(arrays):
    X_true, _, mask = arrays
    df = summary_table(X_true, X_true, mask)
    np.testing.assert_array_almost_equal(df["RMSE"].values, 0.0)
