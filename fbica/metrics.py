import numpy as np


# RMSE, MAE, R2 — computed only over missing cells when a mask is passed.

def rmse(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is None:
        diff = X_imputed - X_true
    else:
        diff = np.where(mask, X_imputed - X_true, np.nan)
    return float(np.sqrt(np.nanmean(diff ** 2)))


def mae(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is None:
        diff = np.abs(X_imputed - X_true)
    else:
        diff = np.where(mask, np.abs(X_imputed - X_true), np.nan)
    return float(np.nanmean(diff))


def r_squared(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is None:
        y   = X_true.ravel()
        yh  = X_imputed.ravel()
    else:
        y   = X_true[mask]
        yh  = X_imputed[mask]

    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def rmse_per_variable(
    X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray = None
) -> np.ndarray:
    m = X_true.shape[2]
    out = np.zeros(m)
    for b in range(m):
        msk_b = None if mask is None else mask[:, :, b:b+1]
        out[b] = rmse(X_true[:, :, b:b+1], X_imputed[:, :, b:b+1], msk_b)
    return out


def summary_table(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    mask: np.ndarray,
    var_names=None,
) -> "pandas.DataFrame":
    import pandas as pd

    m = X_true.shape[2]
    if var_names is None:
        var_names = [f"var_{b}" for b in range(m)]

    rows = []
    for b in range(m):
        msk_b = mask[:, :, b:b+1]
        rows.append({
            "Variable": var_names[b],
            "MAE"     : mae(X_true[:, :, b:b+1],       X_imputed[:, :, b:b+1],      msk_b),
            "RMSE"    : rmse(X_true[:, :, b:b+1],      X_imputed[:, :, b:b+1],      msk_b),
            "R2"      : r_squared(X_true[:, :, b:b+1], X_imputed[:, :, b:b+1],      msk_b),
        })

    rows.append({
        "Variable": "TOTAL",
        "MAE"     : mae(X_true,       X_imputed,      mask),
        "RMSE"    : rmse(X_true,      X_imputed,      mask),
        "R2"      : r_squared(X_true, X_imputed,      mask),
    })

    return pd.DataFrame(rows).set_index("Variable").round(4)
