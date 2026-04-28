# fbica

Python implementation of the FBI-CA estimator from:

> Bretschneider, T. & Kaddoura, Y. (2026). *Factor-based imputation of missing values using cross-sectional averages.*

---

## What it does

FBI-CA imputes missing values in a panel (T x N x m) by projecting each unit's observations onto cross-sectional averages of the other units, which serve as proxies for the latent common factors. The imputed value for a missing cell is the fitted common component from that projection.

The leave-one-out (LOO) variant (the default)  excludes unit i when constructing the factor proxy for unit i.

The package also provides bootstrap confidence and prediction intervals:

- **CI** — for the common component. Block-wild bootstrap with block length ceil(T^(1/3)), accounting for temporal dependence.
- **PI** — for the realised value including the idiosyncratic error. iid pairs bootstrap with residual resampling. Wider than CI by construction.


---

## Installation

```bash
git clone https://github.com/YousefKad/fbica.git
cd fbica
conda env create -f environment.yml
conda activate fbica
```

Or with pip: `pip install -e .`

---

## Basic example

```python
from fbica import FBICA, FBICABootstrap

# X is (T, N, m) with np.nan at missing entries
imp = FBICA(use_loo=True)
X_filled = imp.fit_transform(X)

# CI: block-wild bootstrap for the common component C_{i,t}.
# Accounts for serial dependence via non-overlapping blocks of length ceil(T^(1/3)).
bs = FBICABootstrap(interval_type="CI", B=499, alpha=0.05)
result = bs.fit(X, target_points=[(5, 3, 0), (15, 10, 1)])

# PI: iid pairs bootstrap for the realised value x_{i,t} = C_{i,t} + e_{i,t}.
# Resamples residuals independently.
# bs = FBICABootstrap(interval_type="PI", B=499, alpha=0.05)

result.point_est   # imputed values
result.lower       # lower bounds
result.upper       # upper bounds
```

In general, `FBICA` and `FBICABootstrap` should always be given the same proxy settings (`use_loo`, `factor_vars`, `always_observed`) — the bootstrap builds its factor proxy the same way as the original fit, so any mismatch will give inconsistent intervals.

If you have a "tall block" of units that are always fully observed, pass their indices as `always_observed`. The factor proxy is then built from that block instead of the LOO average. Same rule applies for `use_loo=False`:

```python
# units 0–4 are always observed — use them as the factor proxy
imp = FBICA(always_observed=[0, 1, 2, 3, 4])
X_filled = imp.fit_transform(X)

bs = FBICABootstrap(interval_type="CI", always_observed=[0, 1, 2, 3, 4], B=499)
result = bs.fit(X, target_points=[(5, 10, 0)])

# same applies when turning off LOO
imp = FBICA(use_loo=False)
bs = FBICABootstrap(interval_type="CI", use_loo=False, B=499)
```

For mixed-frequency panels, pass `factor_vars` to restrict which variables are used to build the factor proxies. For real-time expanding-window imputation, use `fit_transform_expanding`.

---

## Notebooks

`01_simulation_demo.ipynb` — imputation accuracy across panel sizes, LOO vs plain, spatial dependence.

`02_bootstrap_coverage.ipynb` — CI and PI coverage at N=T=25, replicating the Monte Carlo results in the paper.
