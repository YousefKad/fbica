# fbica

Python implementation of the FBI-CA estimator from:

> Bretschneider, T. & Kaddoura, Y. (2026). *Factor-based imputation of missing values using cross-sectional averages.*

---

## What it does

FBI-CA imputes missing values in a panel (T x N x m) by projecting each unit's observations onto cross-sectional averages of the other units, which serve as proxies for the latent common factors. The imputed value for a missing cell is the fitted common component from that projection.

The leave-one-out (LOO) variant (the default)  excludes unit i when constructing the factor proxy for unit i to reducing finite-sample bias.

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

## Here is a basic example:

```python
from fbica import FBICA, FBICABootstrap

# X is (T, N, m) with np.nan at missing entries
imp = FBICA(use_loo=True)
X_filled = imp.fit_transform(X)

# Bootstrap intervals for specific missing cells
bs = FBICABootstrap(interval_type="CI", B=499, alpha=0.05)
result = bs.fit(X_obs, target_points=[(5, 3, 0), (15, 10, 1)])

result.point_est   # imputed values
result.lower       # lower bounds
result.upper       # upper bounds
```

Use `interval_type="PI"` for prediction intervals.

For mixed-frequency panels, pass `factor_vars` to restrict which variables are used to build the factor proxies. For real-time expanding-window imputation, use `fit_transform_expanding`.

---

## Notebooks

`01_simulation_demo.ipynb` — imputation accuracy across panel sizes, LOO vs plain, spatial dependence.

`02_bootstrap_coverage.ipynb` — CI and PI coverage at N=T=25, replicating the Monte Carlo results in the paper.
