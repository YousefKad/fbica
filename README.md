# fbica — Factor-Based Imputation via Cross-Sectional Averages

Implementation of the FBI-CA estimator from:

> Bretschneider, T. & Kaddoura, Y. (2025).  
> *Factor-based imputation of missing values using cross-sectional averages.*

---

## Method

FBI-CA imputes missing entries in a panel of dimension T x N x m by using cross-sectional averages as proxies for the latent common factors. For each missing cell, the imputed value is the fitted common component

&nbsp;&nbsp;&nbsp;&nbsp;C(i,t,b) = lambda(i,b)' * f(t)

where the factor proxy f(t) is the vector of cross-sectional averages across variables at time t. In the leave-one-out (LOO) form, unit i is excluded when constructing its own factor proxy, which reduces finite-sample bias when the number of variables exceeds the number of true factors.

**Algorithm.** For each unit i and variable b:

1. Compute the LOO cross-sectional average for each variable and time period, using all observed units except i.
2. Stack these averages into the factor proxy vector.
3. Estimate loadings by least squares on the observed time periods for (i, b).
4. Impute all missing entries for (i, b) using the fitted factor proxy and estimated loadings.

Setting `use_loo=False` uses the full cross-section instead.

---

## Bootstrap inference

The package provides two types of intervals for a missing cell, via `FBICABootstrap`.

**Confidence interval (CI)** targets the common component. Uses a block-wild bootstrap: cross-sectional indices are resampled with replacement, and time-series residuals are multiplied by Rademacher weights drawn in non-overlapping blocks of length ceil(T^(1/3)). This accounts for temporal dependence.

**Prediction interval (PI)** targets the realised value, including the idiosyncratic error. Uses an iid pairs bootstrap where centred residuals are resampled to mimic the idiosyncratic shock at the target cell. PI widths are wider than CI widths because they must also cover the idiosyncratic component.

Both intervals use the percentile-t (reflective) method.

---

## Installation

**conda (recommended)**

```bash
git clone https://github.com/YousefKad/Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA.git
cd Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA
conda env create -f environment.yml
conda activate fbica
```

**pip**

```bash
git clone https://github.com/YousefKad/Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA.git
cd Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA
pip install -e .
```

---

## Usage

### Imputation

```python
from fbica import FBICA

# X has shape (T, N, m) with np.nan at missing entries
imp = FBICA(use_loo=True)
X_filled = imp.fit_transform(X)
```

### Bootstrap intervals

```python
from fbica import FBICABootstrap

# target_points: list of (t, i, b) index tuples for the cells to be imputed
bs = FBICABootstrap(interval_type="CI", B=499, alpha=0.05)
result = bs.fit(X_obs, target_points=[(5, 3, 0), (15, 10, 1)])

print(result.point_est)   # imputed values
print(result.lower)       # lower bounds
print(result.upper)       # upper bounds
```

Set `interval_type="PI"` for prediction intervals.

### Mixed-frequency panels

If one variable is low-frequency and the rest are high-frequency, restrict the factor proxies to the high-frequency variables:

```python
imp = FBICA(use_loo=True, factor_vars=list(range(1, 9)))
X_filled = imp.fit_transform(X)
```

### Expanding-window imputation

```python
X_rt = imp.fit_transform_expanding(X, min_window=60)
```

### Monte Carlo

```python
from fbica import run_simulation

res = run_simulation(N=50, T=100, m=4, r=3, phi=0.3, n_sim=500)
print(res["rmse"], res["bias"])
```

---

## Notebooks

| Notebook | Content |
|----------|---------|
| `notebooks/01_simulation_demo.ipynb` | Imputation demo, RMSE across (N, T) grids, LOO vs plain, spatial dependence |
| `notebooks/02_bootstrap_coverage.ipynb` | CI and PI coverage at N=T=25, replicating the Monte Carlo results in the paper |

---

## Package structure

```
fbica/
├── imputer.py       FBICA estimator
├── bootstrap.py     FBICABootstrap (CI and PI)
├── dgp.py           data-generating process
├── simulation.py    Monte Carlo helper
└── metrics.py       RMSE, MAE
```
