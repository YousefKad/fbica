# fbica — Factor-Based Imputation via Cross-Sectional Averages

Implementation of the FBI-CA estimator from:

> Bretschneider, T. & Kaddoura, Y. (2025).  
> *Factor-based imputation of missing values using cross-sectional averages.*

---

## Method

FBI-CA imputes missing entries in a panel $X$ of dimension $T \times N \times m$ by approximating the latent common factors with cross-sectional averages. For a missing entry $x_{i,t,b}$, the imputed value is the fitted common component

$$
\widetilde{C}_{i,t,b} = \hat{\lambda}_{i,b}'\, \hat{f}_{-i,t},
$$

where $\hat{f}_{-i,t}$ is the vector of cross-sectional averages across variables at time $t$, excluding unit $i$ (the leave-one-out form), and $\hat{\lambda}_{i,b}$ is estimated by least squares on the observed time periods for unit $i$, variable $b$.

**Algorithm.** For each unit $i$ and variable $b$:

1. Compute the LOO factor proxy at each $t$:
$$\hat{f}_{-i,t} = \left(\bar{x}_{-i,t,1},\dots,\bar{x}_{-i,t,m}\right)', \quad \bar{x}_{-i,t,b} = \frac{1}{N_{b,t}}\sum_{j \neq i} d_{j,t,b}\, x_{j,t,b}.$$

2. Estimate loadings from observed periods:
$$\hat{\lambda}_{i,b} = \Bigl(\sum_{s:\, d_{i,s,b}=1} \hat{f}_{-i,s}\hat{f}_{-i,s}'\Bigr)^{-1} \Bigl(\sum_{s:\, d_{i,s,b}=1} \hat{f}_{-i,s}\, x_{i,s,b}\Bigr).$$

3. Impute: $\widetilde{C}_{i,t,b} = \hat{\lambda}_{i,b}'\hat{f}_{-i,t}$ for all missing $(i,t,b)$.

Setting `use_loo=False` uses the full cross-section instead of the LOO form.

---

## Bootstrap inference

The package provides two types of intervals for a missing cell $(i,t,b)$, via `FBICABootstrap`.

**Confidence interval (CI)** for the common component $C_{i,t,b} = \lambda_{i,b}'f_t$.  
Uses a block-wild bootstrap: cross-sectional indices are resampled with replacement, and the time-series residuals are multiplied by Rademacher weights drawn in non-overlapping blocks of length $\lceil T^{1/3}\rceil$. This accounts for temporal dependence in the errors and factors.

**Prediction interval (PI)** for the realised value $x_{i,t,b} = C_{i,t,b} + \nu_{i,t,b}$.  
Uses an iid pairs bootstrap: cross-sectional indices are resampled and centred residuals are drawn with replacement to mimic the idiosyncratic shock at the target cell. PI widths are wider than CI widths because they must also cover the idiosyncratic component.

Both intervals are constructed by the percentile-t (reflective) method.

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

If variable `0` is low-frequency and variables `1, ..., 8` are high-frequency, restrict the factor proxies to the high-frequency variables:

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
| `notebooks/01_simulation_demo.ipynb` | Imputation demo, RMSE across $(N,T)$ grids, LOO vs plain, spatial dependence |
| `notebooks/02_bootstrap_coverage.ipynb` | CI and PI coverage at $N=T=25$, replicating the Monte Carlo results in the paper |

---

