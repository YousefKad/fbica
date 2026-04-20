# fbica — Factor-Based Imputation via Cross-Sectional Averages

  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`fbica` implements **FBI-CA** (*Factor-Based Imputation via Cross-Sectional Averages*), the estimator developed in:

> Bretschneider, T. & Kaddoura, Y. (2025).  
> *Factor-based imputation of missing values using cross-sectional averages.*

The package is designed for **panel data with missing values**, including settings with general missing patterns, mixed-frequency variables, ragged-edge panels, and expanding-window real-time imputation.

---

## What FBI-CA does

FBI-CA uses **cross-sectional averages** as proxies for latent common factors. It then estimates unit- and variable-specific loadings by projection and imputes missing entries using the fitted common component.

For a missing entry $x_{i,t,b}$, the imputed common component is

$$
\widetilde{C}_{i,t,b} = \hat{\lambda}_{i,b}' \hat{f}_{-i,t}.
$$

When `use_loo=True`, the factor proxy is computed in **leave-one-out (LOO)** form, excluding unit $i$ when imputing $x_{i,t,b}$. This helps reduce finite-sample bias, especially when the number of proxies exceeds the true number of factors.

---

## The algorithm

For each variable $b = 1, \dots, m$, unit $i = 1, \dots, N$, and time $t = 1, \dots, T$:

### 1. Estimate factors using cross-sectional averages

For each time period, compute cross-sectional averages using the observed units:


$$\bar{x}_{-i,t,b} = \frac{1}{N_{b,t}} \sum_{j \neq i} d_{j,t,b} x_{j,t,b}, $$

and stack them into the factor proxy

$$
\hat{f}_{-i,t}
= \left(
\bar{x}_{-i,t,1}, \dots, \bar{x}_{-i,t,m}
\right)'. $$

If `use_loo=False`, the full cross-section is used instead.

### 2. Estimate loadings by least squares

For each unit-variable pair $(i,b)$, estimate loadings from the observed time periods:

$$
\hat{\lambda}_{i,b}
= \left(
\sum_{s=1}^T d_{i,s,b}\hat{f}_{-i,s}\hat{f}_{-i,s}'
\right)^{-1}
\left(
\sum_{s=1}^T d_{i,s,b}\hat{f}_{-i,s}x_{i,s,b}
\right).
$$

### 3. Impute missing values

Construct the fitted common component

$$
\widetilde{C}_{i,t,b} = \hat{\lambda}_{i,b}' \hat{f}_{-i,t},
$$

and use it to fill in missing entries. Observed values are left unchanged.

---

## Installation

### Option 1 — conda (recommended)

```bash
git clone https://github.com/YousefKad/Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA.git
cd Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA
conda env create -f environment.yml
conda activate fbica
```

The `environment.yml` creates an environment called `fbica`, installs all dependencies, and runs `pip install -e .` automatically so the package is importable from any notebook or script.

### Option 2 — pip + virtualenv

```bash
git clone https://github.com/YousefKad/Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA.git
cd Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`[dev]` pulls in Jupyter, notebook, and ipykernel on top of the core dependencies.

---

## Quick start

### Basic imputation

```python
from fbica import FBICA

# X has shape (T, N, m) with np.nan for missing entries
imp = FBICA(use_loo=True)
X_filled = imp.fit_transform(X)
```

### Compare LOO and plain FBI-CA

```python
from fbica import FBICA

imp_loo = FBICA(use_loo=True)
imp_plain = FBICA(use_loo=False)

X_loo = imp_loo.fit_transform(X)
X_plain = imp_plain.fit_transform(X)
```

### Mixed-frequency imputation

If variable `0` is low frequency and variables `1, ..., 8` are high frequency, use only the high-frequency variables to build factor proxies:

```python
imp = FBICA(use_loo=True, factor_vars=list(range(1, 9)))
X_filled = imp.fit_transform(X)
```

### Expanding-window imputation

```python
imp = FBICA(use_loo=True, factor_vars=list(range(1, 9)))
X_rt = imp.fit_transform_expanding(X, min_window=60)
```

### Monte Carlo simulation

```python
from fbica import run_simulation

results = run_simulation(
    N=50, T=100, m=4, r=3,
    phi=0.3, pi=0.0,
    miss_probs=[0.1, 0.13, 0.16, 0.20],
    n_sim=500,
    use_loo=True,
)

print("True C :", results["true_C"])
print("RMSE   :", results["rmse"])
print("Bias   :", results["bias"])
```

---

## Main options

| Argument | Description |
|---|---|
| `use_loo` | Use leave-one-out factor proxies |
| `factor_vars` | Indices of variables used to construct factor proxies |
| `min_window` | Minimum initial window for expanding-window imputation |

---

## Package structure

```text
fbica/
├── __init__.py
├── imputer.py       # FBICA estimator
├── dgp.py           # data-generating processes
├── simulation.py    # Monte Carlo routines
└── metrics.py       # RMSE, MAE, R²

notebooks/
├── 01_simulation_demo.ipynb
└── 02_gdp_nowcasting.ipynb

data/
├── SQGDP1__ALL_AREAS_2005_2024.csv
└── Employment-SeasonalAdj.xlsx
```

---

## Use cases

`fbica` is particularly useful for:
- incomplete multi-variable panels,
- mixed-frequency macroeconomic data,
- ragged-edge nowcasting exercises,
- simulation studies on factor-based imputation.

---

## License

MIT — see [LICENSE](LICENSE).
