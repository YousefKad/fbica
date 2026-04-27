# fbica — Factor-Based Imputation via Cross-Sectional Averages

Implementation of the FBI-CA estimator from:

> Bretschneider, T. & Kaddoura, Y. (2026).  
> *Factor-based imputation of missing values using cross-sectional averages.*

---

## Method

FBI-CA imputes missing entries in a panel of dimension \(T \times N \times m\) by using cross-sectional averages as proxies for the latent common factors. For each missing cell, the imputed value is the fitted common component

\[
C_{i,t,b} = \lambda_{i,b}' f_t
\]

where the factor proxy \(f_t\) is the vector of cross-sectional averages across variables at time \(t\). In the leave-one-out (LOO) form, unit \(i\) is excluded when constructing its own factor proxy, which reduces finite-sample bias when the number of variables exceeds the number of true factors.

**Algorithm.** For each unit \(i\) and variable \(b\):

1. Compute the LOO cross-sectional average for each variable and time period, using all observed units except \(i\).
2. Stack these averages into the factor proxy vector.
3. Estimate loadings by least squares on the observed time periods for \((i,b)\).
4. Impute all missing entries for \((i,b)\) using the fitted factor proxy and estimated loadings.

Setting `use_loo=False` uses the full cross-section instead.

---

## Bootstrap inference

The package provides two types of intervals for a missing cell, via `FBICABootstrap`.

**Confidence interval (CI)** targets the common component. Uses a block-wild bootstrap: cross-sectional indices are resampled with replacement, and time-series residuals are multiplied by Rademacher weights drawn in non-overlapping blocks of length \(\lceil T^{1/3} \rceil\). This accounts for temporal dependence.

**Prediction interval (PI)** targets the realised value, including the idiosyncratic error. Uses an iid pairs bootstrap where centred residuals are resampled to mimic the idiosyncratic shock at the target cell. PI widths are wider than CI widths because they must also cover the idiosyncratic component.

Both intervals use the percentile-\(t\), or reflective, method.

---

## Installation

**conda (recommended)**

```bash
git clone https://github.com/YousefKad/Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA.git
cd Imputation-of-missing-values-using-cross-section-averages-via-FBI-CA
conda env create -f environment.yml
conda activate fbica
