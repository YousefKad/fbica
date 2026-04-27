import numpy as np
from tqdm import tqdm
from numbers import Integral


class FBICA:
    """
    Factor-Based Imputation via Cross-sectional Averages (FBI-CA).
    use_loo : bool  — exclude unit i 
    factor_vars : list of int or None 
    """

    def __init__(self, use_loo=True, factor_vars=None):
        self.use_loo     = use_loo
        self.factor_vars = factor_vars
        self.fhat_    = None
        self.lam_hat_ = None
        self.C_fit_   = None

    def __repr__(self):
        status = "[fitted]" if self.C_fit_ is not None else "[not fitted]"
        fv = "all" if self.factor_vars is None else str(self.factor_vars)
        return f"FBICA(use_loo={self.use_loo}, factor_vars={fv}) {status}"

    def fit_transform(self, X):
        X = self._validate(X, "fit_transform")
        return self._fit_loo(X) if self.use_loo else self._fit_plain(X)

    def fit_transform_expanding(self, X, min_window=60, verbose=True):
        X = self._validate(X, "fit_transform_expanding")
        T = X.shape[0]

        if not isinstance(min_window, Integral) or isinstance(min_window, bool):
            raise TypeError(f"min_window must be an integer, got {type(min_window).__name__}.")
        if not (1 <= min_window <= T):
            raise ValueError(f"min_window={min_window} must be between 1 and T={T}.")

        X_rt = np.full(X.shape, np.nan, dtype=float)
        X_rt[:min_window] = self.fit_transform(X[:min_window])

        iterator = range(min_window, T)
        if verbose:
            iterator = tqdm(iterator, desc="Expanding window")

        for t in iterator:
            imp_t = FBICA(use_loo=self.use_loo, factor_vars=self.factor_vars)
            X_rt[t] = imp_t.fit_transform(X[:t + 1])[t]

        imp_full = FBICA(use_loo=self.use_loo, factor_vars=self.factor_vars)
        imp_full.fit_transform(X)
        self.fhat_    = imp_full.fhat_
        self.lam_hat_ = imp_full.lam_hat_
        self.C_fit_   = imp_full.C_fit_

        return X_rt

    def get_common_component(self):
        if self.C_fit_ is None:
            raise RuntimeError("Call fit_transform() first.")
        return self.C_fit_

    def get_residuals(self, X):
        if self.C_fit_ is None:
            raise RuntimeError("Call fit_transform() first.")
        X = self._validate(X, "get_residuals")
        if X.shape != self.C_fit_.shape:
            raise ValueError(
                f"Shape mismatch: X is {X.shape} but fitted component is {self.C_fit_.shape}."
            )
        return np.where(~np.isnan(X), X - self.C_fit_, np.nan)

    def _fit_plain(self, X):
        T, N, m  = X.shape
        fvar_idx = self._factor_var_idx(m)
        m_f      = len(fvar_idx)
        X_f      = X[:, :, fvar_idx]

        self._check_plain_proxy(X_f, fvar_idx)

        fhat = np.nanmean(X_f, axis=1)
        self._check_finite(fhat, "factor proxy fhat")
        self.fhat_ = fhat

        lam_hat = np.empty((N, m_f, m), dtype=float)
        for i in range(N):
            for k in range(m):
                obs   = ~np.isnan(X[:, i, k])
                n_obs = int(obs.sum())
                if n_obs <= m_f:
                    raise ValueError(self._underid_msg("plain", i, k, n_obs, m_f))
                lam_hat[i, :, k] = self._ols(fhat[obs], X[obs, i, k], i, k, "plain")

        self._check_finite(lam_hat, "loading tensor lam_hat")
        self.lam_hat_ = lam_hat

        C_fit = np.einsum("idb,td->tib", lam_hat, fhat)
        self._check_finite(C_fit, "common component C_fit")
        self._check_all_imputed(X, C_fit)
        self.C_fit_ = C_fit

        X_filled = np.where(np.isnan(X), C_fit, X)
        self._check_no_nan(X_filled)
        return X_filled

    def _fit_loo(self, X):
        T, N, m  = X.shape
        fvar_idx = self._factor_var_idx(m)
        m_f      = len(fvar_idx)

        if N < 2:
            raise ValueError(f"LOO requires at least N=2 units, got N={N}.")

        X_f      = X[:, :, fvar_idx]
        obs_f    = (~np.isnan(X_f)).astype(float)
        X_f_safe = np.where(~np.isnan(X_f), X_f, 0.0)

        full_sum   = X_f_safe.sum(axis=1)
        full_count = obs_f.sum(axis=1)

        fhat_loo = np.empty((N, T, m_f), dtype=float)
        for i in range(N):
            i_vals = np.where(np.isnan(X_f[:, i, :]), 0.0, X_f[:, i, :])
            i_obs  = (~np.isnan(X_f[:, i, :])).astype(float)
            loo_sum   = full_sum   - i_vals
            loo_count = full_count - i_obs
            self._check_loo_proxy(loo_count, full_count, i_obs, i, fvar_idx)
            fhat_loo[i] = loo_sum / loo_count

        self._check_finite(fhat_loo, "LOO factor proxy fhat_loo")
        self.fhat_ = fhat_loo

        lam_hat = np.empty((N, m_f, m), dtype=float)
        for i in range(N):
            fi = fhat_loo[i]
            for k in range(m):
                obs   = ~np.isnan(X[:, i, k])
                n_obs = int(obs.sum())
                if n_obs <= m_f:
                    raise ValueError(self._underid_msg("LOO", i, k, n_obs, m_f))
                lam_hat[i, :, k] = self._ols(fi[obs], X[obs, i, k], i, k, "LOO")

        self._check_finite(lam_hat, "LOO loading tensor lam_hat")
        self.lam_hat_ = lam_hat

        # C_fit[t,i,b] = lam_hat[i,:,b] @ fhat_loo[i,t,:]
        C_fit = np.einsum("idb,itd->tib", lam_hat, fhat_loo)
        self._check_finite(C_fit, "LOO common component C_fit")
        self._check_all_imputed(X, C_fit)
        self.C_fit_ = C_fit

        X_filled = np.where(np.isnan(X), C_fit, X)
        self._check_no_nan(X_filled)
        return X_filled

    def _factor_var_idx(self, m):
        if self.factor_vars is None:
            return list(range(m))
        idx = list(self.factor_vars)
        if len(idx) == 0:
            raise ValueError("factor_vars cannot be empty.")
        for pos, j in enumerate(idx):
            if not isinstance(j, (int, np.integer)):
                raise TypeError(f"factor_vars[{pos}]={j!r} is not an integer.")
            if not (0 <= j < m):
                raise ValueError(f"factor_vars index {j} out of range for m={m}.")
        if len(set(idx)) != len(idx):
            raise ValueError("factor_vars has duplicate indices.")
        return idx

    def _ols(self, f_sub, z_sub, i, k, mode):
        n, p = f_sub.shape
        if n <= p:
            raise ValueError(
                f"{mode} loading (i={i}, k={k}): {n} observed rows but {p} regressors. OLS not identified."
            )
        self._check_finite(f_sub, f"factor matrix (i={i}, k={k})")
        self._check_finite(z_sub, f"response (i={i}, k={k})")
        if np.linalg.matrix_rank(f_sub) < p:
            raise ValueError(
                f"Rank-deficient factor matrix for {mode} loading (i={i}, k={k}).  "
                "Consider reducing factor_vars to remove collinear proxies."
            )
        try:
            beta = np.linalg.lstsq(f_sub, z_sub, rcond=None)[0]
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"lstsq failed for {mode} loading (i={i}, k={k})."
            ) from exc
        self._check_finite(beta, f"{mode} loading (i={i}, k={k})")
        return beta

    def _validate(self, X, caller):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{caller}: X must be a numeric array.") from exc
        if X.ndim != 3:
            raise ValueError(
                f"{caller}: X must be 3-dimensional (T, N, m), got shape {X.shape}."
            )
        if min(X.shape) < 1:
            raise ValueError(f"{caller}: all dimensions must be positive, got {X.shape}.")
        if np.any(np.isinf(X)):
            t, i, k = np.argwhere(np.isinf(X))[0]
            raise ValueError(f"{caller}: inf at (t={t}, i={i}, k={k}) — only finite values and NaN are allowed.")
        return X

    def _check_plain_proxy(self, X_f, fvar_idx):
        all_missing = np.isnan(X_f).all(axis=1)
        if not np.any(all_missing):
            return
        bad = np.argwhere(all_missing)
        examples = [f"(t={t}, var={fvar_idx[j]})" for t, j in bad[:5]]
        raise ValueError(
            "Plain FBI-CA: all units missing for some (t, factor_var). Cannot compute cross-sectional mean.  "
            "Locations: " + ", ".join(examples) + "."
        )

    def _check_loo_proxy(self, loo_count, full_count, i_obs, i, fvar_idx):
        bad = np.argwhere(loo_count <= 0)
        if len(bad) == 0:
            return
        examples = []
        for t, lj in bad[:5]:
            orig = fvar_idx[lj]
            if full_count[t, lj] == 0:
                why = "all units missing"
            elif full_count[t, lj] == 1 and i_obs[t, lj] == 1:
                why = f"unit {i} is the only observed unit"
            else:
                why = "no peers left after LOO exclusion"
            examples.append(f"(i={i}, t={t}, var={orig}: {why})")
        raise ValueError(
            "LOO factor proxy undefined. No peers remain after excluding the target unit.  "
            + ", ".join(examples) + "."
        )

    def _check_all_imputed(self, X, C_fit):
        mask = np.isnan(X) & ~np.isfinite(C_fit)
        if np.any(mask):
            t, i, k = np.argwhere(mask)[0]
            raise ValueError(
                f"Missing entry at (t={t}, i={i}, k={k}) would remain unimputed. C_fit is not finite there."
            )

    def _check_no_nan(self, X_filled):
        if np.any(np.isnan(X_filled)):
            t, i, k = np.argwhere(np.isnan(X_filled))[0]
            raise ValueError(
                f"NaN remains in the filled array at (t={t}, i={i}, k={k}). likely an error."
            )

    def _check_finite(self, arr, name):
        bad = ~np.isfinite(arr)
        if np.any(bad):
            idx   = tuple(np.argwhere(bad)[0])
            value = arr[idx]
            raise ValueError(f"{name} has a non-finite value ({value}) at index {idx}.")

    def _underid_msg(self, mode, i, k, n_obs, m_f):
        return (
            f"{mode} loading (i={i}, k={k}): {n_obs} observed rows but {m_f} regressors. OLS not identified.  "
            f"In mixed-frequency settings try increasing min_window to at least {m_f * 3 + 1} periods, "
            "or reduce factor_vars."
        )
