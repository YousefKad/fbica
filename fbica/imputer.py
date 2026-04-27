import numpy as np
from tqdm import tqdm
from numbers import Integral


class FBICA:
    """
    FBI-CA imputer. Fits loadings by OLS on the LOO cross-sectional averages
    and fills missing cells with the fitted common component.

    use_loo=True is the default and what the paper uses. set False to use
    the full cross-section instead (faster but more biased in small samples).
    factor_vars lets you pick which variables go into the proxy, handy for
    mixed-frequency panels.
    """

    def __init__(self, use_loo=True, factor_vars=None):
        self.use_loo = use_loo
        self.factor_vars = factor_vars
        self.fhat_ = None
        self.lam_hat_ = None
        self.C_fit_ = None

    def fit_transform(self, X):
        X = self._check_X(X)
        T, N, m = X.shape
        fvar = self._fvar_idx(m)
        m_f = len(fvar)

        if self.use_loo and N < 2:
            raise ValueError(f"LOO needs N>=2, got N={N}.")

        X_f = X[:, :, fvar]
        fhat = self._proxy_loo(X_f, fvar) if self.use_loo else self._proxy_plain(X_f, fvar)
        self.fhat_ = fhat

        lam = np.empty((N, m_f, m))
        for i in range(N):
            fi = fhat[i] if self.use_loo else fhat
            for k in range(m):
                obs = ~np.isnan(X[:, i, k])
                n_obs = int(obs.sum())
                if n_obs <= m_f:
                    raise ValueError(
                        f"under-identified OLS at (i={i}, k={k}): {n_obs} obs vs {m_f} regressors. "
                        f"try min_window >= {3*m_f + 1}, or shrink factor_vars."
                    )
                lam[i, :, k] = self._ols(fi[obs], X[obs, i, k], i, k)
        self.lam_hat_ = lam

        # C[t,i,b] = fhat[i,t,:] @ lam[i,:,b]
        C = (np.einsum("idb,itd->tib", lam, fhat) if self.use_loo
             else np.einsum("idb,td->tib", lam, fhat))

        bad = np.isnan(X) & ~np.isfinite(C)
        if bad.any():
            t, i, k = np.argwhere(bad)[0]
            raise ValueError(f"common component non-finite at ({t},{i},{k}); cannot impute.")
        self.C_fit_ = C

        out = np.where(np.isnan(X), C, X)
        assert not np.isnan(out).any(), "NaN survived imputation"
        return out

    def fit_transform_expanding(self, X, min_window=60, verbose=True):
        X = self._check_X(X)
        T = X.shape[0]
        if not isinstance(min_window, Integral) or isinstance(min_window, bool):
            raise TypeError("min_window must be an int.")
        if not 1 <= min_window <= T:
            raise ValueError(f"min_window={min_window} not in [1, T={T}].")

        out = np.full(X.shape, np.nan)
        out[:min_window] = self.fit_transform(X[:min_window])

        itr = range(min_window, T)
        if verbose:
            itr = tqdm(itr, desc="Expanding window")
        for t in itr:
            sub = FBICA(use_loo=self.use_loo, factor_vars=self.factor_vars)
            out[t] = sub.fit_transform(X[:t + 1])[t]

        # re-fit on the full sample so fhat_/lam_hat_/C_fit_ reflect everything
        full = FBICA(use_loo=self.use_loo, factor_vars=self.factor_vars)
        full.fit_transform(X)
        self.fhat_, self.lam_hat_, self.C_fit_ = full.fhat_, full.lam_hat_, full.C_fit_
        return out

    def get_common_component(self):
        if self.C_fit_ is None:
            raise RuntimeError("fit first.")
        return self.C_fit_

    def get_residuals(self, X):
        if self.C_fit_ is None:
            raise RuntimeError("fit first.")
        X = self._check_X(X)
        if X.shape != self.C_fit_.shape:
            raise ValueError(f"shape {X.shape} != fitted {self.C_fit_.shape}.")
        return np.where(~np.isnan(X), X - self.C_fit_, np.nan)

    def _proxy_plain(self, X_f, fvar):
        all_nan = np.isnan(X_f).all(axis=1)
        if all_nan.any():
            t, j = np.argwhere(all_nan)[0]
            raise ValueError(
                f"plain proxy: every unit missing at t={t}, var={fvar[j]}. use LOO or drop that period."
            )
        return np.nanmean(X_f, axis=1)

    def _proxy_loo(self, X_f, fvar):
        # subtract unit i's contribution from the total before dividing
        N = X_f.shape[1]
        obs = (~np.isnan(X_f)).astype(float)
        Xz = np.where(np.isnan(X_f), 0.0, X_f)
        s_all = Xz.sum(axis=1)
        n_all = obs.sum(axis=1)

        out = np.empty((N,) + s_all.shape)
        for i in range(N):
            n_i = n_all - obs[:, i, :]
            if (n_i <= 0).any():
                t, j = np.argwhere(n_i <= 0)[0]
                raise ValueError(
                    f"LOO proxy undefined for i={i} at (t={t}, var={fvar[j]}): no peers left."
                )
            out[i] = (s_all - Xz[:, i, :]) / n_i
        return out

    def _ols(self, F, y, i, k):
        if np.linalg.matrix_rank(F) < F.shape[1]:
            raise ValueError(f"rank-deficient F at (i={i}, k={k}).")
        beta, *_ = np.linalg.lstsq(F, y, rcond=None)
        if not np.isfinite(beta).all():
            raise ValueError(f"non-finite OLS solution at (i={i}, k={k}).")
        return beta

    def _fvar_idx(self, m):
        if self.factor_vars is None:
            return list(range(m))
        idx = list(self.factor_vars)
        if not idx:
            raise ValueError("factor_vars is empty.")
        if len(set(idx)) != len(idx):
            raise ValueError("factor_vars has duplicates.")
        for j in idx:
            if not isinstance(j, (int, np.integer)) or not 0 <= j < m:
                raise ValueError(f"bad factor_vars entry: {j!r} (m={m}).")
        return idx

    def _check_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3 or min(X.shape) < 1:
            raise ValueError(f"X must be 3-D with positive dims, got {X.shape}.")
        if np.isinf(X).any():
            raise ValueError("X contains inf; only finite values and NaN are allowed.")
        return X
