import warnings
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .imputer import FBICA


@dataclass
class BootstrapResult:
    point_est: np.ndarray  # (n_targets,)
    lower: np.ndarray
    upper: np.ndarray
    draws: np.ndarray      # (B, n_targets)


class FBICABootstrap:
    """Bootstrap CI/PI for FBI-CA imputations.

    interval_type="CI" uses a block-wild bootstrap (Algorithm 1).
    interval_type="PI" uses an iid pairs bootstrap (Algorithm 2).
    """

    def __init__(self, interval_type="CI", use_loo=True, factor_vars=None,
                 always_observed=None, block_length="auto", B=499, alpha=0.05, seed=None):
        if interval_type not in ("CI", "PI"):
            raise ValueError("interval_type must be 'CI' or 'PI'.")
        if not isinstance(B, Integral) or isinstance(B, bool) or B <= 0:
            raise ValueError("B must be a positive integer.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")

        self.interval_type = interval_type
        self.use_loo = use_loo
        self.factor_vars = factor_vars
        self.always_observed = always_observed
        self.block_length = block_length
        self.B = B
        self.alpha = alpha
        self.seed = seed

    def fit(self, X, target_points):
        X = np.asarray(X, dtype=float)
        T, N, m = X.shape
        rng = np.random.default_rng(self.seed)

        target_points = self._check_targets(target_points, T, N, m)

        imp = FBICA(use_loo=self.use_loo, factor_vars=self.factor_vars,
                    always_observed=self.always_observed)
        X_filled = imp.fit_transform(X)
        point_est = np.array([X_filled[t, i, k] for t, i, k in target_points])
        resid_c = self._centred_residuals(X, imp)

        fvar_idx = imp._fvar_idx(m)
        ao = None if self.always_observed is None else np.asarray(list(self.always_observed), dtype=int)

        if self.interval_type == "CI":
            draws = self._loop_ci(X, imp, resid_c, target_points, fvar_idx, ao, rng)
        else:
            draws = self._loop_pi(X, imp, resid_c, target_points, fvar_idx, ao, rng)

        # check we actually got draws
        valid = np.isfinite(draws).sum(axis=0)
        if (valid == 0).any():
            j = int(np.where(valid == 0)[0][0])
            raise ValueError(f"no valid bootstrap draws for target index {j}.")
        low = max(10, int(np.ceil(0.1 * self.B)))
        if (valid < low).any():
            j = int(np.where(valid < low)[0][0])
            warnings.warn(f"few valid draws for target {j}: {int(valid[j])}/{self.B}.",
                          RuntimeWarning, stacklevel=2)

        a = self.alpha
        q_lo = np.nanquantile(draws, a / 2, axis=0)
        q_hi = np.nanquantile(draws, 1 - a / 2, axis=0)

        return BootstrapResult(
            point_est=point_est,
            lower=2 * point_est - q_hi,
            upper=2 * point_est - q_lo,
            draws=draws,
        )

    def _loop_ci(self, X, imp, resid_c, target_points, fvar_idx, ao, rng):
        T, N, m = X.shape
        lam = imp.lam_hat_
        obs_mask = ~np.isnan(X)

        L = self._block_len(T)
        blocks = [np.arange(t, min(t + L, T)) for t in range(0, T, L)]

        unique_ik = list({(i, k) for _, i, k in target_points})
        ik_idx = {ik: j for j, ik in enumerate(unique_ik)}

        draws = np.full((self.B, len(target_points)), np.nan)

        for b in range(self.B):
            boot_idx = self._draw_idx(rng, N, ao)

            eta_blk = rng.choice([-1.0, 1.0], size=(len(blocks), len(unique_ik)))
            eta_full = np.empty((T, len(unique_ik)))
            for bl, times in enumerate(blocks):
                eta_full[times, :] = eta_blk[bl, :]

            for j, (t_pt, i_pt, k_pt) in enumerate(target_points):
                obs_ik = obs_mask[:, i_pt, k_pt]
                f_star = self._f_star(X, boot_idx, i_pt, fvar_idx, ao)
                if f_star is None:
                    continue

                n_f = f_star.shape[1]
                if obs_ik.sum() <= n_f:
                    continue

                tu = ik_idx[(i_pt, k_pt)]
                lam_ik = lam[i_pt, :, k_pt]
                z_star = (f_star[obs_ik, :] @ lam_ik
                          + eta_full[obs_ik, tu] * resid_c[obs_ik, i_pt, k_pt])

                f_sub = f_star[obs_ik, :]
                if np.linalg.matrix_rank(f_sub) < n_f:
                    continue
                try:
                    lam_star = np.linalg.lstsq(f_sub, z_star, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                draws[b, j] = f_star[t_pt, :] @ lam_star

        return draws

    def _loop_pi(self, X, imp, resid_c, target_points, fvar_idx, ao, rng):
        T, N, m = X.shape
        lam = imp.lam_hat_
        obs_mask = ~np.isnan(X)

        draws = np.full((self.B, len(target_points)), np.nan)

        for b in range(self.B):
            boot_idx = self._draw_idx(rng, N, ao)

            for j, (t_pt, i_pt, k_pt) in enumerate(target_points):
                obs_ik = obs_mask[:, i_pt, k_pt]
                f_star = self._f_star(X, boot_idx, i_pt, fvar_idx, ao)
                if f_star is None:
                    continue

                n_f = f_star.shape[1]
                if obs_ik.sum() <= n_f:
                    continue

                res_pool = resid_c[obs_ik, i_pt, k_pt]
                lam_ik = lam[i_pt, :, k_pt]

                e_star = rng.choice(res_pool, size=obs_ik.sum(), replace=True)
                z_star = f_star[obs_ik, :] @ lam_ik + e_star

                f_sub = f_star[obs_ik, :]
                if np.linalg.matrix_rank(f_sub) < n_f:
                    continue
                try:
                    lam_star = np.linalg.lstsq(f_sub, z_star, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                draws[b, j] = f_star[t_pt, :] @ lam_star + rng.choice(res_pool)

        return draws

    def _draw_idx(self, rng, N, ao):
        if ao is not None:
            return ao[rng.integers(0, len(ao), size=len(ao))]
        return rng.integers(0, N, size=N)

    def _f_star(self, X, boot_idx, i_pt, fvar_idx, ao):
        if ao is not None:
            f = np.nanmean(X[:, boot_idx, :][:, :, fvar_idx], axis=1)
        elif self.use_loo:
            mask = boot_idx != i_pt
            if mask.sum() == 0:
                return None
            f = np.nanmean(X[:, boot_idx[mask], :][:, :, fvar_idx], axis=1)
        else:
            f = np.nanmean(X[:, boot_idx, :][:, :, fvar_idx], axis=1)

        if not np.isfinite(f).all():
            return None
        return f

    def _centred_residuals(self, X, imp):
        resid = imp.get_residuals(X)
        resid_c = resid.copy()
        _, N, m = X.shape
        obs = ~np.isnan(X)
        for i in range(N):
            for k in range(m):
                mask = obs[:, i, k]
                if mask.any():
                    resid_c[mask, i, k] -= np.nanmean(resid[mask, i, k])
        return resid_c

    def _check_targets(self, target_points, T, N, m):
        out = []
        for p in target_points:
            t, i, k = p
            if not (0 <= t < T and 0 <= i < N and 0 <= k < m):
                raise ValueError(f"target {(t, i, k)} out of bounds for X of shape {(T, N, m)}.")
            out.append((int(t), int(i), int(k)))
        if not out:
            raise ValueError("target_points is empty.")
        return out

    def _block_len(self, T):
        if self.block_length == "auto":
            return max(1, int(np.ceil(T ** (1 / 3))))
        try:
            L = int(self.block_length)
        except (TypeError, ValueError) as exc:
            raise ValueError("block_length must be 'auto' or a positive integer.") from exc
        if L < 1:
            raise ValueError("block_length must be >= 1.")
        return min(L, T)
