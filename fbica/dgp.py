import numpy as np

# DGP used in the notebooks and MC studies — swap in your own if needed.
def generate_panel(
    N: int,
    T: int,
    m: int,
    r: int,
    phi: float = 0.3,
    pi: float = 0.0,
    L: int = 4,
    seed_factors: int = 99,
    seed_errors: int = None,
    fix_factors_and_loadings: bool = False,
    _fixed_F=None,
    _fixed_lam=None,
):
    rng_fix = np.random.default_rng(seed_factors)

    if _fixed_F is not None:
        F = _fixed_F
    else:
        F = _draw_factors(T, r, phi, rng_fix)

    if _fixed_lam is not None:
        lam = _fixed_lam
    else:
        lam = _draw_loadings(N, m, r, rng_fix)

    rng_err = np.random.default_rng(seed_errors)
    nu = _draw_errors(T, N, m, phi, pi, L, rng_err)

    # X[t, i, b] = F[t, :] @ lam[i, b, :] + nu[t, i, b]
    common = np.einsum("tr,ibr->tib", F, lam)
    X = common + nu

    return X, F, lam, nu


def generate_missing(
    X: np.ndarray,
    miss_probs,
    forced_missing=None,
    seed: int = None,
):
    T, N, m = X.shape
    rng = np.random.default_rng(seed)

    probs_arr = np.atleast_1d(miss_probs)
    if probs_arr.size != 1 and probs_arr.size != m:
        raise ValueError(
            f"miss_probs must have length m={m} or be a scalar, "
            f"got length {probs_arr.size}."
        )
    probs = np.broadcast_to(probs_arr, (m,))

    miss = np.zeros((T, N, m), dtype=bool)
    for b in range(m):
        miss[:, :, b] = rng.random((T, N)) < probs[b]

    if forced_missing is not None:
        for (t, i, b) in forced_missing:
            miss[t, i, b] = True

    X_obs = X.copy().astype(float)
    X_obs[miss] = np.nan
    return X_obs, miss


def _draw_factors(T, r, phi, rng):
    F = np.zeros((T, r))
    u = rng.normal(0, 1, (T, r))
    F[0] = u[0]
    for t in range(1, T):
        F[t] = phi * F[t - 1] + u[t]
    return F


def _draw_loadings(N, m, r, rng):
    lam_mean = rng.normal(0, 1, (m, r))
    lam = np.zeros((N, m, r))
    for i in range(N):
        for b in range(m):
            lam[i, b] = lam_mean[b] + rng.normal(0, 0.5, r)
    return lam


def _draw_errors(T, N, m, phi, pi, L, rng):
    # AR(1) errors with optional spatial cross-unit spillovers; pad to avoid edge effects.
    pad = L + 1
    e = rng.normal(0, 1, (T, N + 2 * pad, m))
    nu = np.zeros((T, N, m))
    for i in range(N):
        i_pad = i + pad
        for b in range(m):
            for t in range(T):
                spatial = sum(
                    pi * e[t, i_pad + ell, b] + pi * e[t, i_pad - ell, b]
                    for ell in range(1, L + 1)
                )
                base = e[t, i_pad, b] + spatial
                if t == 0:
                    nu[t, i, b] = base
                else:
                    nu[t, i, b] = phi * nu[t - 1, i, b] + base
    return nu


def draw_fixed_factors_and_loadings(N, T, m, r, phi=0.3, seed=99):
    rng = np.random.default_rng(seed)
    F   = _draw_factors(T, r, phi, rng)
    lam = _draw_loadings(N, m, r, rng)
    return F, lam
