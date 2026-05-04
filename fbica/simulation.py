import numpy as np
from .imputer import FBICA
from .dgp import generate_panel, generate_missing, draw_fixed_factors_and_loadings



def run_simulation(
    N: int = 50,
    T: int = 50,
    m: int = 4,
    r: int = 3,
    phi: float = 0.3,
    pi: float = 0.0,
    miss_probs=None,
    miss_type: str = "mcar",
    n_sim: int = 500,
    use_loo: bool = True,
    factor_vars=None,
    fixed_points=None,
    seed_fixed: int = 99,
    verbose: bool = True,
) -> dict:
    if miss_probs is None:
        miss_probs = list(np.linspace(0.1, 0.2, m))
    miss_probs = np.atleast_1d(miss_probs)
    if miss_probs.size == 1:
        miss_probs = np.repeat(miss_probs, m)

    if fixed_points is None:
        t0 = min(4, T - 1)
        i0 = min(4, N - 1)
        fixed_points = [(t0, i0, 0)]

    n_pts = len(fixed_points)

    F_fixed, lam_fixed = draw_fixed_factors_and_loadings(N, T, m, r, phi, seed_fixed)

    true_C = np.array([
        F_fixed[t] @ lam_fixed[i, b] for (t, i, b) in fixed_points
    ])

    store = np.zeros((n_sim, n_pts))

    for sim in range(n_sim):
        if verbose and (sim % 100 == 0):
            print(f"  sim {sim}/{n_sim}  (N={N}, T={T})")

        X, _, _, _ = generate_panel(
            N, T, m, r, phi=phi, pi=pi,
            seed_errors=sim * 1000 + N * 7 + T * 3,
            _fixed_F=F_fixed, _fixed_lam=lam_fixed,
        )

        X_obs, _ = generate_missing(
            X, miss_probs=miss_probs,
            forced_missing=fixed_points,
            seed=sim,
        )

        imp = FBICA(use_loo=use_loo, factor_vars=factor_vars)
        X_filled = imp.fit_transform(X_obs)

        for j, (t, i, b) in enumerate(fixed_points):
            store[sim, j] = X_filled[t, i, b]

    mean_imp = store.mean(axis=0)
    rmse     = np.sqrt(((store - true_C[None, :]) ** 2).mean(axis=0))
    bias     = mean_imp - true_C

    return {
        "mean_imputed" : mean_imp,
        "true_C"       : true_C,
        "rmse"         : rmse,
        "bias"         : bias,
        "store"        : store,
        "fixed_points" : fixed_points,
        "params" : dict(
            N=N, T=T, m=m, r=r, phi=phi, pi=pi,
            miss_probs=miss_probs.tolist(),
            n_sim=n_sim, use_loo=use_loo,
        ),
    }


def compare_loo_vs_plain(
    N: int = 50,
    T: int = 50,
    m: int = 4,
    r: int = 3,
    phi: float = 0.3,
    pi: float = 0.0,
    miss_probs=None,
    n_sim: int = 500,
    seed_fixed: int = 99,
    verbose: bool = True,
) -> dict:
    kw = dict(N=N, T=T, m=m, r=r, phi=phi, pi=pi,
              miss_probs=miss_probs, n_sim=n_sim,
              seed_fixed=seed_fixed, verbose=verbose)

    print("=== LOO ===")
    res_loo   = run_simulation(**kw, use_loo=True)
    print("=== Plain (no LOO) ===")
    res_plain = run_simulation(**kw, use_loo=False)

    return {"loo": res_loo, "plain": res_plain}
