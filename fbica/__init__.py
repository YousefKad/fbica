"""
fbica — Factor-Based Imputation via Cross-sectional Averages
=============================================================

Implements the FBI-CA estimator from:

    Bretschneider & Kaddoura (2025)
    "Factor-based imputation of missing values using cross-section averages"

Cross-sectional averages proxy latent common factors.  Missing entries are
imputed with the fitted common component C_{i,t,b} = lambda_hat_{i,b}' f_hat_t.
The leave-one-out (LOO) variant excludes unit i from its own factor proxy,
reducing finite-sample bias when m > r.

Public API
----------
FBICA            : main imputer (LOO or plain, full or expanding window)
FBICABootstrap   : bootstrap CI and PI for imputed values
BootstrapResult  : dataclass returned by FBICABootstrap.fit()
run_simulation   : Monte-Carlo helper
generate_panel   : DGP helper
generate_missing : missingness helper
"""

from .imputer   import FBICA
from .bootstrap import FBICABootstrap, BootstrapResult
from .simulation import run_simulation
from .dgp import generate_panel, generate_missing

__version__ = "0.1.0"
__author__  = "Tilman Bretschneider & Yousef Kaddoura"
__all__     = [
    "FBICA", "FBICABootstrap", "BootstrapResult",
    "run_simulation", "generate_panel", "generate_missing",
]
