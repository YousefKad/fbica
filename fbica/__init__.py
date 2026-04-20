"""
fbica — Factor-Based Imputation via Cross-sectional Averages
=============================================================
Implements the FBI-CA estimator from:

    Bretschneider & Kaddoura (2025)
    "Factor-based imputation of missing values using cross-section averages"
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
