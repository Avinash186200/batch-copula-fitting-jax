"""Batch maximum-likelihood fitting of bivariate copulas: scipy reference, vectorised NumPy and JAX."""
from .engines import FitResult, fit
from .families import FAMILIES, FAMILY_NAMES, pseudo_obs

__all__ = ["FAMILIES", "FAMILY_NAMES", "FitResult", "fit", "pseudo_obs"]
__version__ = "1.0.0"
