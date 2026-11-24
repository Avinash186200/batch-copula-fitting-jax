"""Synthetic data with known dependence.

* ``pairs``: N independent bivariate samples from given families and parameters, returned
  as pseudo-observations, for recovery, agreement, selection and speed runs.
* ``regime_returns``: a multi-asset daily return panel whose copula switches from a calm
  Gaussian regime (moderate correlation, no tail dependence) to a stress regime with strong
  lower-tail dependence (a multivariate Clayton copula: assets crash together) and back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from . import families as F


def pairs(specs, n: int, rng: np.random.Generator):
    """specs: list of (family, params). Returns pseudo-observations U, V of shape (N, n)."""
    U = np.empty((len(specs), n))
    V = np.empty((len(specs), n))
    for i, (name, params) in enumerate(specs):
        u, v = F.sample(name, params, n, rng)
        U[i], V[i] = F.pseudo_obs(u), F.pseudo_obs(v)
    return U, V


@dataclass(frozen=True)
class RegimeConfig:
    n_assets: int = 10
    days: int = 1500
    stress_start: int = 750
    stress_end: int = 1050          # exclusive
    calm_rho: float = 0.3           # Gaussian equicorrelation in calm days
    stress_theta: float = 2.0       # Clayton theta in stress days (tau 0.5, lambda_L 0.71)
    marginal_df: float = 5.0        # Student-t marginals
    calm_vol: float = 0.010
    stress_vol: float = 0.025

    def stress_mask(self) -> np.ndarray:
        d = np.arange(self.days)
        return (d >= self.stress_start) & (d < self.stress_end)


def _gaussian_equicorr(n: int, d: int, rho: float, rng) -> np.ndarray:
    common = rng.standard_normal((n, 1))
    z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * rng.standard_normal((n, d))
    return special.ndtr(z)


def _clayton_multi(n: int, d: int, theta: float, rng) -> np.ndarray:
    g = rng.gamma(1.0 / theta, 1.0, (n, 1))
    e = rng.exponential(1.0, (n, d))
    return (1.0 + e / g) ** (-1.0 / theta)


def regime_returns(cfg: RegimeConfig, rng: np.random.Generator) -> np.ndarray:
    """(days, n_assets) returns. Uniforms come from the regime's copula, then t marginals."""
    stress = cfg.stress_mask()
    u = np.empty((cfg.days, cfg.n_assets))
    u[~stress] = _gaussian_equicorr(int((~stress).sum()), cfg.n_assets, cfg.calm_rho, rng)
    u[stress] = _clayton_multi(int(stress.sum()), cfg.n_assets, cfg.stress_theta, rng)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    vol = np.where(stress, cfg.stress_vol, cfg.calm_vol)[:, None]
    scale = np.sqrt((cfg.marginal_df - 2.0) / cfg.marginal_df)     # unit-variance t
    return vol * scale * special.stdtrit(cfg.marginal_df, u)
