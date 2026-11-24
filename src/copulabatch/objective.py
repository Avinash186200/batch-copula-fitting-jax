"""Per-family objectives (mean negative log-likelihood) and the fitting recipe, written once.

``u`` and ``v`` are pseudo-observations whose LAST axis is the sample: shape (n,) for one
pair (scipy reference, or one lane under ``jax.vmap``) or (N, n) for the vectorised NumPy
engine. Parameters broadcast against the leading axes. Transforms that do not depend on the
parameter (rotation, logs, normal scores) are computed once per family, outside the solver.

``fit_families`` runs the full recipe for every family with any 1-D minimiser
``minimize(f, lower, upper) -> (x, fx, converged)``; the NumPy and JAX engines pass their
own Brent.
"""
from __future__ import annotations

import math

from .families import (
    FAMILIES,
    NU_MAX,
    NU_MIN,
    RHO_MAX,
    Family,
    Ops,
    clayton_logc,
    frank_logc,
    gauss_logc,
    get,
    gumbel_logc,
    rotate_uv,
    t_logc,
)

LOG_NU_MIN, LOG_NU_MAX = math.log(NU_MIN), math.log(NU_MAX)


def _col(theta, xp):
    return xp.asarray(theta)[..., None]


def make_nll(fam: Family, u, v, ops: Ops):
    """f(theta) -> mean negative log-likelihood for a one-parameter family."""
    xp = ops.xp
    ub, vb = rotate_uv(u, v, fam.rotation)
    if fam.base == "gaussian":
        x, y = ops.ndtri(ub), ops.ndtri(vb)
        return lambda th: -xp.mean(gauss_logc(_col(th, xp), x, y, xp), axis=-1)
    if fam.base in ("clayton", "gumbel"):
        lu, lv = xp.log(ub), xp.log(vb)
        logc = clayton_logc if fam.base == "clayton" else gumbel_logc
        return lambda th: -xp.mean(logc(_col(th, xp), lu, lv, xp), axis=-1)
    if fam.base == "frank":
        return lambda th: -xp.mean(frank_logc(_col(th, xp), ub, vb, xp), axis=-1)
    raise ValueError(fam.name)


def t_nll_rho(x, y, nu, ops: Ops):
    """f(rho) at fixed nu, with the t scores x, y already computed for that nu."""
    xp = ops.xp
    nuc = _col(nu, xp)
    return lambda rho: -xp.mean(t_logc(_col(rho, xp), nuc, x, y, ops), axis=-1)


def t_nll_lognu(u, v, rho, ops: Ops):
    """f(log nu) at fixed rho; the t scores are recomputed for every nu."""
    xp = ops.xp
    rhoc = _col(rho, xp)

    def f(lognu):
        nu = xp.exp(_col(lognu, xp))
        x, y = ops.tppf(u, nu), ops.tppf(v, nu)
        return -xp.mean(t_logc(rhoc, nu, x, y, ops), axis=-1)

    return f


def t_scorer(u, v, ops: Ops, ranks=None):
    """nu -> (t scores of u, t scores of v), nu shaped like ``_col(nu)``.

    With ``ranks = (ru, rv)`` (0-based integer ranks, no ties) the pseudo-observations are a
    permutation of the grid i / (n + 1), so the quantiles are computed once on the lower half
    of the grid (the t distribution is symmetric) and gathered: n/2 quantiles per trial nu
    instead of 2n. Without ranks (ties present) every value is inverted directly.
    """
    xp = ops.xp
    if ranks is None:
        return lambda nu: (ops.tppf(u, nu), ops.tppf(v, nu))
    ru, rv = ranks
    n = u.shape[-1]
    p_lo = xp.asarray([i / (n + 1.0) for i in range(1, n // 2 + 1)])

    def scores(nu):
        lo = ops.tppf(p_lo, nu)
        mid = xp.zeros(lo.shape[:-1] + (n % 2,))
        full = xp.concatenate([lo, mid, -xp.flip(lo, axis=-1)], axis=-1)
        full = full * xp.ones(ru.shape[:-1] + (1,))
        return xp.take_along_axis(full, ru, axis=-1), xp.take_along_axis(full, rv, axis=-1)

    return scores


def normal_scores_corr(u, v, ops: Ops):
    """Correlation of normal scores (van der Waerden): the starting rho for the t fit."""
    xp = ops.xp
    x, y = ops.ndtri(u), ops.ndtri(v)
    return xp.mean(x * y, axis=-1) / xp.sqrt(xp.mean(x * x, axis=-1) * xp.mean(y * y, axis=-1))


def fit_t(u, v, ops: Ops, minimize, ranks=None):
    """t copula by profile likelihood: an outer Brent over log nu, an inner Brent over rho.

    For each trial nu the t scores are computed once and rho is re-optimised, so the outer
    objective is the profile  min_rho NLL(rho, nu).  This reaches the joint optimum (up to
    the solver tolerances) with one t-quantile evaluation per outer step; coordinate ascent
    (alternating nu | rho and rho | nu) needed several rounds on development data to get as close,
    because rho and nu are correlated in the likelihood.
    """
    xp = ops.xp
    scores = t_scorer(u, v, ops, ranks)

    def rho_given_nu(lognu):
        nu = xp.exp(lognu)
        x, y = scores(_col(nu, xp))
        return minimize(t_nll_rho(x, y, nu, ops), -RHO_MAX, RHO_MAX)

    lognu, _, ok_outer = minimize(lambda ln: rho_given_nu(ln)[1], LOG_NU_MIN, LOG_NU_MAX)
    rho, nll, ok_inner = rho_given_nu(lognu)
    return rho, xp.exp(lognu), nll, ok_outer & ok_inner


def fit_families(u, v, ops: Ops, minimize, families=None, ranks=None):
    """Fit every family. Returns {name: (theta, nu_or_None, mean_nll, converged)}.

    ``converged`` is False where a Brent solve stopped at its iteration cap (for t: the
    outer nu solve or the final inner rho solve).

    ``ranks`` (optional, see ``t_scorer``) only speeds up the t copula.
    """
    out = {}
    for fam in (FAMILIES if families is None else [get(n) for n in families]):
        if fam.base == "t":
            out[fam.name] = fit_t(u, v, ops, minimize, ranks)
        else:
            th, f, ok = minimize(make_nll(fam, u, v, ops), fam.lower, fam.upper)
            out[fam.name] = (th, None, f, ok)
    return out
