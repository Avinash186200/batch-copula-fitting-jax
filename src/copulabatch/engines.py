"""Three engines that fit every family to N pairs and pick one by AIC.

* ``reference``: a plain Python loop over pairs with scipy: ``minimize_scalar(bounded)``
  per one-parameter family and a joint 2-D L-BFGS-B over (rho, log nu) for the t copula.
  This is the "one by one" baseline and the correctness yardstick.
* ``numpy``: all N pairs at once, as (N, n) arrays, with the branch-free Brent in ``brent``.
* ``jax``: the same recipe for one pair, ``jax.vmap``-ed over pairs and ``jax.jit``-compiled;
  would run on a GPU when jaxlib has one (untested; only CPU was measured). Pairs are
  fed in chunks to bound memory.

All engines take pseudo-observations ``U, V`` of shape (N, n) and return a ``FitResult``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy import optimize

from . import brent
from .families import FAMILY_NAMES, NP_OPS, get
from .objective import (
    LOG_NU_MAX,
    LOG_NU_MIN,
    RHO_MAX,
    fit_families,
    make_nll,
    normal_scores_corr,
    t_nll_lognu,
)

MAXITER = 500        # Brent iteration cap per solve (both batch engines)
XTOL = 1e-7          # absolute tolerance on each 1-D parameter (rho, theta or log nu)
ENGINES = ("reference", "numpy", "jax")


@dataclass
class FitResult:
    families: tuple[str, ...]
    theta: np.ndarray      # (N, F) rho or theta
    nu: np.ndarray         # (N, F) degrees of freedom for t, NaN elsewhere
    loglik: np.ndarray     # (N, F) total log-likelihood
    n: int
    converged: np.ndarray  # (N, F) False where a solver stopped at its iteration cap

    @property
    def k(self) -> np.ndarray:
        return np.array([get(f).k for f in self.families])

    @property
    def aic(self) -> np.ndarray:
        return 2.0 * self.k[None, :] - 2.0 * self.loglik

    @property
    def best(self) -> np.ndarray:
        return np.argmin(self.aic, axis=1)

    @property
    def best_name(self) -> list[str]:
        return [self.families[i] for i in self.best]

    def params(self, i: int, family: str):
        """Parameters of ``family`` for pair i, in the form ``families`` functions expect."""
        j = self.families.index(family)
        return (float(self.theta[i, j]), float(self.nu[i, j])) if get(family).base == "t" \
            else float(self.theta[i, j])


def _assemble(fits: dict, families, n: int) -> FitResult:
    theta = np.stack([np.asarray(fits[f][0], dtype=float) for f in families], axis=-1)
    nu = np.stack([np.asarray(fits[f][1], dtype=float) if fits[f][1] is not None
                   else np.full(theta.shape[0], np.nan) for f in families], axis=-1)
    loglik = np.stack([-n * np.asarray(fits[f][2], dtype=float) for f in families], axis=-1)
    conv = np.stack([np.asarray(fits[f][3], dtype=bool) for f in families], axis=-1)
    return FitResult(tuple(families), theta, nu, loglik, n, conv)


def _check(U, V):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    if U.ndim == 1:
        U, V = U[None, :], V[None, :]
    if U.shape != V.shape or U.ndim != 2:
        raise ValueError("U and V must both have shape (N, n)")
    if not ((U > 0) & (U < 1) & (V > 0) & (V < 1)).all():
        raise ValueError("pseudo-observations must lie strictly inside (0, 1)")
    return U, V


def grid_ranks(U, V):
    """0-based integer ranks if every row of U and V is a permutation of i / (n + 1), else None."""
    n = U.shape[1]
    ru = np.rint(U * (n + 1.0)).astype(np.int64) - 1
    rv = np.rint(V * (n + 1.0)).astype(np.int64) - 1
    target = np.arange(n)
    for r, X in ((ru, U), (rv, V)):
        if not (np.array_equal(np.sort(r, axis=1), np.broadcast_to(target, r.shape))
                and np.allclose(X, (r + 1.0) / (n + 1.0), rtol=0.0, atol=1e-12)):
            return None
    return ru, rv


# ---------------------------------------------------------------- reference: scipy, one pair at a time


def _ref_pair(u, v, families, xtol):
    out = {}
    for name in families:
        fam = get(name)
        if fam.base == "t":
            rho0 = float(np.clip(normal_scores_corr(u, v, NP_OPS), -0.98, 0.98))

            def nll(p):
                return float(t_nll_lognu(u, v, p[0], NP_OPS)(p[1]))

            r = optimize.minimize(nll, x0=[rho0, np.log(8.0)], method="L-BFGS-B",
                                  bounds=[(-RHO_MAX, RHO_MAX), (LOG_NU_MIN, LOG_NU_MAX)],
                                  options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500})
            out[name] = (r.x[0], np.exp(r.x[1]), r.fun, bool(r.success))
        else:
            f = make_nll(fam, u, v, NP_OPS)
            r = optimize.minimize_scalar(lambda th: float(f(th)), bounds=(fam.lower, fam.upper),
                                         method="bounded", options={"xatol": xtol, "maxiter": 500})
            out[name] = (r.x, None, r.fun, bool(r.success))
    return out


def fit_reference(U, V, families=FAMILY_NAMES, xtol=XTOL) -> FitResult:
    U, V = _check(U, V)
    per_pair = [_ref_pair(U[i], V[i], families, xtol) for i in range(U.shape[0])]
    fits = {f: tuple(None if per_pair[0][f][k] is None else np.array([p[f][k] for p in per_pair])
                     for k in range(4)) for f in families}
    return _assemble(fits, families, U.shape[1])


# ---------------------------------------------------------------- vectorised NumPy


def fit_numpy(U, V, families=FAMILY_NAMES, xtol=XTOL) -> FitResult:
    U, V = _check(U, V)
    N = U.shape[0]

    def minimize(f, lo, hi):
        x, fx, _, ok = brent.minimize_numpy(f, np.full(N, lo), np.full(N, hi), xtol=xtol,
                                            maxiter=MAXITER)
        return x, fx, ok

    with np.errstate(all="ignore"):
        fits = fit_families(U, V, NP_OPS, minimize, families, grid_ranks(U, V))
    return _assemble(fits, families, U.shape[1])


# ---------------------------------------------------------------- JAX: jit(vmap(one pair))


def jax_available() -> bool:
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


@cache
def _jax_fitter(families: tuple[str, ...], xtol: float, use_ranks: bool):
    import jax
    import jax.numpy as jnp

    from .jax_ops import JAX_OPS

    def minimize(f, lo, hi):
        s = brent.init(f, jnp.asarray(lo, jnp.float64), jnp.asarray(hi, jnp.float64), jnp)
        s = jax.lax.while_loop(lambda s: (~s.done) & (s.nfev <= MAXITER),
                               lambda s: brent.step(s, f, xtol, jnp), s)
        return s.x, s.fx, brent.converged(s, xtol, jnp)

    def one_pair(u, v, ru, rv):
        fits = fit_families(u, v, JAX_OPS, minimize, families, (ru, rv) if use_ranks else None)
        nan = jnp.asarray(jnp.nan, jnp.float64)
        return tuple((th, nan if nu is None else nu, f, ok) for th, nu, f, ok in
                     (fits[name] for name in families))

    batched = jax.vmap(one_pair)
    return jax.jit(batched), jax.pmap(batched)


def fit_jax(U, V, families=FAMILY_NAMES, xtol=XTOL, chunk=4096) -> FitResult:
    """jit(vmap) over pairs, ``chunk`` pairs per device per call.

    With several local devices (several GPUs, or CPU "devices" made with
    ``XLA_FLAGS=--xla_force_host_platform_device_count=k`` before JAX is imported) the
    chunks are spread over them with ``pmap``; XLA's CPU backend otherwise runs these
    element-wise kernels on one core.
    """
    import jax
    import jax.numpy as jnp

    from .jax_ops import JAX_OPS  # noqa: F401  (enables float64)

    U, V = _check(U, V)
    ranks = grid_ranks(U, V)
    jit_fn, pmap_fn = _jax_fitter(tuple(families), float(xtol), ranks is not None)
    N, n = U.shape
    RU, RV = ranks if ranks is not None else (np.zeros(U.shape, np.int64),) * 2
    D = jax.local_device_count()
    need = -(-N // D)
    per = min(chunk, 1 << max(3, (need - 1).bit_length()))   # power of two: few compiled shapes
    block = D * per
    parts = []
    for s in range(0, N, block):
        e = min(N, s + block)
        arrs = [X[s:e] for X in (U, V, RU, RV)]
        if e - s < block:                     # pad the last block: one compiled shape
            pad = block - (e - s)
            arrs = [np.concatenate([X, np.repeat(X[-1:], pad, 0)]) for X in arrs]
        if D == 1:
            res = jit_fn(*(jnp.asarray(X) for X in arrs))
        else:
            res = pmap_fn(*(jnp.asarray(X.reshape((D, per) + X.shape[1:])) for X in arrs))
        res = jax.block_until_ready(res)
        parts.append([[np.asarray(a).reshape(-1)[: e - s] for a in fam] for fam in res])
    fits = {}
    for j, name in enumerate(families):
        th, nu, f, ok = (np.concatenate([p[j][k] for p in parts]) for k in range(4))
        fits[name] = (th, nu if get(name).base == "t" else None, f, ok)
    return _assemble(fits, families, n)


def fit(U, V, engine: str = "jax", families=FAMILY_NAMES, xtol=XTOL, **kw) -> FitResult:
    """Fit all families to every pair. ``engine`` in ('reference', 'numpy', 'jax')."""
    if engine == "reference":
        return fit_reference(U, V, families, xtol)
    if engine == "numpy":
        return fit_numpy(U, V, families, xtol)
    if engine == "jax":
        return fit_jax(U, V, families, xtol, **kw)
    raise ValueError(f"unknown engine {engine!r}; pick one of {ENGINES}")
