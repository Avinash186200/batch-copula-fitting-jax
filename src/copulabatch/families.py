"""Bivariate copula families: densities, Kendall's tau, tail dependence, CDFs and samplers.

The log-density formulas are written once against a small ``Ops`` namespace, so the
scipy reference fitter, the vectorised NumPy fitter and the JAX fitter all evaluate
the same expressions (``ops.xp`` is ``numpy`` or ``jax.numpy``).

Rotations follow one convention throughout: with (U', V') drawn from the base copula,
  rotation 180 (survival):  (U, V) = (1 - U', 1 - V')   lower and upper tails swap
  rotation  90:             (U, V) = (1 - U', V')       negative dependence
  rotation 270:             (U, V) = (U', 1 - V')       negative dependence
so the density of a rotated copula is the base density at the flipped coordinates.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, special, stats

# ---------------------------------------------------------------- family registry


@dataclass(frozen=True)
class Family:
    name: str
    base: str            # gaussian | t | clayton | gumbel | frank
    rotation: int        # 0, 90, 180, 270
    lower: float         # bounds of the main parameter (rho or theta)
    upper: float

    @property
    def k(self) -> int:
        """Number of free parameters (for AIC)."""
        return 2 if self.base == "t" else 1


RHO_MAX = 0.995
NU_MIN, NU_MAX = 2.0, 50.0

FAMILIES: tuple[Family, ...] = (
    Family("gaussian", "gaussian", 0, -RHO_MAX, RHO_MAX),
    Family("t", "t", 0, -RHO_MAX, RHO_MAX),
    Family("clayton", "clayton", 0, 1e-4, 30.0),
    Family("clayton_180", "clayton", 180, 1e-4, 30.0),
    Family("clayton_90", "clayton", 90, 1e-4, 30.0),
    Family("clayton_270", "clayton", 270, 1e-4, 30.0),
    Family("gumbel", "gumbel", 0, 1.0, 20.0),
    Family("gumbel_180", "gumbel", 180, 1.0, 20.0),
    Family("gumbel_90", "gumbel", 90, 1.0, 20.0),
    Family("gumbel_270", "gumbel", 270, 1.0, 20.0),
    Family("frank", "frank", 0, -40.0, 40.0),
)
FAMILY_NAMES = tuple(f.name for f in FAMILIES)
BY_NAME = {f.name: f for f in FAMILIES}


def get(name: str) -> Family:
    return BY_NAME[name]


def rotate_uv(u, v, rotation: int):
    """Map observed (u, v) to base-copula coordinates (works for numpy and jax arrays)."""
    if rotation == 0:
        return u, v
    if rotation == 180:
        return 1.0 - u, 1.0 - v
    if rotation == 90:
        return 1.0 - u, v
    if rotation == 270:
        return u, 1.0 - v
    raise ValueError(rotation)


# ---------------------------------------------------------------- shared log-densities


@dataclass(frozen=True)
class Ops:
    xp: object
    ndtri: Callable
    gammaln: Callable
    tppf: Callable        # tppf(p, nu): Student-t quantile


NP_OPS = Ops(np, special.ndtri, special.gammaln, lambda p, nu: special.stdtrit(nu, p))


def gauss_logc(rho, x, y, xp):
    """Gaussian copula log-density at normal scores x = ndtri(u), y = ndtri(v)."""
    r2 = rho * rho
    om = 1.0 - r2
    return -0.5 * xp.log(om) - (r2 * (x * x + y * y) - 2.0 * rho * x * y) / (2.0 * om)


def t_logc(rho, nu, x, y, ops: Ops):
    """Student-t copula log-density at t scores x = t_nu^-1(u), y = t_nu^-1(v)."""
    xp, gl = ops.xp, ops.gammaln
    om = 1.0 - rho * rho
    const = gl((nu + 2.0) / 2.0) + gl(nu / 2.0) - 2.0 * gl((nu + 1.0) / 2.0) - 0.5 * xp.log(om)
    quad = (x * x + y * y - 2.0 * rho * x * y) / (nu * om)
    return (const - (nu + 2.0) / 2.0 * xp.log1p(quad)
            + (nu + 1.0) / 2.0 * (xp.log1p(x * x / nu) + xp.log1p(y * y / nu)))


def clayton_logc(th, lu, lv, xp):
    """Clayton log-density, theta > 0, at lu = log u, lv = log v."""
    a, b = -th * lu, -th * lv
    log_s = xp.log(xp.expm1(a) + xp.exp(b))          # log(u^-th + v^-th - 1), accurate as th -> 0
    return xp.log1p(th) - (1.0 + th) * (lu + lv) - (2.0 + 1.0 / th) * log_s


def gumbel_logc(th, lu, lv, xp):
    """Gumbel log-density, theta >= 1, at lu = log u, lv = log v."""
    lx, ly = xp.log(-lu), xp.log(-lv)
    log_s = xp.logaddexp(th * lx, th * ly)            # log((-log u)^th + (-log v)^th)
    a = xp.exp(log_s / th)
    return (-a + (th - 1.0) * (lx + ly) + (1.0 / th - 2.0) * log_s
            + xp.log(a + th - 1.0) - lu - lv)


def frank_logc(th, u, v, xp):
    """Frank log-density, theta != 0 (|theta| is floored at 1e-10)."""
    th = xp.where(xp.abs(th) < 1e-10, 1e-10, th)
    em = -xp.expm1(-th)                               # 1 - e^-th, same sign as th
    d = em - xp.expm1(-th * u) * xp.expm1(-th * v)
    return xp.log(th * em) - th * (u + v) - 2.0 * xp.log(xp.abs(d))


def logpdf(name: str, params, u, v, ops: Ops = NP_OPS):
    """Pointwise log-density of family ``name`` (params = theta, or (rho, nu) for t)."""
    fam = get(name)
    xp = ops.xp
    ub, vb = rotate_uv(u, v, fam.rotation)
    if fam.base == "gaussian":
        return gauss_logc(params, ops.ndtri(ub), ops.ndtri(vb), xp)
    if fam.base == "t":
        rho, nu = params
        return t_logc(rho, nu, ops.tppf(ub, nu), ops.tppf(vb, nu), ops)
    if fam.base == "clayton":
        return clayton_logc(params, xp.log(ub), xp.log(vb), xp)
    if fam.base == "gumbel":
        return gumbel_logc(params, xp.log(ub), xp.log(vb), xp)
    if fam.base == "frank":
        return frank_logc(params, ub, vb, xp)
    raise ValueError(name)


# ---------------------------------------------------------------- Kendall's tau


def _debye1(th: float) -> float:
    if abs(th) < 1e-8:
        return 1.0
    val, _ = integrate.quad(lambda t: t / np.expm1(t) if t != 0 else 1.0, 0.0, th)
    return val / th


def tau(name: str, params) -> float:
    """Kendall's tau implied by the parameters."""
    fam = get(name)
    if fam.base in ("gaussian", "t"):
        rho = params[0] if fam.base == "t" else params
        t = 2.0 / np.pi * np.arcsin(rho)
    elif fam.base == "clayton":
        t = params / (params + 2.0)
    elif fam.base == "gumbel":
        t = 1.0 - 1.0 / params
    else:  # frank
        th = float(params)
        t = 0.0 if abs(th) < 1e-8 else 1.0 - 4.0 / th * (1.0 - _debye1(th))
    return -t if fam.rotation in (90, 270) else t


def param_from_tau(name: str, t: float, nu: float = 4.0):
    """Inverse of ``tau`` (for setting simulation parameters). For t, nu is passed through."""
    fam = get(name)
    if fam.rotation in (90, 270):
        t = -t
    if fam.base == "gaussian":
        return float(np.sin(np.pi * t / 2.0))
    if fam.base == "t":
        return (float(np.sin(np.pi * t / 2.0)), float(nu))
    if fam.base == "clayton":
        return 2.0 * t / (1.0 - t)
    if fam.base == "gumbel":
        return 1.0 / (1.0 - t)
    if fam.base == "frank":
        if abs(t) < 1e-10:
            return 0.0
        return float(optimize.brentq(lambda th: tau("frank", th) - t,
                                     1e-6 if t > 0 else -100.0, 100.0 if t > 0 else -1e-6))
    raise ValueError(name)


# ---------------------------------------------------------------- tail dependence


def tail_dependence(name: str, params) -> tuple[float, float]:
    """Closed-form (lambda_L, lambda_U) in the main-diagonal corners."""
    fam = get(name)
    if fam.rotation in (90, 270):
        return 0.0, 0.0           # dependence sits in the off-diagonal corners
    if fam.base == "t":
        rho, nu = params
        lam = 2.0 * stats.t.cdf(-np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)), nu + 1.0)
        lo = up = float(lam)
    elif fam.base == "clayton":
        lo, up = float(2.0 ** (-1.0 / params)), 0.0
    elif fam.base == "gumbel":
        lo, up = 0.0, float(2.0 - 2.0 ** (1.0 / params))
    else:  # gaussian (|rho| < 1) and frank
        lo = up = 0.0
    return (up, lo) if fam.rotation == 180 else (lo, up)


def tail_corners(name: str) -> frozenset:
    """Corners with tail dependence: 'LL' (joint crash), 'UU' (joint boom), 'HL' (u high, v low), 'LH'."""
    fam = get(name)
    if fam.base == "t":
        return frozenset({"LL", "UU"})
    if fam.base in ("gaussian", "frank"):
        return frozenset()
    corner = "LL" if fam.base == "clayton" else "UU"
    flip = {0: lambda c: c, 180: lambda c: {"LL": "UU", "UU": "LL"}[c],
            90: lambda c: {"LL": "HL", "UU": "LH"}[c], 270: lambda c: {"LL": "LH", "UU": "HL"}[c]}
    return frozenset({flip[fam.rotation](corner)})


def empirical_tail(u: np.ndarray, v: np.ndarray, q: float) -> tuple[float, float]:
    """Non-parametric (lambda_L(q), lambda_U(q)): joint exceedance share over q."""
    n = len(u)
    lo = np.sum((u <= q) & (v <= q)) / (n * q)
    up = np.sum((u > 1.0 - q) & (v > 1.0 - q)) / (n * q)
    return float(lo), float(up)


# ---------------------------------------------------------------- CDF (for joint-crash probabilities)


def _base_cdf(base: str, params, u: float, v: float) -> float:
    if u <= 0.0 or v <= 0.0:
        return 0.0
    if u >= 1.0:
        return float(v)
    if v >= 1.0:
        return float(u)
    if base == "clayton":
        return float((u ** -params + v ** -params - 1.0) ** (-1.0 / params))
    if base == "gumbel":
        return float(np.exp(-((-np.log(u)) ** params + (-np.log(v)) ** params) ** (1.0 / params)))
    if base == "frank":
        th = params
        if abs(th) < 1e-10:
            return u * v
        return float(-np.log1p(np.expm1(-th * u) * np.expm1(-th * v) / np.expm1(-th)) / th)
    if base == "gaussian":
        rho = params
        a, b = special.ndtri(u), special.ndtri(v)
        if u == v and u != 0.5:
            # diagonal: Phi2(h, h; rho) = Phi(h) - 2 T(h, sqrt((1 - rho) / (1 + rho))) (Owen's T)
            return float(u - 2.0 * special.owens_t(a, np.sqrt((1.0 - rho) / (1.0 + rho))))
        s = np.sqrt(1.0 - rho * rho)
        f = lambda x: stats.norm.pdf(x) * special.ndtr((b - rho * x) / s)
        val, _ = integrate.quad(f, -np.inf, a, epsabs=1e-13, epsrel=1e-10, limit=200)
        return float(val)
    if base == "t":
        rho, nu = params
        a, b = special.stdtrit(nu, u), special.stdtrit(nu, v)

        def f(x):
            scale = np.sqrt((nu + x * x) * (1.0 - rho * rho) / (nu + 1.0))
            return stats.t.pdf(x, nu) * special.stdtr(nu + 1.0, (b - rho * x) / scale)

        val, _ = integrate.quad(f, -np.inf, a, epsabs=1e-13, epsrel=1e-10, limit=200)
        return float(val)
    raise ValueError(base)


def cdf(name: str, params, u: float, v: float) -> float:
    """C(u, v) for any family, rotations included.

    Precision note: the 180-degree rotation evaluates u + v - 1 + C(1 - u, 1 - v), which
    loses absolute precision near 1e-16 for small u, v (a relative error of about 1e-12 at
    the joint-crash probabilities used here, q >= 0.01). Fine for these experiments, not
    for probabilities far out in the corner.
    """
    fam = get(name)
    c = lambda a, b: _base_cdf(fam.base, params, a, b)
    if fam.rotation == 0:
        return c(u, v)
    if fam.rotation == 180:
        return u + v - 1.0 + c(1.0 - u, 1.0 - v)
    if fam.rotation == 90:
        return v - c(1.0 - u, v)
    return u - c(u, 1.0 - v)


def joint_crash_prob(name: str, params, q: float) -> float:
    """P(U <= q, V <= q): both assets below their own q-quantile on the same day."""
    return cdf(name, params, q, q)


# ---------------------------------------------------------------- samplers


def _stable_positive(alpha: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Positive alpha-stable draws with Laplace transform exp(-s^alpha) (Chambers-Mallows-Stuck)."""
    if alpha >= 1.0:
        return np.ones(size)
    w = rng.uniform(0.0, np.pi, size)
    e = rng.exponential(1.0, size)
    return (np.sin(alpha * w) / np.sin(w) ** (1.0 / alpha)
            * (np.sin((1.0 - alpha) * w) / e) ** ((1.0 - alpha) / alpha))


def sample_base(base: str, params, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if base == "gaussian":
        rho = params
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * rng.standard_normal(n)
        return special.ndtr(z1), special.ndtr(z2)
    if base == "t":
        rho, nu = params
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * rng.standard_normal(n)
        w = np.sqrt(nu / rng.chisquare(nu, n))
        return special.stdtr(nu, z1 * w), special.stdtr(nu, z2 * w)
    if base == "clayton":
        th = params
        g = rng.gamma(1.0 / th, 1.0, n)
        e1, e2 = rng.exponential(1.0, n), rng.exponential(1.0, n)
        return (1.0 + e1 / g) ** (-1.0 / th), (1.0 + e2 / g) ** (-1.0 / th)
    if base == "gumbel":
        th = params
        s = _stable_positive(1.0 / th, n, rng)
        e1, e2 = rng.exponential(1.0, n), rng.exponential(1.0, n)
        return np.exp(-((e1 / s) ** (1.0 / th))), np.exp(-((e2 / s) ** (1.0 / th)))
    if base == "frank":
        th = params
        u = rng.uniform(size=n)
        w = rng.uniform(size=n)
        if abs(th) < 1e-10:
            return u, w
        # conditional inverse of dC/du
        v = -np.log1p(w * np.expm1(-th) / (w + (1.0 - w) * np.exp(-th * u))) / th
        return u, v
    raise ValueError(base)


def sample(name: str, params, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n draws (U, V) from family ``name``."""
    fam = get(name)
    u, v = sample_base(fam.base, params, n, rng)
    return rotate_uv(u, v, fam.rotation)


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    """Ranks / (n + 1) along the last axis (average ranks for ties): the empirical PIT."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    return stats.rankdata(x, axis=-1) / (n + 1.0)
