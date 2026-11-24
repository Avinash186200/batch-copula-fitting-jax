"""The full offline run behind every number in the README (except the speed table).

    python -m copulabatch.cli bench --out results            # held-out seeds (committed)
    python -m copulabatch.cli bench --out out --split dev    # development seeds

Experiments (each writes results/<name>.json):
  recovery   - fit the true family to simulated pairs; bias and RMSE vs sample size
  agreement  - scipy reference vs vectorised NumPy vs JAX on the same pairs, all families
  selection  - AIC choice among all 11 families; confusion matrix of true vs selected
  tail       - closed-form tail coefficients vs empirical estimators; Gaussian vs t on
               joint-crash probabilities when the truth is a t copula
  demo       - rolling fits on a synthetic multi-asset panel with a stress regime, and a
               stress scenario (pair VaR / ES under different fitted copulas)

All randomness comes from ``np.random.default_rng([split, experiment, ...])``. Development
seeds (split 1) were used while building; the committed results use held-out seeds (split 2).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np

from . import families as F
from .engines import XTOL, fit, jax_available
from .sim import RegimeConfig, pairs, regime_returns

SPLITS = {"dev": 1, "holdout": 2}
RESULT_FILES = ("recovery", "agreement", "selection", "tail", "demo")

TRUE_FAMILIES = ("gaussian", "t", "clayton", "clayton_180", "gumbel", "gumbel_180", "frank",
                 "clayton_90", "gumbel_90")
NEGATIVE = ("clayton_90", "clayton_270", "gumbel_90", "gumbel_270")

# agreement tolerances, asserted in tests/test_engines.py and recorded in agreement.json
TOL_PARAM = 1e-4            # |theta_batch - theta_ref|, one-parameter families
TOL_LOGLIK = 1e-6           # |loglik_batch - loglik_ref| (total over n), one-parameter families
TOL_T_RHO = 1e-4            # t copula: |rho_batch - rho_ref| (the reference is a 2-D L-BFGS-B)
TOL_T_LOGLIK = 1e-6         # t copula: loglik_batch >= loglik_ref - TOL_T_LOGLIK


@dataclass(frozen=True)
class Sizes:
    rec_reps: int = 100
    rec_ns: tuple = (250, 1000, 4000)
    agree_pairs: int = 54           # 6 per true family
    agree_n: int = 500
    sel_reps: int = 100
    sel_ns: tuple = (250, 1000)
    tail_n: int = 20000
    tail_reps: int = 100
    crash_reps: int = 100
    crash_n: int = 1000
    demo_window: int = 250
    demo_step: int = 25
    demo_draws: int = 100_000
    demo_cfg: RegimeConfig = RegimeConfig()


FULL = Sizes()


def _rng(split: str, *key: int) -> np.random.Generator:
    return np.random.default_rng([SPLITS[split], *key])


def _true_params(name: str, tau: float, nu: float = 4.0):
    return F.param_from_tau(name, -tau if name in NEGATIVE else tau, nu)


def _f(x) -> float:
    return float(x)


def _capped(r) -> int:
    """Pairs where any family's solver stopped at its iteration cap (or, for the scipy
    reference, reported no success)."""
    return int(np.sum(~r.converged.all(axis=1)))


# ---------------------------------------------------------------- 1. parameter recovery


def _recovery_engine(name: str, engine: str) -> str:
    """Recovery fits one family per call. For one-parameter families the NumPy engine is
    quicker than paying a JAX compile per (family, n); the t copula needs many t quantiles,
    where JAX is several times faster than scipy's stdtrit."""
    return "jax" if (name == "t" and engine == "jax") else "numpy"


def recovery(split: str, s: Sizes, engine: str) -> dict:
    tau0, nu0 = 0.5, 4.0
    out = {"tau": tau0, "t_nu": nu0, "reps": s.rec_reps, "rows": []}
    for fi, name in enumerate(TRUE_FAMILIES):
        p = _true_params(name, tau0, nu0)
        theta0 = p[0] if isinstance(p, tuple) else p
        for n in s.rec_ns:
            U, V = pairs([(name, p)] * s.rec_reps, n, _rng(split, 1, fi, n))
            r = fit(U, V, engine=_recovery_engine(name, engine), families=(name,))
            th = r.theta[:, 0]
            tau_hat = np.array([F.tau(name, (t, 4.0) if name == "t" else t) for t in th])
            row = {"family": name, "n": n, "pairs": s.rec_reps, "theta_true": _f(theta0),
                   "theta_bias": _f(np.mean(th - theta0)),
                   "theta_rmse": _f(np.sqrt(np.mean((th - theta0) ** 2))),
                   "tau_true": _f(F.tau(name, p)), "tau_bias": _f(np.mean(tau_hat - F.tau(name, p))),
                   "tau_rmse": _f(np.sqrt(np.mean((tau_hat - F.tau(name, p)) ** 2)))}
            row["sqrt_n_theta_rmse"] = _f(np.sqrt(n) * row["theta_rmse"])
            row["pairs_hit_iteration_cap"] = _capped(r)
            if name == "t":
                nu = r.nu[:, 0]
                row.update({"nu_true": nu0, "nu_median": _f(np.median(nu)),
                            "nu_q25": _f(np.quantile(nu, 0.25)), "nu_q75": _f(np.quantile(nu, 0.75)),
                            "nu_at_upper_bound": int(np.sum(nu > F.NU_MAX - 1e-3))})
            out["rows"].append(row)
    out["pairs_hit_iteration_cap"] = sum(x["pairs_hit_iteration_cap"] for x in out["rows"])
    return out


# ---------------------------------------------------------------- 2. agreement with the reference


def _mixed_specs(rng, count: int, nu_range=(3.0, 8.0)):
    specs = []
    for i in range(count):
        name = TRUE_FAMILIES[i % len(TRUE_FAMILIES)]
        tau = rng.uniform(0.2, 0.7)
        specs.append((name, _true_params(name, tau, rng.uniform(*nu_range))))
    return specs


def agreement(split: str, s: Sizes, engines=("numpy", "jax")) -> dict:
    rng = _rng(split, 2)
    specs = _mixed_specs(rng, s.agree_pairs)
    U, V = pairs(specs, s.agree_n, rng)
    ref = fit(U, V, engine="reference")
    out = {"pairs": s.agree_pairs, "n": s.agree_n, "xtol": XTOL,
           "tolerances": {"param": TOL_PARAM, "loglik": TOL_LOGLIK, "t_rho": TOL_T_RHO,
                          "t_loglik": TOL_T_LOGLIK},
           "reference_pairs_not_converged": _capped(ref), "engines": {}}
    one = [j for j, f in enumerate(ref.families) if f != "t"]
    jt = ref.families.index("t")
    for eng in engines:
        r = fit(U, V, engine=eng)
        d_par = np.abs(r.theta[:, one] - ref.theta[:, one])
        d_ll = np.abs(r.loglik[:, one] - ref.loglik[:, one])
        t_gain = r.loglik[:, jt] - ref.loglik[:, jt]
        out["engines"][eng] = {
            "diff_max_param_one_param_families": _f(d_par.max()),
            "diff_max_loglik_one_param_families": _f(d_ll.max()),
            "diff_max_t_rho": _f(np.abs(r.theta[:, jt] - ref.theta[:, jt]).max()),
            "diff_max_t_rel_nu": _f(np.max(np.abs(r.nu[:, jt] - ref.nu[:, jt]) / ref.nu[:, jt])),
            "diff_min_t_loglik_gain_vs_ref": _f(t_gain.min()),
            "diff_max_t_loglik_gain_vs_ref": _f(t_gain.max()),
            "same_selected_family": int(np.sum(r.best == ref.best)),
            "pairs_hit_iteration_cap": _capped(r),
            "within_tolerance": bool(d_par.max() <= TOL_PARAM and d_ll.max() <= TOL_LOGLIK
                                     and np.abs(r.theta[:, jt] - ref.theta[:, jt]).max() <= TOL_T_RHO
                                     and t_gain.min() >= -TOL_T_LOGLIK),
        }
    return out


# ---------------------------------------------------------------- 3. family selection by AIC


def selection(split: str, s: Sizes, engine: str) -> dict:
    out = {"true_families": list(TRUE_FAMILIES), "candidates": list(F.FAMILY_NAMES),
           "reps_per_family": s.sel_reps, "tau_range": [0.2, 0.7], "t_nu_range": [3.0, 8.0],
           "by_n": []}
    for n in s.sel_ns:
        rng = _rng(split, 3, n)
        specs = []
        for name in TRUE_FAMILIES:
            specs += [(name, _true_params(name, rng.uniform(0.2, 0.7), rng.uniform(3.0, 8.0)))
                      for _ in range(s.sel_reps)]
        U, V = pairs(specs, n, rng)
        r = fit(U, V, engine=engine)
        chosen = r.best_name
        conf = {t: {c: 0 for c in F.FAMILY_NAMES} for t in TRUE_FAMILIES}
        tail_ok = 0
        for (t, _), c in zip(specs, chosen):
            conf[t][c] += 1
            tail_ok += F.tail_corners(t) == F.tail_corners(c)
        correct = sum(conf[t][t] for t in TRUE_FAMILIES)
        out["by_n"].append({
            "n": n, "pairs": len(specs), "pairs_hit_iteration_cap": _capped(r),
            "accuracy": correct / len(specs),
            "tail_class_accuracy": tail_ok / len(specs),
            "per_family_accuracy": {t: conf[t][t] / s.sel_reps for t in TRUE_FAMILIES},
            "confusion": conf})
    return out


# ---------------------------------------------------------------- 4. tail dependence


def tail(split: str, s: Sizes, engine: str) -> dict:
    qs = (0.01, 0.05)
    tau0, nu0 = 0.5, 4.0
    rows = []
    for fi, name in enumerate(("gaussian", "t", "clayton", "gumbel", "clayton_180", "frank")):
        p = _true_params(name, tau0, nu0)
        lo, up = F.tail_dependence(name, p)
        row = {"family": name, "tau": tau0, "params": list(p) if isinstance(p, tuple) else p,
               "n": s.tail_n, "samples": s.tail_reps, "lambda_L": lo, "lambda_U": up}
        rng = _rng(split, 4, fi)
        est = np.array([[F.empirical_tail(*F.sample(name, p, s.tail_n, rng), q) for q in qs]
                        for _ in range(s.tail_reps)])           # (reps, q, 2)
        for k, q in enumerate(qs):
            row[f"emp_L_q{q}"] = _f(est[:, k, 0].mean())
            row[f"emp_L_q{q}_sd"] = _f(est[:, k, 0].std(ddof=1))
            row[f"emp_U_q{q}"] = _f(est[:, k, 1].mean())
            row[f"exact_L_q{q}"] = _f(F.cdf(name, p, q, q) / q)        # what the estimator targets at this q
        rows.append(row)

    # joint crashes: truth is a t copula; fit Gaussian and t to each sample
    rho0 = float(np.sin(np.pi * tau0 / 2.0))
    truth = ("t", (rho0, nu0))
    rng = _rng(split, 4, 99)
    U, V = pairs([truth] * s.crash_reps, s.crash_n, rng)
    r = fit(U, V, engine=engine, families=("gaussian", "t"))
    crash = {"truth": {"family": "t", "rho": rho0, "nu": nu0}, "pairs": s.crash_reps,
             "pairs_hit_iteration_cap": _capped(r),
             "n": s.crash_n, "rows": []}
    t_better = int(np.sum(r.aic[:, 1] < r.aic[:, 0]))
    for q in (0.01, 0.05):
        true_p = F.joint_crash_prob(*truth, q)
        g = np.array([F.joint_crash_prob("gaussian", r.params(i, "gaussian"), q) for i in range(s.crash_reps)])
        t = np.array([F.joint_crash_prob("t", r.params(i, "t"), q) for i in range(s.crash_reps)])
        emp = np.mean((U <= q) & (V <= q), axis=1)
        crash["rows"].append({"q": q, "true_prob": true_p, "independence_prob": q * q,
                              "gaussian_fit_mean": _f(g.mean()), "t_fit_mean": _f(t.mean()),
                              "empirical_mean": _f(emp.mean()),
                              "gaussian_over_true": _f(g.mean() / true_p), "t_over_true": _f(t.mean() / true_p)})
    lam_t = [F.tail_dependence("t", r.params(i, "t"))[0] for i in range(s.crash_reps)]
    crash.update({"t_preferred_by_aic": t_better, "true_lambda": F.tail_dependence(*truth)[0],
                  "t_fit_lambda_mean": _f(np.mean(lam_t)), "gaussian_fit_lambda": 0.0,
                  "gaussian_rho_mean": _f(r.theta[:, 0].mean()), "t_rho_mean": _f(r.theta[:, 1].mean()),
                  "t_nu_median": _f(np.median(r.nu[:, 1]))})
    return {"closed_form_vs_empirical": rows, "joint_crash": crash}


# ---------------------------------------------------------------- 5. demo: rolling fits and a stress scenario


def _phase(start: int, end: int, cfg: RegimeConfig) -> str:
    """Window [start, end) relative to the stress block."""
    if end <= cfg.stress_start:
        return "calm_before"
    if start >= cfg.stress_end:
        return "calm_after"
    if start >= cfg.stress_start and end <= cfg.stress_end:
        return "stress"
    return "mixed"


def _pair_portfolio_tail(x: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> tuple[float, float]:
    """VaR and ES (positive numbers = losses) of a 50/50 pair portfolio."""
    pnl = 0.5 * (x + y)
    var = -np.quantile(pnl, alpha)
    return float(var), float(-pnl[pnl <= -var].mean())


def window_pseudo_obs(R: np.ndarray, ends, W: int, prs):
    """Point in time: the window ending at day e (exclusive) uses days [e - W, e) only, and
    ranks are computed inside the window. Rows are ordered window-major, then pair."""
    U = np.stack([F.pseudo_obs(R[e - W:e, i]) for e in ends for i, _ in prs])
    V = np.stack([F.pseudo_obs(R[e - W:e, j]) for e in ends for _, j in prs])
    return U, V


def demo(split: str, s: Sizes, engine: str) -> dict:
    cfg = s.demo_cfg
    R = regime_returns(cfg, _rng(split, 5))
    W, step = s.demo_window, s.demo_step
    prs = list(combinations(range(cfg.n_assets), 2))
    ends = list(range(W, cfg.days + 1, step))
    U, V = window_pseudo_obs(R, ends, W, prs)
    r = fit(U, V, engine=engine)
    # q * W must be an integer: with pseudo-observations rank / (W + 1), "u <= q" then
    # selects exactly q * W days per asset (q = 0.05 with W = 250 would select 12, a 4.8% tail)
    q = round(0.04 * W) / W
    windows = []
    k = 0
    for e in ends:
        sel, lamL, crash_sel, crash_g, emp = [], [], [], [], []
        for _ in prs:
            name = r.families[r.best[k]]
            sel.append(name)
            p = r.params(k, name)
            lamL.append(F.tail_dependence(name, p)[0])
            crash_sel.append(F.joint_crash_prob(name, p, q))
            crash_g.append(F.joint_crash_prob("gaussian", r.params(k, "gaussian"), q))
            emp.append(float(np.mean((U[k] <= q) & (V[k] <= q))))
            k += 1
        windows.append({"end": e, "phase": _phase(e - W, e, cfg),
                        "share_lower_tail_family": _f(np.mean(["LL" in F.tail_corners(n) for n in sel])),
                        "mean_lambda_L_selected": _f(np.mean(lamL)),
                        "crash_prob_selected": _f(np.mean(crash_sel)),
                        "crash_prob_gaussian": _f(np.mean(crash_g)),
                        "crash_prob_empirical": _f(np.mean(emp)),
                        "families": {n: sel.count(n) for n in sorted(set(sel))}})
    true_crash = {"calm": F.joint_crash_prob("gaussian", cfg.calm_rho, q),
                  "stress": F.joint_crash_prob("clayton", cfg.stress_theta, q)}
    phases = {}
    for ph in ("calm_before", "mixed", "stress", "calm_after"):
        ws = [w for w in windows if w["phase"] == ph]
        if not ws:
            continue
        phases[ph] = {"windows": len(ws), "pairs_per_window": len(prs),
                      **{key: _f(np.mean([w[key] for w in ws])) for key in
                         ("share_lower_tail_family", "mean_lambda_L_selected", "crash_prob_selected",
                          "crash_prob_gaussian", "crash_prob_empirical")}}

    # stress scenario: the last window fully inside the stress block vs the last calm window
    i_stress = max(i for i, w in enumerate(windows) if w["phase"] == "stress")
    i_calm = max(i for i, w in enumerate(windows) if w["phase"] == "calm_before")
    e_s = windows[i_stress]["end"]
    rows = {"true_stress_copula": [], "selected_on_stress_window": [],
            "gaussian_on_stress_window": [], "gaussian_on_last_calm_window": []}
    for pi, (a, b) in enumerate(prs):
        xa, xb = R[e_s - W:e_s, a], R[e_s - W:e_s, b]   # empirical marginals of the stress window
        ks, kc = i_stress * len(prs) + pi, i_calm * len(prs) + pi
        best = r.families[r.best[ks]]
        models = {"true_stress_copula": ("clayton", cfg.stress_theta),
                  "selected_on_stress_window": (best, r.params(ks, best)),
                  "gaussian_on_stress_window": ("gaussian", r.params(ks, "gaussian")),
                  "gaussian_on_last_calm_window": ("gaussian", r.params(kc, "gaussian"))}
        for mi, (key, (name, p)) in enumerate(models.items()):
            # one generator per (pair, model): a change in one fit cannot shift the draws of another
            u, v = F.sample(name, p, s.demo_draws, _rng(split, 5, 1, pi, mi))
            rows[key].append(_pair_portfolio_tail(np.quantile(xa, u), np.quantile(xb, v)))
    scenario = {"window_end": e_s, "calm_window_end": windows[i_calm]["end"], "pairs": len(prs),
                "draws_per_pair": s.demo_draws, "alpha": 0.01, "rows": {}}
    for key, vals in rows.items():
        arr = np.array(vals)
        scenario["rows"][key] = {"var99_mean": _f(arr[:, 0].mean()), "es99_mean": _f(arr[:, 1].mean())}
    return {"config": asdict(cfg), "window": W, "step": step, "windows_total": len(windows),
            "pairs_per_window": len(prs), "fits": len(ends) * len(prs),
            "pairs_hit_iteration_cap": _capped(r), "q": q,
            "true_crash_prob": true_crash, "phases": phases, "windows": windows,
            "stress_scenario": scenario}


# ---------------------------------------------------------------- driver


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")


def run(out: Path, split: str = "holdout", sizes: Sizes = FULL, engine: str | None = None,
        log=print) -> dict:
    from .report import render

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    engine = engine or ("jax" if jax_available() else "numpy")
    agree_engines = ("numpy", "jax") if jax_available() else ("numpy",)
    steps = {"recovery": lambda: recovery(split, sizes, engine),
             "agreement": lambda: agreement(split, sizes, agree_engines),
             "selection": lambda: selection(split, sizes, engine),
             "tail": lambda: tail(split, sizes, engine),
             "demo": lambda: demo(split, sizes, engine)}
    res = {}
    for name, f in steps.items():
        t0 = time.perf_counter()
        eng = {"recovery": "numpy, and jax for t" if engine == "jax" else "numpy",
               "agreement": "reference vs " + "/".join(agree_engines)}.get(name, engine)
        res[name] = {"split": split, "engine": eng, **f()}
        write_json(out / f"{name}.json", res[name])
        log(f"{name}: {time.perf_counter() - t0:.1f}s")
    (out / "RESULTS.md").write_text(render(out), encoding="utf-8", newline="\n")
    return res


def small_sizes() -> Sizes:
    """A tiny configuration for tests."""
    return replace(FULL, rec_reps=6, rec_ns=(200,), agree_pairs=9, agree_n=150, sel_reps=4,
                   sel_ns=(200,), tail_n=2000, tail_reps=2, crash_reps=4, crash_n=300,
                   demo_window=120, demo_step=60, demo_draws=2000,
                   demo_cfg=RegimeConfig(n_assets=4, days=600, stress_start=240, stress_end=480))
