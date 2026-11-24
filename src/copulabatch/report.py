"""Render results/*.json as RESULTS.md. Every number in the README is copied from here."""
from __future__ import annotations

import json
from pathlib import Path

SHORT = {"gaussian": "Gauss", "t": "t", "clayton": "Clay", "clayton_180": "Clay180",
         "clayton_90": "Clay90", "clayton_270": "Clay270", "gumbel": "Gumb",
         "gumbel_180": "Gumb180", "gumbel_90": "Gumb90", "gumbel_270": "Gumb270", "frank": "Frank"}


def _load(out: Path, name: str):
    p = out / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _table(head, rows) -> list[str]:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines


def _recovery(r) -> list[str]:
    L = ["## 1. Parameter recovery", "",
         f"True Kendall's tau = {r['tau']} (negative for the 90-degree rotations), t copula nu = "
         f"{r['t_nu']}. Each row: {r['reps']} independent simulated pairs of length n, fitted with "
         f"the true family (engine: {r['engine']}). theta is rho for Gaussian and t.", ""]
    rows = []
    for x in r["rows"]:
        extra = (f"median {x['nu_median']:.2f} (IQR {x['nu_q25']:.2f}-{x['nu_q75']:.2f}), "
                 f"{x['nu_at_upper_bound']} at bound" if "nu_median" in x else "")
        rows.append([x["family"], x["n"], x["pairs"], f"{x['theta_true']:.4f}",
                     f"{x['theta_bias']:+.4f}", f"{x['theta_rmse']:.4f}",
                     f"{x['sqrt_n_theta_rmse']:.2f}", f"{x['tau_bias']:+.4f}",
                     f"{x['tau_rmse']:.4f}", extra])
    L += _table(["family", "n", "pairs", "theta true", "theta bias", "theta RMSE",
                 "sqrt(n) x RMSE", "tau bias", "tau RMSE", "nu (t only)"], rows)
    total = sum(x["pairs"] for x in r["rows"])
    L += ["", f"Pairs whose solver stopped at the iteration cap: {r['pairs_hit_iteration_cap']} of {total}."]
    return L + [""]


def _agreement(a) -> list[str]:
    tol = a["tolerances"]
    L = ["## 2. Batch engines vs the scipy reference", "",
         f"{a['pairs']} pairs (mixed true families), n = {a['n']}, all 11 families fitted by "
         f"each engine. Solver tolerance xtol = {a['xtol']:g}. Tolerances asserted in tests: "
         f"|d theta| <= {tol['param']:g} and |d loglik| <= {tol['loglik']:g} for one-parameter "
         f"families; for t, |d rho| <= {tol['t_rho']:g} and loglik_batch >= loglik_ref - "
         f"{tol['t_loglik']:g}.", ""]
    rows = []
    for eng, e in a["engines"].items():
        rows.append([eng, f"{e['diff_max_param_one_param_families']:.1e}",
                     f"{e['diff_max_loglik_one_param_families']:.1e}",
                     f"{e['diff_max_t_rho']:.1e}", f"{e['diff_max_t_rel_nu']:.1e}",
                     f"{e['diff_min_t_loglik_gain_vs_ref']:+.1e} / {e['diff_max_t_loglik_gain_vs_ref']:+.1e}",
                     f"{e['same_selected_family']} / {a['pairs']}", e["within_tolerance"]])
    caps = ", ".join(f"{eng} {e['pairs_hit_iteration_cap']}" for eng, e in a["engines"].items())
    L += [f"Pairs at the iteration cap: {caps}; scipy reference pairs without a success flag: "
          f"{a['reference_pairs_not_converged']} (of {a['pairs']}).", ""]
    L += _table(["engine", "max abs d theta (1-param)", "max abs d loglik (1-param)",
                 "max abs d rho (t)", "max rel d nu (t)", "t loglik gain vs ref (min / max)",
                 "same AIC choice", "within tolerance"], rows)
    return L + [""]


def _selection(s) -> list[str]:
    L = ["## 3. Family selection by AIC", "",
         f"{s['reps_per_family']} pairs per true family, tau drawn from U{tuple(s['tau_range'])} "
         f"(negated for rotations), t nu from U{tuple(s['t_nu_range'])}; candidates: all 11 "
         f"families (engine: {s['engine']}). Tail class = the set of corners with tail dependence.", ""]
    rows = [[b["n"], b["pairs"], b["pairs_hit_iteration_cap"], f"{b['accuracy']:.3f}",
             f"{b['tail_class_accuracy']:.3f}"]
            + [f"{b['per_family_accuracy'][t]:.2f}" for t in s["true_families"]] for b in s["by_n"]]
    L += _table(["n", "pairs", "at iteration cap", "accuracy", "tail-class accuracy"]
                + [SHORT[t] for t in s["true_families"]], rows)
    for b in s["by_n"]:
        L += ["", f"Confusion matrix, n = {b['n']} (rows: true, columns: selected)", ""]
        cands = s["candidates"]
        L += _table(["true \\ selected"] + [SHORT[c] for c in cands],
                    [[SHORT[t]] + [b["confusion"][t][c] or "." for c in cands]
                     for t in s["true_families"]])
    return L + [""]


def _tail(t) -> list[str]:
    L = ["## 4. Tail dependence", "",
         "Closed-form coefficients vs the empirical estimator lambda_L(q) = #{u <= q, v <= q} / (n q) "
         "(and the upper analogue). 'exact at q' is C(q, q) / q from the copula itself: what the "
         "estimator converges to at that q as n grows, which is not the limit lambda_L.", ""]
    rows = []
    for x in t["closed_form_vs_empirical"]:
        rows.append([x["family"], x["n"], x["samples"], f"{x['lambda_L']:.3f}", f"{x['lambda_U']:.3f}",
                     f"{x['emp_L_q0.01']:.3f} ({x['emp_L_q0.01_sd']:.3f})", f"{x['exact_L_q0.01']:.3f}",
                     f"{x['emp_L_q0.05']:.3f} ({x['emp_L_q0.05_sd']:.3f})", f"{x['exact_L_q0.05']:.3f}",
                     f"{x['emp_U_q0.01']:.3f}", f"{x['emp_U_q0.05']:.3f}"])
    L += _table(["family (tau 0.5)", "n", "samples", "lambda_L", "lambda_U",
                 "emp L q=0.01 mean (sd)", "exact at 0.01", "emp L q=0.05 mean (sd)",
                 "exact at 0.05", "emp U q=0.01", "emp U q=0.05"], rows)
    c = t["joint_crash"]
    tr = c["truth"]
    L += ["", f"Joint crashes. Truth: t copula, rho = {tr['rho']:.4f}, nu = {tr['nu']}. "
          f"{c['pairs']} samples of n = {c['n']}; each fitted with Gaussian and t. "
          f"AIC prefers t in {c['t_preferred_by_aic']} / {c['pairs']}. True lambda = "
          f"{c['true_lambda']:.3f}; fitted t lambda (mean) = {c['t_fit_lambda_mean']:.3f}; "
          f"Gaussian lambda = 0. Pairs at the iteration cap: {c['pairs_hit_iteration_cap']}. "
          f"Mean fitted rho: Gaussian {c['gaussian_rho_mean']:.4f}, "
          f"t {c['t_rho_mean']:.4f}; median fitted nu {c['t_nu_median']:.2f}.", ""]
    rows = [[x["q"], f"{x['true_prob']:.5f}", f"{x['independence_prob']:.5f}",
             f"{x['empirical_mean']:.5f}", f"{x['gaussian_fit_mean']:.5f}", f"{x['t_fit_mean']:.5f}",
             f"{x['gaussian_over_true']:.3f}", f"{x['t_over_true']:.3f}"] for x in c["rows"]]
    L += _table(["q", "true P(U<=q, V<=q)", "if independent", "empirical (mean)",
                 "Gaussian fit (mean)", "t fit (mean)", "Gaussian / true", "t / true"], rows)
    return L + [""]


def _demo(d) -> list[str]:
    cfg = d["config"]
    L = ["## 5. Demo: rolling fits through a stress regime", "",
         f"{cfg['n_assets']} assets, {cfg['days']} days. Days {cfg['stress_start']}-"
         f"{cfg['stress_end'] - 1} are a stress regime (Clayton copula, theta {cfg['stress_theta']}, "
         f"lambda_L {2 ** (-1 / cfg['stress_theta']):.3f}; vol {cfg['stress_vol']}); other days are "
         f"calm (Gaussian copula, rho {cfg['calm_rho']}; vol {cfg['calm_vol']}); t({cfg['marginal_df']:g}) "
         f"marginals. Rolling window {d['window']} days, step {d['step']}: {d['windows_total']} "
         f"windows x {d['pairs_per_window']} pairs = {d['fits']} fits (engine: {d['engine']}; "
         f"{d['pairs_hit_iteration_cap']} at the iteration cap). "
         f"Joint-crash probability at q = {d['q']} (q x window is a whole number of days, so the "
         f"empirical count uses exactly that share of each asset's days): true value {d['true_crash_prob']['calm']:.4f} "
         f"(calm) and {d['true_crash_prob']['stress']:.4f} (stress).", ""]
    rows = [[ph, v["windows"], f"{v['share_lower_tail_family']:.3f}", f"{v['mean_lambda_L_selected']:.3f}",
             f"{v['crash_prob_empirical']:.4f}", f"{v['crash_prob_selected']:.4f}",
             f"{v['crash_prob_gaussian']:.4f}"] for ph, v in d["phases"].items()]
    L += _table(["window phase", "windows", "share of pairs with a lower-tail family",
                 "mean lambda_L (selected)", "crash prob: empirical", "crash prob: selected family",
                 "crash prob: Gaussian fit"], rows)
    sc = d["stress_scenario"]
    L += ["", f"Stress scenario: 50/50 pair portfolio, empirical marginals of the stress window ending "
          f"day {sc['window_end']}, {sc['draws_per_pair']:,} copula draws per pair, averaged over "
          f"{sc['pairs']} pairs. 99% one-day VaR and ES as a fraction of the position.", ""]
    rows = [[k.replace("_", " "), f"{v['var99_mean']:.4f}", f"{v['es99_mean']:.4f}"]
            for k, v in sc["rows"].items()]
    L += _table(["copula", "VaR 99%", "ES 99%"], rows)
    return L + [""]


def _speed(out: Path) -> list[str]:
    files = sorted(out.glob("speed_*.json"))
    if not files:
        return []
    L = ["## 6. Throughput (machine-specific, never checked by tests)", ""]
    for p in files:
        s = json.loads(p.read_text(encoding="utf-8"))
        m = s["machine"]
        L += [f"### {s['label']}", "",
              f"{m['platform']}; {m['processor']}; {m['logical_cpus']} logical CPUs; Python "
              f"{m['python']}; JAX {m.get('jax', '-')} backend {m.get('jax_backend', '-')} "
              f"({m.get('jax_device_kind', '-')}, {len(m.get('jax_devices', []))} device(s)"
              f"{'; XLA_FLAGS ' + m['xla_flags'] if m.get('xla_flags') else ''}). "
              f"All 11 families + AIC per pair, n = {s['n']}. Batch engines: fastest of the timed "
              f"calls after one untimed warm-up call (the first call includes JAX compilation). "
              f"Reference loop: one timed call, no warm-up.", ""]
        rows = [[r["engine"], r["N"], f"{r['pairs_per_second']:,.1f}", f"{r['seconds']:.3f}",
                 r["timed_calls"], f"{r['first_call_seconds']:.2f}"] for r in s["rows"]]
        L += _table(["engine", "N pairs", "pairs / s", "seconds", "timed calls", "first call (s)"],
                    rows) + [""]
    return L


def render(out: Path) -> str:
    out = Path(out)
    L = ["# Results", "",
         "Synthetic data only. Rebuilt by `make bench` (sections 1-5, checked against a fresh run "
         "in CI) and `make speed` (section 6, machine-specific). Seeds: "
         f"{(_load(out, 'recovery') or {}).get('split', '?')} split.", ""]
    for name, fn in (("recovery", _recovery), ("agreement", _agreement), ("selection", _selection),
                     ("tail", _tail), ("demo", _demo)):
        data = _load(out, name)
        if data is not None:
            L += fn(data)
    L += _speed(out)
    return "\n".join(L).rstrip() + "\n"
