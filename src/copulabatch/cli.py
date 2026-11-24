"""Command line: copulabatch bench | speed | report | fit-csv."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _speed(args) -> None:
    if args.cpu_devices:
        # must be set before JAX is first imported
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   f" --xla_force_host_platform_device_count={args.cpu_devices}").strip()
    if args.cpu_only:
        os.environ["JAX_PLATFORMS"] = "cpu"
    from . import speed
    from .bench import write_json
    from .report import render

    sizes = {}
    if args.max_n:
        sizes = {e: tuple(N for N in speed.DEFAULT_N[e] if N <= args.max_n) or (args.max_n,)
                 for e in speed.DEFAULT_N}
    res = speed.run(engines=tuple(args.engines.split(",")), n=args.n, sizes=sizes,
                    repeats=args.repeats)
    res["label"] = args.label
    if args.no_write:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"speed_{args.label}.json", res)
    if all((out / f"{k}.json").exists() for k in ("recovery", "agreement", "selection", "tail", "demo")):
        (out / "RESULTS.md").write_text(render(out), encoding="utf-8", newline="\n")


def _fit_csv(args) -> None:
    import csv
    from itertools import combinations

    import numpy as np

    from . import families as F
    from .engines import fit, jax_available

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    cols = header[1:] if args.index_col else header
    X = np.array([[float(x) for x in (r[1:] if args.index_col else r)] for r in body])
    X = X[-args.window:] if args.window else X
    prs = list(combinations(range(X.shape[1]), 2))
    U = np.stack([F.pseudo_obs(X[:, i]) for i, _ in prs])
    V = np.stack([F.pseudo_obs(X[:, j]) for _, j in prs])
    res = fit(U, V, engine=args.engine or ("jax" if jax_available() else "numpy"))
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["asset_a", "asset_b", "family", "param", "nu", "tau", "lambda_L", "lambda_U", "aic",
                "converged"])
    for k, (i, j) in enumerate(prs):
        name = res.families[res.best[k]]
        p = res.params(k, name)
        lo, up = F.tail_dependence(name, p)
        jb = res.best[k]
        w.writerow([cols[i], cols[j], name, f"{res.theta[k, jb]:.6g}",
                    f"{res.nu[k, jb]:.6g}" if name == "t" else "", f"{F.tau(name, p):.4f}",
                    f"{lo:.4f}", f"{up:.4f}", f"{res.aic[k, jb]:.3f}",
                    int(res.converged[k].all())])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="copulabatch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="all experiments; writes <out>/*.json and RESULTS.md")
    b.add_argument("--out", default="results")
    b.add_argument("--split", choices=("holdout", "dev"), default="holdout")
    b.add_argument("--engine", choices=("numpy", "jax"), default=None)

    s = sub.add_parser("speed", help="pairs/second vs N; writes <out>/speed_<label>.json")
    s.add_argument("--out", default="results")
    s.add_argument("--label", default="local")
    s.add_argument("--engines", default="reference,numpy,jax")
    s.add_argument("--n", type=int, default=500, help="observations per pair")
    s.add_argument("--max-n", type=int, default=0, help="cap on N (pairs) for a quick run")
    s.add_argument("--repeats", type=int, default=2)
    s.add_argument("--cpu-devices", type=int, default=0,
                   help="split the CPU into this many XLA devices (JAX pmap over cores)")
    s.add_argument("--cpu-only", action="store_true", help="hide any GPU from JAX")
    s.add_argument("--no-write", action="store_true")

    r = sub.add_parser("report", help="re-render RESULTS.md from <out>/*.json")
    r.add_argument("--out", default="results")

    f = sub.add_parser("fit-csv", help="fit every pair of columns of a returns CSV; CSV to stdout")
    f.add_argument("csv")
    f.add_argument("--index-col", action="store_true", help="first column is a date/index")
    f.add_argument("--window", type=int, default=0, help="use only the last N rows")
    f.add_argument("--engine", choices=("reference", "numpy", "jax"), default=None)

    args = ap.parse_args(argv)
    if args.cmd == "bench":
        from .bench import run
        run(Path(args.out), split=args.split, engine=args.engine)
    elif args.cmd == "speed":
        _speed(args)
    elif args.cmd == "report":
        from .report import render
        out = Path(args.out)
        (out / "RESULTS.md").write_text(render(out), encoding="utf-8", newline="\n")
    elif args.cmd == "fit-csv":
        _fit_csv(args)
    else:  # pragma: no cover
        ap.error(json.dumps(args.cmd))


if __name__ == "__main__":
    main()
