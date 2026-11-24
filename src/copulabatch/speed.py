"""Throughput benchmark: pairs fitted per second (all 11 families + AIC choice) versus N.

Machine-specific: the numbers depend on the CPU, GPU, OS and library versions, so they are
written to results/speed_<label>.json with the machine description and never checked by
tests. For each engine and N: one untimed warm-up call (for JAX this compiles; the compile
time is recorded separately), then ``repeats`` timed calls; the fastest is reported.
"""
from __future__ import annotations

import os
import platform
import time

import numpy as np

from .bench import _mixed_specs, _rng
from .engines import fit, jax_available
from .sim import pairs

DEFAULT_N = {"reference": (10, 30, 100), "numpy": (10, 100, 1000),
             "jax": (10, 100, 1000, 4000)}


def machine() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or "unknown", "logical_cpus": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__}
    if jax_available():
        import jax
        import jaxlib

        info.update({"jax": jax.__version__, "jaxlib": jaxlib.__version__,
                     "jax_backend": jax.default_backend(),
                     "jax_devices": [str(d) for d in jax.local_devices()],
                     "jax_device_kind": jax.local_devices()[0].device_kind,
                     "xla_flags": os.environ.get("XLA_FLAGS", "")})
    return info


def run(engines=("reference", "numpy", "jax"), n: int = 500, sizes: dict | None = None,
        repeats: int = 2, log=print) -> dict:
    sizes = {**DEFAULT_N, **(sizes or {})}
    n_max = max(max(sizes[e]) for e in engines)
    rng = _rng("holdout", 6)
    U, V = pairs(_mixed_specs(rng, n_max), n, rng)
    rows = []
    for eng in engines:
        for N in sizes[eng]:
            u, v = U[:N], V[:N]
            t0 = time.perf_counter()
            fit(u, v, engine=eng)
            first = time.perf_counter() - t0
            reps = 1 if eng == "reference" else repeats
            times = [first] if eng == "reference" else []
            for _ in range(reps if eng != "reference" else 0):
                t0 = time.perf_counter()
                fit(u, v, engine=eng)
                times.append(time.perf_counter() - t0)
            best = min(times)
            row = {"engine": eng, "N": N, "n": n, "seconds": round(best, 4),
                   "pairs_per_second": round(N / best, 1),
                   "first_call_seconds": round(first, 3), "timed_calls": len(times)}
            rows.append(row)
            log(f"{eng:9s} N={N:6d}  {row['pairs_per_second']:9.1f} pairs/s  "
                f"(first call {first:.2f}s)")
    return {"machine": machine(), "families": 11, "n": n, "rows": rows}
