# Contributing

Thanks for taking a look. Changes are judged on whether they keep the repo offline, correct against the reference fitter, and honest about its numbers.

## Ground rules

1. **Everything in CI runs offline and on CPU.** Tests and `make bench` need no network and no GPU. JAX is installed from PyPI as CPU jaxlib.
2. **Numbers come from results.** Any number in a doc must be in a `results/*.json` file written by `make bench` or `make speed`. If you change code that affects results, rerun `make bench` and commit the new JSON and `RESULTS.md` with the change. CI fails if the committed JSON does not match a fresh run.
3. **Never tune on the held-out seeds.** Solver settings, bounds, tolerances and experiment designs are chosen on the development split (`--split dev`). If a change needs a new choice, make it there, and say in `docs/methodology.md` what changed.
4. **Engines must agree.** A change to a density, a bound or the solver must keep `tests/test_engines.py` passing: the NumPy and JAX engines within the stated tolerances of the scipy reference, with the same AIC choice.
5. **No timing assertions in tests.** CI runners are shared and slow; tests check correctness, never speed. Throughput numbers are printed and written to `results/speed_*.json`, never compared.
6. **Seeds stay fixed.** Every generator takes a seed or a `numpy.random.Generator`.
7. **No secrets.** Nothing here needs credentials; `.env` is git-ignored and CI runs a secret scan.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[dev,jax]"
pytest -q
```

## Adding a family

1. Add its log-density to `families.py` (written against `ops.xp` so NumPy and JAX can both run it), plus `tau`, `param_from_tau`, `tail_dependence`, `tail_corners`, `_base_cdf` and a sampler.
2. Register it in `FAMILIES` with parameter bounds, and teach `objective.make_nll` its transforms.
3. `tests/test_families.py` then checks the density against the CDF, the sampler against Kendall's tau, and the tail coefficients; `tests/test_engines.py` checks the engines.
4. Rerun `make bench` and update the docs from the new JSON.
