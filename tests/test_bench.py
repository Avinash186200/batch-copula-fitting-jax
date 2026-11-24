import json
import math
from pathlib import Path

import pytest

from copulabatch.bench import RESULT_FILES, run, small_sizes

ROOT = Path(__file__).resolve().parents[1]


def test_small_bench_end_to_end(tmp_path):
    res = run(tmp_path, split="dev", sizes=small_sizes(), log=lambda *_: None)
    for k in RESULT_FILES:
        assert (tmp_path / f"{k}.json").exists()
    assert (tmp_path / "RESULTS.md").read_text(encoding="utf-8").startswith("# Results")
    for e in res["agreement"]["engines"].values():
        assert e["within_tolerance"]
    b = res["selection"]["by_n"][0]
    assert sum(sum(row.values()) for row in b["confusion"].values()) == b["pairs"]


def test_results_written_with_lf_only():
    for k in RESULT_FILES:
        assert b"\r\n" not in (ROOT / "results" / f"{k}.json").read_bytes()
    assert b"\r\n" not in (ROOT / "results" / "RESULTS.md").read_bytes()


def test_committed_results_are_from_the_holdout_split():
    for k in RESULT_FILES:
        assert json.loads((ROOT / "results" / f"{k}.json").read_text())["split"] == "holdout"


def _diff(a, b, path="$"):
    """Differences between two JSON trees. Floats match to 1e-4 relative or 1e-6 absolute.
    Different CPUs and OSes round the last bits differently, and a last-bit difference can
    send Brent down a different (equally valid) path that stops elsewhere within its
    tolerance; in flat likelihoods (nu, Frank theta) that moves a parameter by more than
    xtol. The tolerance allows for such solver-path differences, not for method changes.
    Keys starting with 'diff_' are engine-vs-reference gaps near rounding level; they are
    checked against their tolerances by 'within_tolerance' instead of value by value."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(a.keys() | b.keys()):
            if k.startswith("diff_"):
                continue
            if k == "reference_pairs_not_converged":
                # scipy L-BFGS-B's success flag depends on the platform's floating point
                # (1 of 54 on Windows, 3 of 54 on Linux CI); the fitted values still agree.
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k}: missing")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6)
        return [] if ok else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    pytest.importorskip("jax")     # the committed results were made with the JAX engine
    out = tmp_path_factory.mktemp("full")
    run(out, log=lambda *_: None)
    return out


@pytest.mark.parametrize("name", RESULT_FILES)
def test_committed_results_reproduce(full_run, name):
    fresh = json.loads((full_run / f"{name}.json").read_text())
    committed = json.loads((ROOT / "results" / f"{name}.json").read_text())
    diffs = _diff(fresh, committed)
    assert not diffs, "\n".join(diffs[:20])
