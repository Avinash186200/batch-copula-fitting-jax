import csv
import json
import subprocess
import sys

import numpy as np

from copulabatch.cli import main
from copulabatch.sim import RegimeConfig, regime_returns


def test_fit_csv_prints_one_row_per_pair(tmp_path, capsys):
    R = regime_returns(RegimeConfig(n_assets=3, days=300, stress_start=100, stress_end=200),
                       np.random.default_rng(0))
    p = tmp_path / "returns.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "A", "B", "C"])
        for i, row in enumerate(R):
            w.writerow([f"d{i}", *row])
    main(["fit-csv", str(p), "--index-col", "--window", "250", "--engine", "numpy"])
    rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    assert rows[0][:3] == ["asset_a", "asset_b", "family"]
    assert [r[:2] for r in rows[1:]] == [["A", "B"], ["A", "C"], ["B", "C"]]
    assert all(0.0 <= float(r[6]) <= 1.0 for r in rows[1:])
    assert rows[0][-1] == "converged" and all(r[-1] in ("0", "1") for r in rows[1:])


def test_speed_quick_run_writes_json(tmp_path):
    main(["speed", "--out", str(tmp_path), "--label", "ci", "--engines", "reference,numpy",
          "--max-n", "4", "--n", "100", "--repeats", "1"])
    s = json.loads((tmp_path / "speed_ci.json").read_text())
    assert {r["engine"] for r in s["rows"]} == {"reference", "numpy"}
    assert all(r["pairs_per_second"] > 0 for r in s["rows"])   # printed, never compared


def test_module_entry_point_help():
    out = subprocess.run([sys.executable, "-m", "copulabatch.cli", "--help"],
                         capture_output=True, text=True, check=True)
    assert "bench" in out.stdout and "fit-csv" in out.stdout
