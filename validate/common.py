import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import train  # noqa: E402


def dirs(quick):
    """(runs dir, results dir). --quick output goes under runs/_quick so it never mixes with results/."""
    if quick:
        base = ROOT / "runs" / "_quick"
        return base / "runs", base
    return ROOT / "runs" / "validate", ROOT / "results"


def train_run(runs_dir, name, env, episodes, **flags):
    """Train a run, or reuse it if a finished run with the same config already exists."""
    run = Path(runs_dir) / name
    argv = ["--env", env, "--name", name, "--runs_dir", str(runs_dir), "--episodes", str(episodes)]
    for k, v in flags.items():
        argv += [f"--{k}", str(v)]
    want, _ = train.parse_args(argv)
    cfg_path, csv_path = run / "config.json", run / "train.csv"
    if cfg_path.exists() and csv_path.exists() and (run / "last.pt").exists():
        cfg = json.loads(cfg_path.read_text())
        keys = set(train.DEFAULTS[env]) - train.RESUME_OVERRIDES | {"seed", "episodes"}
        if all(cfg[k] == getattr(want, k) for k in keys) and len(pd.read_csv(csv_path)) == episodes:
            print(f"reusing finished run {run}")
            return run
    if run.exists():
        # Stale or partial run created by this script; retrain from scratch.
        shutil.rmtree(run)
    train.main(argv)
    return run


def record(results_dir, tid, what, expected, measured, passed):
    path = Path(results_dir) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary[tid] = {"what": what, "expected": expected, "measured": measured, "pass": passed}
    path.write_text(json.dumps(summary, indent=2))
    print(f"{tid} pass={passed}: {measured}")
