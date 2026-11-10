"""T5: does the trained Q predict the return it actually gets?

From eval_trace.csv (first 20 eval episodes, written by evaluate.py): predicted max_a Q(s_t, a) vs realized
discounted return G_t = r_t + gamma * (1 - done_t) * G_{t+1}, with the same reward processing as training
(clipped reward, life loss terminal for Pac-Man). If an episode was cut by the time limit, G_t near the end
misses its tail, so the last 3/(1-gamma) steps of truncated episodes are dropped.
"""
import argparse
import json
from pathlib import Path

from common import ROOT, dirs, record

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plots import BLUE, MUTED, save, style  # noqa: E402


def returns(tr, gamma):
    keep, g_all = [], []
    for _, ep in tr.groupby("episode", sort=True):
        r, d = ep.r_learn.to_numpy(), ep.done_learn.to_numpy()
        g = np.zeros(len(ep))
        nxt = 0.0
        for t in range(len(ep) - 1, -1, -1):
            nxt = r[t] + gamma * (1 - d[t]) * nxt
            g[t] = nxt
        k = np.ones(len(ep), bool)
        if ep.trunc.iloc[-1] and not ep.term.iloc[-1]:
            k[max(0, len(ep) - int(3 / (1 - gamma))):] = False
        keep.append(k)
        g_all.append(g)
    return np.concatenate(g_all), np.concatenate(keep)


def analyse(run):
    run = Path(run)
    trace = run / "eval_trace.csv"
    if not trace.exists():
        raise SystemExit(f"{trace} not found; run evaluate.py --run {run} first")
    cfg = json.loads((run / "config.json").read_text())
    tr = pd.read_csv(trace)
    g, keep = returns(tr, cfg["gamma"])
    q = tr.maxq.to_numpy()[keep]
    g = g[keep]
    bias = float(np.mean(q - g))
    r = float(np.corrcoef(q, g)[0, 1]) if np.std(q) > 0 and np.std(g) > 0 else float("nan")
    return cfg, q, g, {"n_steps": int(len(q)), "episodes": int(tr.episode.nunique()), "mean_q": round(float(q.mean()), 3),
                       "mean_G": round(float(g.mean()), 3), "bias": round(bias, 3),
                       "rel_bias": round(bias / float(g.mean()), 4) if g.mean() else None,
                       "pearson_r": round(r, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="write to runs/_quick (point --*_run at short runs)")
    p.add_argument("--cartpole_run", default=str(ROOT / "runs" / "cartpole_main"))
    p.add_argument("--pacman_run", default=str(ROOT / "runs" / "pacman_main"))
    a = p.parse_args()
    _, res = dirs(a.quick)

    style()
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    measured, ok = {}, True
    for ax, (game, run) in zip(axes, (("cartpole", a.cartpole_run), ("pacman", a.pacman_run))):
        cfg, q, g, m = analyse(run)
        measured[game] = m
        ok &= m["rel_bias"] is not None and abs(m["rel_bias"]) < 0.10
        ax.scatter(g, q, s=3, color=BLUE, alpha=0.25, lw=0)
        lo, hi = min(q.min(), g.min()), max(q.max(), g.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1, label="y = x")
        unit = "clipped reward" if game == "pacman" else "reward"
        ax.set(xlabel=f"realized return G$_t$ [{unit}]", ylabel=f"predicted max$_a$ Q(s$_t$,a) [{unit}]",
               title=f"{game}: bias {m['bias']:+.2f}, r = {m['pearson_r']:.2f}")
        ax.legend(loc="upper left")
        print(game, m)
    fig.suptitle("Predicted Q vs realized discounted return (eval episodes)", fontsize=10)
    save(fig, res, "t5_q_vs_return.png")
    record(res, "T5", "Predicted max Q vs realized discounted return on eval episodes (positive bias = overestimation)",
           "bias near 0 (|bias| < 10% of mean G)", measured, bool(ok))


if __name__ == "__main__":
    main()
