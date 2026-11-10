import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#2b2b2b"
MUTED = "#8a8a85"
# Single-hue sequential ramp (light = low) for magnitudes: V(s), visit counts.
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
PREFIX = {"cartpole": "cartpole", "pacman": "pacman"}


def style():
    plt.rcParams.update({
        "figure.figsize": (6.4, 3.6), "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
        "legend.frameon": False, "axes.axisbelow": True, "hatch.color": "#7a7a75",
    })


def ma100(x):
    return pd.Series(x).rolling(100, min_periods=1).mean()


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    plt.close(fig)
    print(f"wrote {out / name}")


def plot_maxq(df, cfg, out, pre):
    fig, ax = plt.subplots()
    ax.plot(df.episode, df.mean_maxq, color=BLUE_LIGHT, lw=0.6, label="per episode")
    ax.plot(df.episode, ma100(df.mean_maxq), color=BLUE, lw=2, label="moving average (100 ep)")
    # Pac-Man's bound (100) sits far above the learned values and flattens the curve, so only CartPole shows it.
    if cfg["env"] == "cartpole":
        bound = 1 / (1 - cfg["gamma"])
        ax.axhline(bound, color=MUTED, ls="--", lw=1, label=f"upper bound 1/(1-γ) = {bound:.0f}")
    unit = "clipped reward" if cfg["env"] == "pacman" else "reward"
    ax.set(xlabel="training episode", ylabel=f"mean of max$_a$ Q(s,a) [{unit}]",
           title=f"Max Q-value during training (γ = {cfg['gamma']})")
    ax.legend()
    save(fig, out, f"{pre}_maxq.png")


def plot_reward(df, checks, cfg, out, pre):
    fig, ax = plt.subplots()
    ax.plot(df.episode, df.reward, color=BLUE_LIGHT, lw=0.6, label="per episode")
    ax.plot(df.episode, ma100(df.reward), color=BLUE, lw=2, label="moving average (last 100 ep)")
    if checks is not None:
        ax.plot(checks.episode, checks["mean"], ls="none", marker="o", ms=4, color=ORANGE,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"greedy check (eps={cfg['check_eps']:g}, mean of {cfg['check_episodes']})")
    if cfg["env"] == "cartpole":
        ax.axhline(500, color=MUTED, ls="--", lw=1, label="max (500 steps)")
    ylabel = "episode reward [raw game score]" if cfg["env"] == "pacman" else "episode reward [steps balanced]"
    ax.set(xlabel="training episode", ylabel=ylabel, title="Episode reward during training")
    if cfg["env"] == "cartpole":
        # Opaque box so curves behind the legend do not show through its text.
        ax.legend(loc="center left", frameon=True, framealpha=1, facecolor="white", edgecolor="none")
    else:
        ax.legend(loc="best")
    save(fig, out, f"{pre}_reward.png")


def plot_eval_hist(ev, summ, cfg, out, pre):
    fig, ax = plt.subplots()
    label = f"mean {summ['mean']:.1f} ± std {summ['std']:.1f}\n(n = {summ['n']}, ε = {summ['eps']})"
    if cfg["env"] == "cartpole":
        bins = np.arange(5 * np.floor(ev.reward.min() / 5), ev.reward.max() + 10, 5)
        label += f"\n{int((ev.reward >= 500).sum())}/{summ['n']} reached 500 steps"
    else:
        bins = np.histogram_bin_edges(ev.reward, "auto")
    ax.hist(ev.reward, bins=bins, color=BLUE, edgecolor="white", lw=0.5)
    ax.axvline(summ["mean"], color=ORANGE, lw=2, label=label)
    unit = "raw game score" if cfg["env"] == "pacman" else "steps balanced"
    ax.set(xlabel=f"episode reward [{unit}]", ylabel="episodes", title="Rollouts of the trained network")
    ax.legend(loc="upper right" if cfg["env"] == "pacman" else "best")
    save(fig, out, f"{pre}_eval_hist.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run, out = Path(a.run), Path(a.out)
    cfg = json.loads((run / "config.json").read_text())
    pre = PREFIX[cfg["env"]]
    style()
    df = pd.read_csv(run / "train.csv")
    plot_maxq(df, cfg, out, pre)
    checks = pd.read_csv(run / "checks.csv") if (run / "checks.csv").exists() else None
    plot_reward(df, checks, cfg, out, pre)
    if (run / "eval.json").exists():
        plot_eval_hist(pd.read_csv(run / "eval.csv"), json.loads((run / "eval.json").read_text()), cfg, out, pre)
    else:
        print(f"no {run / 'eval.json'}; run evaluate.py first for {pre}_eval_hist.png")


if __name__ == "__main__":
    main()
