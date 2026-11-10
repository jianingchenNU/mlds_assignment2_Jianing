"""T4: remove each DQN ingredient on CartPole and compare across 3 seeds.

full: default DQN. no_target: target net synced every env step (target = online). no_replay: buffer of 64 = batch,
so each update uses only the most recent transitions (strongly correlated, no reuse of old data).

Training MA100 cannot reach 475 with eps = 0.02 exploration, so each run is summarized by its best greedy check
(checks.csv, 10 greedy episodes every 10), final MA100 and peak mean_maxq.
Prediction: without a target net, Q chases its own moving target and diverges past 1/(1-gamma) = 20; and the
ordering full > no_replay > no_target holds for both the mean best greedy check and the mean final MA100.
"""
import argparse

from common import dirs, record, train_run

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plots import AQUA, BLUE, MUTED, ORANGE, ma100, save, style  # noqa: E402

VARIANTS = {"full": {}, "no_target": {"target_sync": 1}, "no_replay": {"buffer": 64, "warmup": 64}}
SEEDS = [0, 1, 2]
GAMMA = 0.95
KEYS = ["best_check", "final_ma100", "peak_maxq"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--episodes", type=int, default=None, help="default 500 (quick: 10)")
    a = p.parse_args()
    n = a.episodes or (10 if a.quick else 500)
    runs, res = dirs(a.quick)
    bound = 1 / (1 - GAMMA)

    measured, rew, qs = {}, {}, {}
    for v, flags in VARIANTS.items():
        per_seed = []
        for s in SEEDS:
            f = dict(flags)
            if a.quick and "warmup" not in f:
                f["warmup"] = 200
            run = train_run(runs, f"t4_{v}_s{s}", "cartpole", n, seed=s, log_every=100, **f)
            df = pd.read_csv(run / "train.csv")
            checks = pd.read_csv(run / "checks.csv")
            per_seed.append({"seed": s, "best_check": round(float(checks["mean"].max()), 1),
                             "final_ma100": round(float(df.reward.tail(100).mean()), 1),
                             "peak_maxq": round(float(df.mean_maxq.max()), 3)})
            rew[(v, s)] = ma100(df.reward).to_numpy()
            qs[(v, s)] = ma100(df.mean_maxq).to_numpy()
        mean = {k: round(float(np.mean([r[k] for r in per_seed])), 3) for k in KEYS}
        measured[v] = {"per_seed": per_seed, "mean": mean}
        print(v, mean)

    m = {v: measured[v]["mean"] for v in VARIANTS}
    diverges = m["no_target"]["peak_maxq"] > bound
    order = {k: bool(m["full"][k] > m["no_replay"][k] > m["no_target"][k]) for k in ("best_check", "final_ma100")}
    measured["checks"] = {"no_target_peak_maxq_gt_20": bool(diverges),
                          "order_best_check": order["best_check"], "order_final_ma100": order["final_ma100"]}
    ok = diverges and all(order.values())

    style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    ep = np.arange(1, n + 1)
    for (v, _), c in zip(VARIANTS.items(), (BLUE, ORANGE, AQUA)):
        for ax, d in ((ax1, rew), (ax2, qs)):
            arr = np.stack([d[(v, s)] for s in SEEDS])
            ax.fill_between(ep, arr.min(0), arr.max(0), color=c, alpha=0.18, lw=0)
            ax.plot(ep, arr.mean(0), color=c, lw=2, label=v.replace("_", " "))
    ax1.set(xlabel="training episode", ylabel="episode reward, MA100 [steps]", title="(a) Reward")
    ax1.legend()
    line = ax2.axhline(bound, color=MUTED, ls="--", lw=1, label=f"1/(1-γ) = {bound:.0f}")
    ax2.set_yscale("log")
    # Q is ~0 during warmup; clip so the log axis is spent on the learning phase.
    ax2.set_ylim(bottom=1e-2)
    ax2.set(xlabel="training episode", ylabel="mean max$_a$ Q(s,a), MA100 [reward, log]", title="(b) Max Q")
    ax2.legend(handles=[line], loc="upper left")
    fig.suptitle("CartPole ablation: mean (line) and range (band) over 3 seeds", fontsize=10)
    save(fig, res, "t4_ablation.png")
    record(res, "T4", "CartPole ablation, 3 seeds x {full DQN, no target net, no replay}: best greedy check, "
           "final MA100 and peak mean_maxq per seed and mean over seeds",
           "no_target diverges (mean peak mean_maxq > 1/(1-gamma) = 20); full > no_replay > no_target for both "
           "the mean best greedy check and the mean final MA100", measured, bool(ok))


if __name__ == "__main__":
    main()
