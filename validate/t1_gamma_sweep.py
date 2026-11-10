"""T1: learned max Q on CartPole vs 1/(1-gamma), the value of never failing (reward 1 per step forever).

Horizon argument: the Q gap between never failing and failing after 20 steps is gamma^20/(1-gamma).
At gamma 0.8 that gap is tiny, so the agent has almost no signal to avoid a late failure and should never
balance 500 steps; at gamma >= 0.9 the gap is large enough that it should. Max Q still approaches 1/(1-gamma)
in all cases, because a pole that falls in 20+ steps already earns most of the discounted value.
"""
import argparse

from common import dirs, record, train_run

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plots import AQUA, BLUE, ORANGE, ma100, save, style  # noqa: E402

GAMMAS = [0.8, 0.9, 0.95]
SEEDS = [0, 1]
GAP_STEPS = 20


def gap(g):
    return g ** GAP_STEPS / (1 - g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--episodes", type=int, default=None, help="default 500 (quick: 10)")
    a = p.parse_args()
    eps_n = a.episodes or (10 if a.quick else 500)
    runs, res = dirs(a.quick)

    rows = []
    curves = {}
    for g in GAMMAS:
        for s in SEEDS:
            run = train_run(runs, f"t1_g{g}_s{s}", "cartpole", eps_n, gamma=g, seed=s, log_every=100,
                            **({"warmup": 200} if a.quick else {}))
            df = pd.read_csv(run / "train.csv")
            curves[(g, s)] = df
            rows.append({"gamma": g, "seed": s, "q_last50": float(df.mean_maxq.tail(50).mean()),
                         "bound": 1 / (1 - g), "ma100_final": float(df.reward.tail(100).mean()),
                         "max_len": int(df.length.max())})
    t = pd.DataFrame(rows)
    t["rel_err"] = (t.q_last50 - t.bound) / t.bound
    print(t.to_string(index=False))

    measured, ok = {}, True
    for g in GAMMAS:
        sub = t[t.gamma == g]
        q = float(sub.q_last50.mean())
        err = (q - 1 / (1 - g)) * (1 - g)
        measured[str(g)] = {"q": round(q, 3), "bound": round(1 / (1 - g), 3), "rel_err": round(err, 4),
                            "q_pass": bool(abs(err) <= 0.10), "gap_fail_at_20": round(gap(g), 3),
                            "ma100_final": [round(x, 1) for x in sub.ma100_final],
                            "max_len": [int(x) for x in sub.max_len], "seeds": SEEDS}
        reach = bool((sub.max_len >= 500).all()) if g >= 0.9 else bool((sub.max_len < 500).all())
        measured[str(g)]["max_len_pass"] = reach
        ok &= abs(err) <= 0.10 and reach

    style()
    fig, ax = plt.subplots()
    for g, c in zip(GAMMAS, (BLUE, ORANGE, AQUA)):
        q = np.mean([ma100(curves[(g, s)].mean_maxq).to_numpy() for s in SEEDS], axis=0)
        ax.plot(curves[(g, SEEDS[0])].episode, q, color=c, lw=2, label=f"γ = {g}")
        ax.axhline(1 / (1 - g), color=c, ls="--", lw=1)
    ax.set(xlabel="training episode", ylabel="max$_a$ Q(s,a), MA100 [reward]",
           title="Learned max Q vs 1/(1-γ) (dashed), mean of 2 seeds")
    ax.legend()
    fig.text(0.5, -0.03, "Q gap, never failing vs failing after 20 steps = γ$^{20}$/(1-γ): "
             + ", ".join(f"{gap(g):.3g} (γ = {g})" for g in GAMMAS), ha="center", fontsize=8)
    save(fig, res, "t1_gamma_sweep.png")
    record(res, "T1", "CartPole gamma sweep: mean max Q over the last 50 episodes of each run (2 seeds), "
           "plus final MA100 reward and max episode length per seed",
           {"q": "within 10% of 1/(1-gamma) = 5, 10, 20 (pass criterion)",
            "horizon": "Q gap between never failing and failing after 20 steps = gamma^20/(1-gamma) = "
                       + ", ".join(f"{gap(g):.3g} ({g})" for g in GAMMAS)
                       + ". Max episode length: gamma=0.8 never reaches 500, gamma>=0.9 does (pass criterion, every seed)"},
           measured, bool(ok))


if __name__ == "__main__":
    main()
