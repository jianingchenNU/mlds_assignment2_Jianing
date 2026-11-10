"""T3: CartPole with a 50-step time limit.

Bootstrapping through truncation should learn Q ~ 1/(1-gamma) = 20: the time limit is not part of the task,
so a truncated step is valued as if the episode went on.
Treating truncation as terminal is a negative control. Time is not in the state, so the same state is
terminal at step 49 and not at step 0: the target is non-Markov and the learned Q is expected to be unstable
rather than to settle at a particular value.
The dashed line at (1 - gamma^50)/(1 - gamma) = 18.46 is the largest discounted return any 50-step episode
can actually pay; a bootstrapped Q above it is correct, because it values the untruncated task.

Run length: with hard target syncs, Q after K syncs is about 20 * (1 - gamma^K), so the bootstrap run needs
on the order of 100+ syncs (target_sync 500 env steps) to get close to 20; 1500 episodes gives ~150.
"""
import argparse

from common import dirs, record, train_run

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from plots import BLUE, MUTED, ORANGE, ma100, save, style  # noqa: E402

GAMMA = 0.95
T = 50


def stats(df):
    late = df.episode > 300
    return {"q_last50": round(float(df.mean_maxq.tail(50).mean()), 3),
            "q_ma100_max": round(float(ma100(df.mean_maxq).max()), 3),
            "loss_last500": round(float(df.loss.tail(500).mean()), 5),
            "reward_ma100_min_after_ep300": round(float(ma100(df.reward)[late].min()), 2) if late.any() else None,
            "frac_last50_reaching_cap": round(float((df.length.tail(50) == T).mean()), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--episodes", type=int, default=None, help="default 1500 (quick: 15)")
    a = p.parse_args()
    n = a.episodes or (15 if a.quick else 1500)
    runs, res = dirs(a.quick)

    boot = 1 / (1 - GAMMA)
    cap = (1 - GAMMA ** T) / (1 - GAMMA)
    measured, dfs = {}, {}
    for mode, flag in (("bootstrap", 0), ("terminal", 1)):
        run = train_run(runs, f"t3_{mode}", "cartpole", n, max_steps=T, trunc_as_terminal=flag,
                        log_every=100, **({"warmup": 200} if a.quick else {}))
        dfs[mode] = pd.read_csv(run / "train.csv")
        measured[mode] = stats(dfs[mode])
    q = measured["bootstrap"]["q_last50"]
    err = (q - boot) / boot
    measured["bootstrap"]["rel_err"] = round(err, 4)
    measured["gap_bootstrap_minus_terminal"] = round(q - measured["terminal"]["q_last50"], 3)
    ok = abs(err) <= 0.10

    style()
    fig, ax = plt.subplots()
    for (mode, df), c in zip(dfs.items(), (BLUE, ORANGE)):
        ax.plot(df.episode, df.mean_maxq, color=c, lw=0.5, alpha=0.4)
        ax.plot(df.episode, ma100(df.mean_maxq), color=c, lw=2, label=f"{mode} on truncation")
    ax.axhline(boot, color=BLUE, ls="--", lw=1, label=f"1/(1-γ) = {boot:.0f}")
    ax.axhline(cap, color=MUTED, ls="--", lw=1, label=f"max true return, {T}-step cap = {cap:.2f}")
    ax.set(xlabel="training episode", ylabel="mean max$_a$ Q(s,a) [reward]",
           title=f"CartPole with time limit {T}: truncation handling")
    ax.legend()
    save(fig, res, "t3_truncation.png")
    record(res, "T3", f"CartPole with time limit {T}: bootstrap through truncation vs truncation as terminal "
           "(negative control: non-Markov target, expected unstable)",
           {"bootstrap": f"q_last50 within 10% of 1/(1-gamma) = {boot:.0f} (pass criterion)",
            "terminal": "negative control, no target value; expected unstable Q, loss and reward",
            "max_true_return_50_step_cap": round(cap, 3)},
           measured, bool(ok))


if __name__ == "__main__":
    main()
