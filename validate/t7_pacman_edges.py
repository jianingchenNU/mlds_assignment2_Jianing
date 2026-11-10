"""T7: Pac-Man edge-case counters from a short training run, plus a random-policy baseline.

- frame stacks sampled for updates that would mix two episodes without the episode-start mask, and after it
- life-loss terminals (learning target only, game continues) vs game-over terminals vs time-limit truncations
- raw reward values seen vs the clipped values used for learning
- random-policy score vs the trained agent's eval score (read from the main run's eval.json)
"""
import argparse
import json

from common import ROOT, dirs, record, train_run

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluate import run_eval  # noqa: E402
from plots import BLUE, ORANGE, save, style  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--episodes", type=int, default=None, help="short training run; default 30 (~20k steps), quick 2")
    p.add_argument("--warmup", type=int, default=None, help="default 5000, quick 500")
    p.add_argument("--random_episodes", type=int, default=None, help="default 100, quick 2")
    p.add_argument("--main_run", default=str(ROOT / "runs" / "pacman_main"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    n = a.episodes or (2 if a.quick else 30)
    warmup = a.warmup or (500 if a.quick else 5000)
    n_rand = a.random_episodes or (2 if a.quick else 100)
    runs, res = dirs(a.quick)

    run = train_run(runs, "t7_pacman", "pacman", n, warmup=warmup, device=a.device, log_every=10)
    c = json.loads((run / "counters.json").read_text())
    rand = run_eval(run, n_rand, random=True, ckpt="last.pt", device=a.device)

    main_eval = ROOT / a.main_run / "eval.json"
    trained = json.loads(main_eval.read_text())["mean"] if main_eval.exists() else None
    if trained is None:
        print(f"{main_eval} not found; run evaluate.py on the main run, then rerun T7 for the baseline comparison")

    raw = {float(k): v for k, v in c["raw_reward"].items()}
    learn = {float(k): v for k, v in c["learn_reward"].items()}
    measured = {
        "env_steps": int(sum(raw.values())), "episodes": n,
        "stacks_sampled": c["buffer"]["stacks"], "stacks_crossing_episode_without_mask": c["buffer"]["cross_without_mask"],
        "stacks_crossing_episode_after_mask": c["buffer"]["cross_after_mask"],
        "life_loss_terminals": c["life_loss_terminals"], "game_over_terminals": c["terminated"],
        "truncations": c["truncated"],
        "raw_reward_counts": {str(k): v for k, v in sorted(raw.items())},
        "clipped_reward_counts": {str(k): v for k, v in sorted(learn.items())},
        "random_policy_mean": round(rand["mean"], 1), "random_policy_std": round(rand["std"], 1),
        "random_policy_n": rand["n"], "trained_eval_mean": trained,
    }
    ok = c["buffer"]["cross_after_mask"] == 0 and c["buffer"]["stacks"] > 0
    ok_baseline = None if trained is None else trained > rand["mean"]
    print(json.dumps(measured, indent=1))

    style()
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    ks = sorted(raw)
    axes[0].bar([str(int(k)) for k in ks], [raw[k] for k in ks], color=BLUE, edgecolor="white")
    axes[0].set(yscale="log", xlabel="raw reward per step [points]", ylabel="steps (log)",
                title="raw rewards seen in training")
    kl = sorted(learn)
    axes[1].bar([str(int(k)) for k in kl], [learn[k] for k in kl], color=ORANGE, edgecolor="white")
    axes[1].set(yscale="log", xlabel="clipped reward sign(r)", title="rewards used for learning")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Pac-Man reward clipping ({measured['env_steps']} steps); masked cross-episode stacks: "
                 f"{c['buffer']['cross_without_mask']} -> {c['buffer']['cross_after_mask']}", fontsize=9)
    save(fig, res, "t7_pacman_edges.png")
    record(res, "T7", "Pac-Man edge cases: episode-boundary frame stacks, life-loss vs game-over terminals, "
           "reward clipping, random-policy baseline",
           "0 cross-episode stacks after masking; trained eval mean > random mean", measured,
           bool(ok and ok_baseline) if ok_baseline is not None else (None if ok else False))


if __name__ == "__main__":
    main()
