import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dqn.utils import load_policy, make_eval_env


def run_eval(run, n=None, random=False, ckpt="best.pt", eps=None, device="cpu", trace_episodes=20, seed0=1000):
    """Roll out a checkpoint (or a uniform-random policy) and write eval*.csv / eval*.json into the run dir."""
    run = Path(run)
    agent, cfg = load_policy(run, ckpt, device)
    pac = cfg["env"] == "pacman"
    n = n or (100 if random else 500)
    eps = 1.0 if random else (eps if eps is not None else (0.01 if pac else 0.0))
    env = make_eval_env(cfg)
    tag = "eval_random" if random else "eval"

    rows, trace = [], []
    for i in range(n):
        seed = seed0 + i
        agent.rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        score, t = 0.0, 0
        while True:
            act, maxq = agent.act(obs, eps)
            obs, rew, term, trunc, info = env.step(act)
            life_loss = info.get("life_loss", False)
            if not random and i < trace_episodes:
                # Same reward processing and terminal rule as training, for T5.
                trace.append({"episode": i, "t": t, "maxq": maxq, "r_learn": float(np.sign(rew)) if pac else rew,
                              "done_learn": int(term or life_loss), "term": int(term), "trunc": int(trunc)})
            score += rew
            t += 1
            if term or trunc:
                break
        rows.append({"episode": i, "seed": seed, "reward": score, "length": t, "truncated": int(trunc and not term)})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{n} episodes, running mean {np.mean([r['reward'] for r in rows]):.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(run / f"{tag}.csv", index=False)
    if trace:
        pd.DataFrame(trace).to_csv(run / "eval_trace.csv", index=False)
    summary = {"mean": float(df.reward.mean()), "std": float(df.reward.std(ddof=0)),
               "min": float(df.reward.min()), "max": float(df.reward.max()), "n": n,
               "checkpoint": "random" if random else ckpt, "eps": eps,
               "truncated": int(df.truncated.sum()), "mean_length": float(df.length.mean())}
    (run / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"{tag} {run.name}: mean {summary['mean']:.1f} std {summary['std']:.1f} "
          f"min {summary['min']:.0f} max {summary['max']:.0f} n {n} eps {eps} truncated {summary['truncated']}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", type=int, default=None, help="default 500, or 100 with --random")
    p.add_argument("--random", action="store_true", help="uniform-random policy baseline")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--eps", type=float, default=None, help="default 0 for cartpole, 0.01 for pacman")
    p.add_argument("--device", default="cpu")
    p.add_argument("--trace_episodes", type=int, default=20)
    p.add_argument("--seed0", type=int, default=1000)
    a = p.parse_args()
    run_eval(a.run, a.episodes, a.random, a.ckpt, a.eps, a.device, a.trace_episodes, a.seed0)


if __name__ == "__main__":
    main()
