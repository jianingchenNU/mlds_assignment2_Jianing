"""T6: trace the first 10 learning updates on CartPole right after warmup, then check determinism.

Each row is one env step taken right after warmup, followed by one update. For that step's transition it shows
the state, action, epsilon, reward, Q(s,.) and the Double-DQN target, with the pieces needed to recompute the
target by hand: a* = argmax_a Q_online(s', a), y = r + gamma * (1 - done) * Q_target(s', a*).
TD error = y - Q(s, a). Loss is the Huber loss of the update's random minibatch (not just this transition).
Determinism: the whole procedure is run twice with the same seed, and a greedy (eps = 0) episode of the
resulting network is rolled out twice; both pairs must match exactly.
"""
import argparse

from common import dirs, record

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dqn.agent import DQN  # noqa: E402
from dqn.buffer import VectorBuffer  # noqa: E402
from dqn.envs import make_cartpole  # noqa: E402
from dqn.nets import MLP  # noqa: E402
from dqn.schedule import piecewise  # noqa: E402
from dqn.utils import seed_all  # noqa: E402
from train import DEFAULTS  # noqa: E402

D = DEFAULTS["cartpole"]


def rnd(x):
    return [round(float(v), 4) for v in x]


def trace(seed, warmup, n_updates):
    seed_all(seed)
    env = make_cartpole()
    agent = DQN(MLP(4, 2, D["hidden"]), 2, D["gamma"], D["lr"], double=True, device="cpu", seed=seed)
    buf = VectorBuffer(D["buffer"], 4, seed=seed)
    pts = [(0, D["eps_start"]), (D["eps_decay"], D["eps_mid"])]
    obs, _ = env.reset(seed=seed)
    rows, t, ep = [], 0, 0
    while len(rows) < n_updates:
        eps = piecewise(t, pts)
        a, _ = agent.act(obs, eps)
        nobs, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, nobs, float(term))
        t += 1
        if len(buf) >= warmup:
            with torch.no_grad():
                s, s2 = torch.as_tensor(obs[None]), torch.as_tensor(nobs[None])
                q, q2_on, q2_tg = agent.q(s)[0], agent.q(s2)[0], agent.q_targ(s2)[0]
                y = agent.targets(s2, torch.tensor([r], dtype=torch.float32), torch.tensor([float(term)]))[0]
            a_star = int(q2_on.argmax())
            loss = agent.update(buf.sample(D["batch"]))
            rows.append({"t": t, "s": rnd(obs), "a": a, "eps": round(eps, 4), "r": r, "done": int(term),
                         "Q(s,.)": rnd(q), "a*": a_star,
                         "Qtarg(s',.)": rnd(q2_tg), "y": round(float(y), 4),
                         "td": round(float(y - q[a]), 4), "loss": round(loss, 5),
                         "_check": float(y) - (r + D["gamma"] * (1 - term) * float(q2_tg[a_star]))})
        obs = nobs
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
    # Greedy rollout of the resulting network, fixed env seed.
    obs, _ = env.reset(seed=12345)
    states = [obs]
    while True:
        a, _ = agent.act(obs, 0.0)
        obs, _, term, trunc, _ = env.step(a)
        states.append(obs)
        if term or trunc:
            break
    return rows, np.array(states)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    _, res = dirs(a.quick)
    warmup = 200 if a.quick else D["warmup"]

    rows1, roll1 = trace(a.seed, warmup, 10)
    rows2, roll2 = trace(a.seed, warmup, 10)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "_check"} for r in rows]
    same_trace = strip(rows1) == strip(rows2)
    same_roll = roll1.shape == roll2.shape and bool((roll1 == roll2).all())
    max_check = max(abs(r["_check"]) for r in rows1)
    # Q, a* and Q_targ are reported rounded; recomputing y from the printed numbers must agree to rounding.
    hand = max(abs(r["y"] - (r["r"] + D["gamma"] * (1 - r["done"]) * r["Qtarg(s',.)"][r["a*"]])) for r in rows1)

    cols = ["t", "s", "a", "eps", "r", "done", "Q(s,.)", "a*", "Qtarg(s',.)", "y", "td", "loss"]
    lines = [f"# CartPole trace: first 10 updates after warmup ({warmup} transitions), seed {a.seed}", "",
             f"gamma = {D['gamma']}, Double DQN target y = r + gamma * (1 - done) * Qtarg(s', a*), "
             "a* = argmax Q_online(s'). td = y - Q(s, a). loss = Huber loss of the update's random batch of "
             f"{D['batch']}.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows1:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines += ["", f"Max |y - recomputed y| from full-precision values: {max_check:.2e}; "
                  f"from the rounded printed values: {hand:.2e}.",
              f"Same seed, run twice: trace identical = {same_trace}; greedy rollout (eps = 0, env seed 12345, "
              f"{len(roll1) - 1} steps) identical = {same_roll}."]
    res.mkdir(parents=True, exist_ok=True)
    (res / "trace_cartpole.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:7]))
    print(f"wrote {res / 'trace_cartpole.md'}")
    ok = same_trace and same_roll and max_check < 1e-5 and hand < 1e-3
    record(res, "T6", "CartPole first 10 updates after warmup: hand-checkable targets, and determinism with a fixed seed",
           "targets recompute exactly; two runs with the same seed identical",
           {"max_target_recompute_err": max_check, "max_err_from_printed": round(hand, 6),
            "trace_identical": same_trace, "greedy_rollout_identical": same_roll,
            "greedy_rollout_steps": int(len(roll1) - 1)}, bool(ok))


if __name__ == "__main__":
    main()
