"""T2: run the same DQN agent on a 10-state deterministic chain where Q* is known exactly.

States 0..9, one-hot. Action 0 = left (clamped at 0), 1 = right. Stepping into state 9 gives +1 and terminates.
Episodes start in a uniformly random non-terminal state and are truncated after 30 steps (bootstrapped).
Behaviour policy is uniform random so every (s, a) is covered. Control run: all rewards set to 0, so Q* = 0.
"""
import argparse

from common import dirs, record

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dqn.agent import DQN  # noqa: E402
from dqn.buffer import VectorBuffer  # noqa: E402
from dqn.nets import MLP  # noqa: E402
from dqn.utils import seed_all  # noqa: E402
from plots import BLUE, MUTED, ORANGE, save, style  # noqa: E402

N, GOAL, GAMMA, T_MAX = 10, 9, 0.95, 30


def step(s, a, reward_scale):
    s2 = max(s - 1, 0) if a == 0 else s + 1
    done = s2 == GOAL
    return s2, (reward_scale if done else 0.0), done


def q_star(reward_scale):
    q = np.zeros((GOAL, 2))
    for _ in range(1000):
        v = np.append(q.max(1), 0.0)  # V(goal) = 0, terminal
        new = np.zeros_like(q)
        for s in range(GOAL):
            for a in range(2):
                s2, r, done = step(s, a, reward_scale)
                new[s, a] = r + GAMMA * (0.0 if done else v[s2])
        if np.abs(new - q).max() < 1e-12:
            break
        q = new
    return q


def onehot(s):
    x = np.zeros(N, np.float32)
    x[s] = 1.0
    return x


def learn(reward_scale, steps, seed):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    agent = DQN(MLP(N, 2, 64), 2, GAMMA, lr=1e-3, double=True, device="cpu", seed=seed)
    buf = VectorBuffer(10_000, N, seed=seed)
    s, t = int(rng.integers(GOAL)), 0
    for i in range(1, steps + 1):
        a = int(rng.integers(2))
        s2, r, done = step(s, a, reward_scale)
        t += 1
        buf.add(onehot(s), a, r, onehot(s2), float(done))
        s = s2
        if done or t >= T_MAX:
            s, t = int(rng.integers(GOAL)), 0
        if len(buf) >= 500:
            agent.update(buf.sample(64))
        if i % 200 == 0:
            agent.sync()
    with torch.no_grad():
        return agent.q(torch.as_tensor(np.stack([onehot(s) for s in range(GOAL)]))).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--steps", type=int, default=None, help="env steps = updates; default 20000 (quick: 1500)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    steps = a.steps or (1500 if a.quick else 20_000)
    _, res = dirs(a.quick)

    qs, ql = q_star(1.0), learn(1.0, steps, a.seed)
    q0 = learn(0.0, steps, a.seed)
    err = float(np.abs(ql - qs).max())
    zero = float(np.abs(q0).max())
    greedy_ok = bool((ql.argmax(1) == 1).all())
    print("Q* (left, right):\n", np.round(qs, 4))
    print("learned:\n", np.round(ql, 4))
    measured = {"max_abs_err": round(err, 5), "zero_reward_max_abs_q": round(zero, 5),
                "greedy_is_right_everywhere": greedy_ok, "steps": steps}

    style()
    fig, axes = plt.subplots(1, 2, sharey=False, constrained_layout=True)
    x = np.arange(GOAL)
    for act, c, name in ((1, BLUE, "right"), (0, ORANGE, "left")):
        axes[0].plot(x, qs[:, act], color=c, lw=1, ls="--")
        axes[0].plot(x, ql[:, act], color=c, lw=0, marker="o", ms=4, label=f"Q(s, {name}) learned")
    axes[0].set(xlabel="state", ylabel="Q(s,a) [discounted reward]", title=f"reward +1 at goal: max err {err:.4f}")
    axes[0].legend(title="dashed = exact Q*", fontsize=7, title_fontsize=7)
    axes[1].plot(x, q0[:, 1], color=BLUE, lw=0, marker="o", ms=4, label="right")
    axes[1].plot(x, q0[:, 0], color=ORANGE, lw=0, marker="o", ms=4, label="left")
    axes[1].axhline(0, color=MUTED, ls="--", lw=1)
    axes[1].set(xlabel="state", title=f"all rewards 0: max |Q| {zero:.4f}")
    axes[1].legend(fontsize=7)
    save(fig, res, "t2_chain_mdp.png")
    record(res, "T2", "DQN on a 10-state deterministic chain vs exact Q* from value iteration (gamma 0.95)",
           "max |Q - Q*| < 0.05; zero-reward control max |Q| ~ 0 (< 0.05)", measured,
           bool(err < 0.05 and zero < 0.05))


if __name__ == "__main__":
    main()
