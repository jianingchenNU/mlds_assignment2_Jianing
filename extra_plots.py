import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch

from dqn.utils import load_ckpt, load_policy, make_eval_env
from plots import BLUE, INK, MUTED, ORANGE, SEQ, save, style

PAC_RGB = (210, 164, 74)
MAZE_ROWS = 172  # rows below this are the score/lives bar


def early_ckpt(cfg):
    return f"ep{int(cfg['snapshot_eps'].split(',')[0])}.pt"


def eval_ckpt(run):
    # The final panel shows the checkpoint evaluate.py was run on, so figures and eval numbers match.
    path = Path(run) / "eval.json"
    assert path.exists(), f"no {path}; run evaluate.py first or pass --best"
    return json.loads(path.read_text())["checkpoint"]


def ckpt_label(run, ck):
    return f"{ck}, episode {load_ckpt(Path(run) / ck, 'cpu')['episode']}"


def policy_map(run, early, best, device, out):
    th = np.linspace(-0.21, 0.21, 201)
    thd = np.linspace(-2, 2, 201)
    TH, THD = np.meshgrid(th, thd)
    s = np.stack([np.zeros_like(TH), np.zeros_like(TH), TH, THD], -1).reshape(-1, 4).astype(np.float32)

    maps = []
    for ck in (early, best):
        agent, cfg = load_policy(run, ck, device)
        with torch.no_grad():
            q = agent.q(torch.as_tensor(s, device=agent.device)).cpu().numpy()
        maps.append((ck, q.max(1).reshape(TH.shape), q.argmax(1).reshape(TH.shape)))
        # Pole leaning right with no angular velocity should push right (action 1).
        lean = (np.abs(thd) < 0.05)[:, None] & (np.abs(th) > 0.05)[None, :]
        right = (q.argmax(1).reshape(TH.shape) == 1)
        agree = float((right == (TH > 0))[lean].mean())
        print(f"policy map {ck}: pushes toward the lean in {agree:.0%} of states with |theta|>0.05, theta_dot~0")

    vmin = min(m[1].min() for m in maps)
    vmax = max(m[1].max() for m in maps)
    fig, axes = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    ext = [th[0], th[-1], thd[0], thd[-1]]
    for ax, (ck, v, act) in zip(axes, maps):
        im = ax.imshow(v, origin="lower", extent=ext, aspect="auto", cmap=SEQ, norm=Normalize(vmin, vmax))
        ax.contourf(TH, THD, act, levels=[0.5, 1.5], colors="none", hatches=["////"])
        cs = ax.contour(TH, THD, act, levels=[0.5], colors="white", linewidths=1.5)
        cs.set(path_effects=[pe.Stroke(linewidth=3, foreground=INK), pe.Normal()])
        ax.set(xlabel=r"pole angle $\theta$ [rad]", title=ckpt_label(run, ck))
        ax.grid(False)
    axes[0].set_ylabel(r"angular velocity $\dot\theta$ [rad/s]")
    fig.colorbar(im, ax=axes, label="V(s) = max$_a$ Q(s,a)", shrink=0.9)
    axes[1].legend(handles=[Patch(facecolor="none", edgecolor=MUTED, hatch="////", label="push right"),
                            Patch(facecolor="none", edgecolor=MUTED, label="push left")],
                   loc="lower right", frameon=True, framealpha=0.85)
    fig.suptitle(r"CartPole greedy policy and value at $x = \dot x = 0$", fontsize=10)
    save(fig, out, "cartpole_policy_map.png")


def visits(run, ck, episodes, device):
    agent, cfg = load_policy(run, ck, device)
    env = make_eval_env(cfg)
    hist = np.zeros((MAZE_ROWS, 160))
    scores, n_steps, n_found = [], 0, 0
    maze = None
    for i in range(episodes):
        agent.rng = np.random.default_rng(1000 + i)
        obs, info = env.reset(seed=1000 + i)
        if maze is None:
            maze = info["rgb"][:MAZE_ROWS].mean(-1)
        score = 0.0
        while True:
            act, _ = agent.act(obs, 0.01)
            obs, rew, term, trunc, info = env.step(act)
            score += rew
            n_steps += 1
            ys, xs = np.nonzero((info["rgb"][:MAZE_ROWS] == PAC_RGB).all(-1))
            if len(ys):
                n_found += 1
                hist[int(ys.mean()), int(xs.mean())] += 1
            if term or trunc:
                break
        scores.append(score)
    print(f"heatmap {ck}: {episodes} episodes, mean score {np.mean(scores):.0f}, "
          f"located in {n_found}/{n_steps} frames")
    return hist, maze, np.mean(scores)


def visit_heatmap(run, early, best, episodes, device, out):
    res = [(ck, *visits(run, ck, episodes, device)) for ck in (early, best)]
    # Coarsen to 4x4-pixel cells so single visits are visible.
    coarse = [h.reshape(MAZE_ROWS // 4, 4, 40, 4).sum((1, 3)) for _, h, _, _ in res]
    vmax = max(c.max() for c in coarse)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    for ax, (ck, _, maze, score), c in zip(axes, res, coarse):
        ax.imshow(maze, cmap="gray", alpha=0.35, extent=[0, 160, MAZE_ROWS, 0])
        im = ax.imshow(np.ma.masked_equal(c, 0), cmap=SEQ, norm=LogNorm(1, max(vmax, 2)),
                       extent=[0, 160, MAZE_ROWS, 0], alpha=0.9)
        cov = (c > 0).mean()
        ax.set_title(f"{ckpt_label(run, ck)}\nscore {score:.0f}, {cov:.0%} of cells", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="steps spent in cell (log)", shrink=0.9)
    fig.suptitle(f"Where Ms. Pac-Man goes ({episodes} games each, ε = 0.01)", fontsize=10)
    save(fig, out, "pacman_visit_heatmap.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="both games")
    p.add_argument("--which", choices=["cartpole", "pacman"])
    p.add_argument("--cartpole_run", default="runs/cartpole_main")
    p.add_argument("--pacman_run", default="runs/pacman_main")
    p.add_argument("--early", default=None, help="early checkpoint; default ep{first snapshot_eps}.pt")
    p.add_argument("--best", default=None, help="final checkpoint; default the one in the run's eval.json")
    p.add_argument("--episodes", type=int, default=10, help="Pac-Man games per checkpoint for the heatmap")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    assert a.all or a.which, "pass --all or --which"
    style()
    out = Path(a.out)
    for game, run in (("cartpole", a.cartpole_run), ("pacman", a.pacman_run)):
        if not (a.all or a.which == game):
            continue
        best = a.best or eval_ckpt(run)
        _, cfg = load_policy(run, best, a.device)
        early = a.early or early_ckpt(cfg)
        if game == "cartpole":
            policy_map(run, early, best, a.device, out)
        else:
            visit_heatmap(run, early, best, a.episodes, a.device, out)


if __name__ == "__main__":
    main()
