import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from dqn.agent import DQN
from dqn.envs import make_cartpole, make_pacman
from dqn.nets import CNN, MLP


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_name(device):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


class CSVLogger:
    """Per-row flushed CSV. On resume, rows past `keep_upto` (written after the checkpoint) are dropped."""

    def __init__(self, path, fields, key="episode", keep_upto=None):
        self.path = Path(path)
        self.fields = fields
        rows = []
        if keep_upto is not None and self.path.exists():
            with open(self.path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r[key]) <= keep_upto]
        self.f = open(self.path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        self.w.writeheader()
        self.w.writerows(rows)
        self.f.flush()
        self.rows = rows

    def log(self, row):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def save_ckpt(path, state):
    # Write then rename so an interruption mid-save never leaves a corrupt checkpoint.
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_ckpt(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def rollout(agent, env, eps, seed):
    """One episode with a fixed epsilon. Returns (score, length)."""
    obs, _ = env.reset(seed=seed)
    score, n = 0.0, 0
    while True:
        a, _ = agent.act(obs, eps)
        obs, r, term, trunc, _ = env.step(a)
        score += r
        n += 1
        if term or trunc:
            return score, n


def load_policy(run, ckpt="best.pt", device="cpu", seed=0):
    """Rebuild the agent from a run's config and load Q weights from any checkpoint (last/best/ep{N})."""
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    if cfg["env"] == "cartpole":
        net, n_act = MLP(4, 2, cfg["hidden"]), 2
    else:
        net, n_act = CNN(9), 9
    agent = DQN(net, n_act, cfg["gamma"], cfg["lr"], device=device, seed=seed)
    d = load_ckpt(run / ckpt, device)
    agent.q.load_state_dict(d["agent"]["q"] if "agent" in d else d["q"])
    agent.q.eval()
    return agent, cfg


def make_eval_env(cfg):
    # CartPole is always evaluated on the standard 500-step limit, even for T3's shortened runs.
    return make_cartpole() if cfg["env"] == "cartpole" else make_pacman(cfg["max_steps"])
