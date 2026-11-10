import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from dqn.agent import DQN
from dqn.buffer import FrameBuffer, VectorBuffer
from dqn.envs import make_cartpole, make_pacman
from dqn.nets import CNN, MLP
from dqn.schedule import piecewise
from dqn.utils import CSVLogger, device_name, load_ckpt, rollout, save_ckpt, seed_all

DEFAULTS = {
    "cartpole": dict(
        episodes=500, gamma=0.95, lr=5e-4, lr_final=None, adam_eps=1e-8, buffer=50_000, batch=64,
        warmup=1000, target_sync=500, train_every=1, eps_start=1.0, eps_mid=0.02, eps_decay=10_000,
        eps_end=0.02, eps_decay2=0, double=1, trunc_as_terminal=0, max_steps=500, hidden=128,
        grad_clip=10.0, device="cpu", ckpt_every=50, check_every=10, check_episodes=10, check_eps=0.0,
        log_every=10, max_hours=None, snapshot_eps="100",
    ),
    "pacman": dict(
        episodes=5000, gamma=0.99, lr=1e-4, lr_final=None, adam_eps=1.5e-4, buffer=300_000, batch=32,
        warmup=50_000, target_sync=10_000, train_every=4, eps_start=1.0, eps_mid=0.1,
        eps_decay=1_000_000, eps_end=0.01, eps_decay2=1_000_000, double=1, trunc_as_terminal=0,
        max_steps=None, hidden=None, grad_clip=10.0,
        device="cuda" if torch.cuda.is_available() else "cpu", ckpt_every=100, check_every=100,
        check_episodes=5, check_eps=0.01, log_every=10, max_hours=None, snapshot_eps="200",
    ),
}
# Flags that may change on --resume; anything else must match the saved config.
RESUME_OVERRIDES = {"episodes", "max_hours", "device", "log_every"}
FIELDS = ["episode", "env_steps", "reward", "length", "mean_maxq", "max_maxq", "eps", "loss", "lr", "wall_s"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["cartpole", "pacman"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--resume", action="store_true")
    for k, v in DEFAULTS["cartpole"].items():
        typ = int if k in {"episodes", "buffer", "batch", "warmup", "target_sync", "train_every", "eps_decay",
                           "eps_decay2", "double", "trunc_as_terminal", "max_steps", "hidden", "ckpt_every",
                           "check_every", "check_episodes", "log_every"} else float
        if k in {"device", "snapshot_eps"}:
            typ = str
        p.add_argument(f"--{k}", type=typ, default=None)
    a = p.parse_args(argv)
    passed = {k for k, v in vars(a).items() if v is not None and k in DEFAULTS[a.env]}
    for k, v in DEFAULTS[a.env].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a, passed


def eps_points(a):
    pts = [(0, a.eps_start), (a.eps_decay, a.eps_mid)]
    if a.eps_decay2 > 0:
        pts.append((a.eps_decay + a.eps_decay2, a.eps_end))
    return pts


def build(a):
    if a.env == "cartpole":
        env = make_cartpole(a.max_steps)
        check_env = make_cartpole()
        obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
        net = MLP(obs_dim, n_act, a.hidden)
        buf = VectorBuffer(a.buffer, obs_dim, seed=a.seed)
    else:
        env = make_pacman(a.max_steps)
        check_env = make_pacman(a.max_steps)
        n_act = env.action_space.n
        net = CNN(n_act)
        buf = FrameBuffer(a.buffer, (88, 80), k=4, seed=a.seed)
    agent = DQN(net, n_act, a.gamma, a.lr, a.adam_eps, bool(a.double), a.grad_clip, a.device, a.seed)
    return env, check_env, buf, agent


def main(argv=None):
    a, passed = parse_args(argv)
    run = Path(a.runs_dir) / a.name
    cfg_path = run / "config.json"

    if a.resume:
        saved = json.loads(cfg_path.read_text())
        assert saved["env"] == a.env, f"run {run} is {saved['env']}, not {a.env}"
        bad = {k for k in passed - RESUME_OVERRIDES if saved[k] != getattr(a, k)}
        if bad:
            sys.exit(f"--resume: flags {sorted(bad)} differ from saved config; only {sorted(RESUME_OVERRIDES)} may change")
        for k in DEFAULTS[a.env]:
            if k not in passed or k not in RESUME_OVERRIDES:
                setattr(a, k, saved[k])
        a.seed = saved["seed"]
    elif (run / "train.csv").exists():
        sys.exit(f"{run} already has a train.csv; use --resume or a new --name")

    assert a.warmup >= a.batch, "warmup must be >= batch so the first sample is valid"
    assert a.buffer >= a.warmup, "buffer must hold at least warmup transitions"

    run.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    dev_name = device_name(a.device)
    print(f"device: {dev_name}")
    cfg = dict(vars(a), device_name=dev_name, torch=torch.__version__)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    env, check_env, buf, agent = build(a)
    eps_pts = eps_points(a)
    # Model-only snapshots at fixed episodes; last.pt is overwritten, these are not (early policy for extra plots).
    snaps = {int(x) for x in a.snapshot_eps.split(",") if x.strip()}

    ep0, env_steps, best, wall0 = 0, 0, -np.inf, 0.0
    if a.resume:
        ck = load_ckpt(run / "last.pt", a.device)
        agent.load_state_dict(ck["agent"])
        ep0, env_steps, best, wall0 = ck["episode"], ck["env_steps"], ck["best"], ck["wall_s"]
        print(f"resumed at episode {ep0}, env_steps {env_steps}; refilling {a.warmup} warmup steps")
    log = CSVLogger(run / "train.csv", FIELDS, keep_upto=ep0 if a.resume else None)
    checks = CSVLogger(run / "checks.csv", ["episode", "env_steps", "mean", "std", "best"],
                       keep_upto=ep0 if a.resume else None)
    recent = deque((float(r["reward"]) for r in log.rows), maxlen=100)

    # Edge-case counters (cumulative across resumes), written to counters.json with each checkpoint.
    pac = a.env == "pacman"
    cnt_path = run / "counters.json"
    cnt = {"terminated": 0, "truncated": 0, "life_loss_terminals": 0, "raw_reward": {}, "learn_reward": {},
           "buffer": {"stacks": 0, "cross_without_mask": 0, "cross_after_mask": 0}}
    if a.resume and cnt_path.exists():
        cnt = json.loads(cnt_path.read_text())
    buf_base = dict(cnt["buffer"])

    def write_counters():
        if pac:
            cnt["buffer"] = {"stacks": buf_base["stacks"] + buf.n_stacks,
                             "cross_without_mask": buf_base["cross_without_mask"] + buf.n_cross_raw,
                             "cross_after_mask": buf_base["cross_after_mask"] + buf.n_cross_masked}
        cnt_path.write_text(json.dumps(cnt, indent=2))

    def ckpt(ep):
        return {"agent": agent.state_dict(), "episode": ep, "env_steps": env_steps, "best": best,
                "wall_s": wall0 + time.time() - t0}

    t0 = time.time()
    steps_at_log, t_at_log = env_steps, t0
    n_done = ep0
    for ep in range(ep0, a.episodes):
        lr = a.lr if a.lr_final is None else a.lr + (a.lr_final - a.lr) * ep / a.episodes
        agent.set_lr(lr)
        obs, info = env.reset(seed=100_000 * (a.seed + 1) + ep)
        if pac:
            buf.add_first(info["frame"])
        ep_rew, ep_len, qs, losses = 0.0, 0, [], []
        while True:
            eps = piecewise(env_steps, eps_pts)
            act, maxq = agent.act(obs, eps)
            qs.append(maxq)
            next_obs, rew, term, trunc, info = env.step(act)
            # Only true termination stops bootstrapping; time-limit truncation does not (unless T3 asks).
            # Pac-Man: a lost life also ends bootstrapping, but the game and the frame stack go on.
            life_loss = info.get("life_loss", False)
            done = term or life_loss or (trunc and a.trunc_as_terminal)
            r_learn = float(np.sign(rew)) if pac else rew
            if pac:
                buf.add(act, r_learn, float(done), info["frame"])
            else:
                buf.add(obs, act, r_learn, next_obs, float(done))
            cnt["life_loss_terminals"] += int(life_loss and not term)
            cnt["raw_reward"][str(rew)] = cnt["raw_reward"].get(str(rew), 0) + 1
            cnt["learn_reward"][str(r_learn)] = cnt["learn_reward"].get(str(r_learn), 0) + 1
            obs = next_obs
            ep_rew += rew
            ep_len += 1
            env_steps += 1
            if len(buf) >= a.warmup and env_steps % a.train_every == 0:
                losses.append(agent.update(buf.sample(a.batch)))
            if env_steps % a.target_sync == 0:
                agent.sync()
            if term or trunc:
                break

        n_done = ep + 1
        cnt["terminated"] += int(term)
        cnt["truncated"] += int(trunc and not term)
        recent.append(ep_rew)
        log.log({"episode": n_done, "env_steps": env_steps, "reward": ep_rew, "length": ep_len,
                 "mean_maxq": np.mean(qs), "max_maxq": np.max(qs), "eps": eps,
                 "loss": np.mean(losses) if losses else np.nan, "lr": lr,
                 "wall_s": round(wall0 + time.time() - t0, 2)})

        if n_done % a.check_every == 0:
            scores = [rollout(agent, check_env, a.check_eps, 900_000 + i)[0] for i in range(a.check_episodes)]
            m = float(np.mean(scores))
            if m > best:
                best = m
                save_ckpt(run / "best.pt", ckpt(n_done))
            checks.log({"episode": n_done, "env_steps": env_steps, "mean": m, "std": float(np.std(scores)),
                        "best": best})

        if n_done in snaps:
            save_ckpt(run / f"ep{n_done}.pt", {"q": agent.q.state_dict(), "episode": n_done, "env_steps": env_steps})

        if n_done % a.ckpt_every == 0:
            save_ckpt(run / "last.pt", ckpt(n_done))
            write_counters()

        if n_done % a.log_every == 0:
            now = time.time()
            sps = (env_steps - steps_at_log) / max(now - t_at_log, 1e-9)
            steps_at_log, t_at_log = env_steps, now
            print(f"ep {n_done} steps {env_steps} rew {ep_rew:.1f} ma100 {np.mean(recent):.1f} "
                  f"maxq {np.mean(qs):.2f} eps {eps:.3f} loss {log_loss(losses)} lr {lr:.2e} sps {sps:.0f}")

        if a.max_hours is not None and wall0 + time.time() - t0 >= a.max_hours * 3600:
            print(f"max_hours reached at episode {n_done}")
            break

    save_ckpt(run / "last.pt", ckpt(n_done))
    write_counters()
    log.close()
    checks.close()
    print(f"done: {n_done} episodes, {env_steps} env steps, best check {best:.1f}, run dir {run}")


def log_loss(losses):
    return f"{np.mean(losses):.4f}" if losses else "-"


if __name__ == "__main__":
    main()
