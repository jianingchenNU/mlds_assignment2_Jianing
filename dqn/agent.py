import copy

import numpy as np
import torch
import torch.nn.functional as F


class DQN:
    def __init__(self, net, n_act, gamma, lr, adam_eps=1e-8, double=True, grad_clip=10.0,
                 device="cpu", seed=0):
        self.device = torch.device(device)
        self.q = net.to(self.device)
        self.q_targ = copy.deepcopy(self.q)
        for p in self.q_targ.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr, eps=adam_eps)
        self.n_act = n_act
        self.gamma = gamma
        self.double = double
        self.grad_clip = grad_clip
        self.rng = np.random.default_rng(seed)
        self.n_updates = 0

    def act(self, obs, eps):
        """Epsilon-greedy action. Also returns max_a Q(obs, a) for logging, from the same forward pass."""
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            q = self.q(x)[0]
        maxq = q.max().item()
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_act)), maxq
        return int(q.argmax().item()), maxq

    def to_tensors(self, b):
        t = lambda x, dtype=None: torch.as_tensor(x, device=self.device, dtype=dtype)
        return (t(b["obs"]), t(b["act"], torch.int64), t(b["rew"], torch.float32),
                t(b["next_obs"]), t(b["done"], torch.float32))

    @torch.no_grad()
    def targets(self, next_obs, rew, done):
        q_next = self.q_targ(next_obs)
        if self.double:
            a_star = self.q(next_obs).argmax(1, keepdim=True)
            v_next = q_next.gather(1, a_star).squeeze(1)
        else:
            v_next = q_next.max(1).values
        # done is `terminated` only; truncated transitions are bootstrapped.
        return rew + self.gamma * (1.0 - done) * v_next

    def update(self, batch):
        obs, act, rew, next_obs, done = self.to_tensors(batch)
        y = self.targets(next_obs, rew, done)
        q = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        loss_val = loss.item()
        if not np.isfinite(loss_val):
            raise RuntimeError(f"non-finite loss {loss_val} at update {self.n_updates}; stopping")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        self.n_updates += 1
        return loss_val

    def sync(self):
        self.q_targ.load_state_dict(self.q.state_dict())

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def state_dict(self):
        return {"q": self.q.state_dict(), "q_targ": self.q_targ.state_dict(),
                "opt": self.opt.state_dict(), "n_updates": self.n_updates,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, d):
        self.q.load_state_dict(d["q"])
        self.q_targ.load_state_dict(d["q_targ"])
        self.opt.load_state_dict(d["opt"])
        self.n_updates = d["n_updates"]
        self.rng.bit_generator.state = d["rng"]
