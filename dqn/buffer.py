import numpy as np


class VectorBuffer:
    def __init__(self, cap, obs_dim, seed=0):
        self.cap = cap
        self.obs = np.zeros((cap, obs_dim), np.float32)
        self.next_obs = np.zeros((cap, obs_dim), np.float32)
        self.act = np.zeros(cap, np.int64)
        self.rew = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = obs, act, rew, next_obs, done
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def __len__(self):
        return self.size

    def sample(self, batch):
        assert self.size >= batch, f"buffer has {self.size} < batch {batch}"
        idx = self.rng.integers(0, self.size, batch)
        return {"obs": self.obs[idx], "act": self.act[idx], "rew": self.rew[idx],
                "next_obs": self.next_obs[idx], "done": self.done[idx]}


class FrameBuffer:
    """Stores each frame once; rebuilds k-frame stacks at sample time.

    Slot i holds frame f_i and, once the next step arrives, the transition (a_i, r_i, done_i)
    leading to f_{i+1}. The last frame of an episode (terminal or truncated) is stored but has
    no transition from it. Frames before an episode start are zero-filled in a stack.
    """

    def __init__(self, cap, frame_shape=(88, 80), k=4, seed=0):
        self.cap = cap
        self.k = k
        self.frames = np.zeros((cap, *frame_shape), np.int8)
        self.act = np.zeros(cap, np.int64)
        self.rew = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.float32)
        self.start = np.zeros(cap, bool)
        self.valid = np.zeros(cap, bool)  # transition slot i -> i+1 exists
        # Episode id per slot; used only to verify the start-flag mask (T7), not for masking.
        self.ep_id = np.full(cap, -1, np.int64)
        self.ptr = 0
        self.size = 0
        self.n_valid = 0
        self.n_eps = -1
        self.rng = np.random.default_rng(seed)
        self.n_stacks = 0
        self.n_cross_raw = 0      # stacks that would mix episodes without the mask
        self.n_cross_masked = 0   # stacks that still mix episodes after masking (should stay 0)

    def _write(self, frame, start):
        i = self.ptr
        self.frames[i] = frame
        self.start[i] = start
        self.n_valid -= int(self.valid[i])
        self.valid[i] = False
        self.ep_id[i] = self.n_eps
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def add_first(self, frame):
        self.n_eps += 1
        self._write(frame, True)

    def add(self, act, rew, done, next_frame):
        i = (self.ptr - 1) % self.cap
        assert self.ep_id[i] == self.n_eps, "add() called before add_first()"
        self.act[i], self.rew[i], self.done[i] = act, rew, done
        self.valid[i] = True
        self.n_valid += 1
        self._write(next_frame, False)

    def __len__(self):
        return self.n_valid

    def _ok(self, idx):
        # Stack window spans slots idx-k+1 .. idx+1. It must not cross the write pointer,
        # where the newest slot is followed by the oldest.
        ok = self.valid[idx].copy()
        for off in range(-self.k + 2, 2):
            ok &= (idx + off) % self.cap != self.ptr
        return ok

    def _stack(self, slots, starts, ep):
        # slots/starts: (B, k), last column is the newest frame.
        keep = np.ones(starts.shape, bool)
        for c in range(self.k - 2, -1, -1):
            keep[:, c] = keep[:, c + 1] & ~starts[:, c + 1]
        x = self.frames[slots] * keep[:, :, None, None].astype(np.int8)
        self.n_stacks += len(slots)
        self.n_cross_raw += int((ep != ep[:, -1:]).any(1).sum())
        self.n_cross_masked += int(((ep != ep[:, -1:]) & keep).any(1).sum())
        return x

    def sample(self, batch):
        assert len(self) >= batch, f"buffer has {len(self)} transitions < batch {batch}"
        idx = np.empty(0, np.int64)
        while len(idx) < batch:
            cand = self.rng.integers(0, self.size, 2 * batch)
            idx = np.concatenate([idx, cand[self._ok(cand)]])
        idx = idx[:batch]
        slots = (idx[:, None] + np.arange(-self.k + 1, 2)) % self.cap  # (B, k+1)
        starts = self.start[slots]
        ep = self.ep_id[slots]
        obs = self._stack(slots[:, :-1], starts[:, :-1], ep[:, :-1])
        next_obs = self._stack(slots[:, 1:], starts[:, 1:], ep[:, 1:])
        return {"obs": obs, "act": self.act[idx], "rew": self.rew[idx],
                "next_obs": next_obs, "done": self.done[idx]}
