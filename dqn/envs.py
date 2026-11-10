import gymnasium as gym
import numpy as np

# Assignment-provided preprocessing, kept verbatim.
mspacman_color = 210 + 164 + 74


def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.sum(axis=2)
    img[img == mspacman_color] = 0
    img = (img // 3 - 128).astype(np.int8)
    return img.reshape(88, 80, 1)


def make_cartpole(max_steps=500):
    # max_steps < 500 is only for the truncation test (T3).
    return gym.make("CartPole-v1", max_episode_steps=max_steps)


class PacmanEnv(gym.Wrapper):
    """Preprocessed 4-frame stack. Returns the raw reward; clipping is done by the learner.

    info["frame"] is the newest preprocessed (88, 80) frame, for the frame buffer.
    info["life_loss"] is True on the step a life is lost. The game is not reset on life loss,
    and the frame stack is only zero-filled on reset(), never on life loss.
    """

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.stack = np.zeros((k, 88, 80), np.int8)
        self.lives = 0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        frame = preprocess_observation(obs)[:, :, 0]
        self.stack[:] = 0
        self.stack[-1] = frame
        self.lives = info["lives"]
        info["frame"] = frame
        info["rgb"] = obs
        return self.stack.copy(), info

    def step(self, act):
        obs, rew, term, trunc, info = self.env.step(act)
        frame = preprocess_observation(obs)[:, :, 0]
        self.stack[:-1] = self.stack[1:]
        self.stack[-1] = frame
        info["life_loss"] = info["lives"] < self.lives
        self.lives = info["lives"]
        info["frame"] = frame
        info["rgb"] = obs
        return self.stack.copy(), rew, term, trunc, info


def make_pacman(max_steps=None):
    import ale_py
    gym.register_envs(ale_py)
    # v5 defaults: frameskip 4, sticky actions 0.25, minimal action set, 108k-frame cap (27k steps).
    env = gym.make("ALE/MsPacman-v5", max_episode_steps=max_steps)
    assert env.action_space.n == 9, env.action_space.n
    return PacmanEnv(env)
