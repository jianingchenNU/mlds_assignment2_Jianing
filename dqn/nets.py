import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_act),
        )

    def forward(self, x):
        return self.net(x.float())


class CNN(nn.Module):
    def __init__(self, n_act, in_ch=4, h=88, w=80):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_ch, h, w)).shape[1]
        assert n_flat == 64 * 7 * 6, n_flat
        self.head = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(), nn.Linear(512, n_act))

    def forward(self, x):
        # Frames are stored as int8 in [-128, 127]; scale to about [-1, 1].
        return self.head(self.conv(x.float() / 128.0))
