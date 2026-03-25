"""
Large BMA teacher model for knowledge distillation.

Same architecture as BMAModel but with much more capacity:
  - bias_net:  12 → 256 → 256 → 128 → 6  (~100K params)
  - noise_net: 12 → 128 →  64 →   6      (~10K params)

This model trains offline and is never deployed to mobile. Its predictions
are used as soft targets to train the smaller student model (BMAModel).
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class BMATeacher(PyroModule):
    def __init__(self, n_features: int = 6, n_temporal: int = 4):
        super().__init__()
        self.n_features = n_features
        self.n_temporal = n_temporal
        inp = n_features + 2 + n_temporal  # 12

        self.bias_net = PyroModule[nn.Sequential](
            nn.Linear(inp, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_features),
        )

        self.noise_net = PyroModule[nn.Sequential](
            nn.Linear(inp, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
            nn.Softplus(),
        )

    def _encode(self, gfs, spatial, temporal=None):
        parts = [gfs, spatial]
        if temporal is not None:
            parts.append(temporal)
        else:
            parts.append(torch.zeros(gfs.shape[0], self.n_temporal, device=gfs.device))
        x = torch.cat(parts, dim=-1)
        mu = gfs + self.bias_net(x)
        sigma = self.noise_net(x) + 1e-6
        return mu, sigma

    def model(self, gfs, spatial, obs=None, temporal=None):
        batch = gfs.shape[0]
        mu, sigma = self._encode(gfs, spatial, temporal)
        with pyro.plate("data", batch):
            return pyro.sample("theta", dist.Normal(mu, sigma).to_event(1), obs=obs)

    def guide(self, gfs, spatial, obs=None, temporal=None):
        batch = gfs.shape[0]
        mu, sigma = self._encode(gfs, spatial, temporal)
        with pyro.plate("data", batch):
            pyro.sample("theta", dist.Normal(mu, sigma).to_event(1))

    def predict(self, gfs, spatial, temporal=None):
        mu, sigma = self._encode(gfs, spatial, temporal)
        return mu, sigma


def build_teacher_svi(model: BMATeacher, lr: float = 1e-3) -> SVI:
    optimizer = Adam({"lr": lr})
    return SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
