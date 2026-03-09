"""
Bayesian Model Averaging (BMA) model implemented in PyTorch + Pyro.

The model treats the GFS seamless forecast as a prior and ERA5 reanalysis
as observations, learning to correct systematic GFS bias via SVI.

Architecture:
  - bias_net:  maps (gfs, lat, lon) → per-variable bias correction
  - noise_net: maps (gfs, lat, lon) → heteroscedastic noise scale
  - Guide uses amortized variational inference (recognition network),
    not stored global variational params, to avoid batch-size mismatch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class BMAModel(PyroModule):
    def __init__(self, n_features: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        inp = n_features + 2  # gfs features + lat + lon

        self.bias_net = PyroModule[nn.Sequential](
            nn.Linear(inp, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

        self.noise_net = PyroModule[nn.Sequential](
            nn.Linear(inp, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_features),
            nn.Softplus(),
        )

    def _encode(self, gfs: torch.Tensor, spatial: torch.Tensor):
        x = torch.cat([gfs, spatial], dim=-1)
        mu    = gfs + self.bias_net(x)
        sigma = self.noise_net(x) + 1e-6
        return mu, sigma

    def model(
        self,
        gfs: torch.Tensor,
        spatial: torch.Tensor,
        obs: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        batch = gfs.shape[0]
        mu, sigma = self._encode(gfs, spatial)
        with pyro.plate("data", batch):
            return pyro.sample(
                "theta",
                dist.Normal(mu, sigma).to_event(1),
                obs=obs,
            )

    def guide(
        self,
        gfs: torch.Tensor,
        spatial: torch.Tensor,
        obs: "torch.Tensor | None" = None,
    ):
        batch = gfs.shape[0]
        mu, sigma = self._encode(gfs, spatial)
        with pyro.plate("data", batch):
            pyro.sample("theta", dist.Normal(mu, sigma).to_event(1))

    def predict(
        self,
        gfs: torch.Tensor,
        spatial: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Returns (posterior_mean, posterior_std) — exported to on-device."""
        mu, sigma = self._encode(gfs, spatial)
        return mu, sigma


def build_svi(model: BMAModel, lr: float = 1e-3) -> SVI:
    optimizer = Adam({"lr": lr})
    return SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
