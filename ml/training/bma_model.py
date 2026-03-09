"""
Bayesian Model Averaging (BMA) model implemented in PyTorch + Pyro.

The model treats the NOAA GFS forecast as a prior and METAR observations
as evidence, computing the posterior P(θ|D) via variational inference.

Architecture:
  - Prior: Gaussian centered on GFS forecast values with learned bias correction
  - Likelihood: Gaussian with learned heteroscedastic noise from ERA5 residuals
  - Posterior: Approximated via Stochastic Variational Inference (SVI)
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class BMAModel(PyroModule):
    """
    Bayesian Model Averaging network for hyper-local weather correction.

    Input features:
        gfs_forecast  (B, F)  — GFS grid values for F variables
        station_obs   (B, F)  — METAR station observations
        spatial_embed (B, 2)  — normalized [lat, lon]

    Output:
        posterior_mean  (B, F)
        posterior_std   (B, F)
    """

    def __init__(self, n_features: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.n_features = n_features

        # Bias correction network: learns systematic GFS error from ERA5
        self.bias_net = PyroModule[nn.Sequential](
            nn.Linear(n_features + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

        # Heteroscedastic noise head
        self.noise_net = PyroModule[nn.Sequential](
            nn.Linear(n_features + 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_features),
            nn.Softplus(),
        )

        # Bayesian weight prior for mixing coefficients
        self.mixing_weights = PyroSample(
            dist.Dirichlet(torch.ones(n_features))
        )

    def model(
        self,
        gfs_forecast: torch.Tensor,
        spatial_embed: torch.Tensor,
        observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pyro probabilistic model (generative)."""
        batch_size = gfs_forecast.shape[0]
        x = torch.cat([gfs_forecast, spatial_embed], dim=-1)

        bias = self.bias_net(x)
        noise_scale = self.noise_net(x)

        # Prior: GFS forecast corrected by learned bias
        prior_mean = gfs_forecast + bias

        # Sample from prior distribution
        with pyro.plate("data", batch_size):
            posterior = pyro.sample(
                "theta",
                dist.Normal(prior_mean, noise_scale).to_event(1),
                obs=observations,
            )
        return posterior

    def guide(
        self,
        gfs_forecast: torch.Tensor,
        spatial_embed: torch.Tensor,
        observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Variational guide (approximate posterior)."""
        batch_size = gfs_forecast.shape[0]
        x = torch.cat([gfs_forecast, spatial_embed], dim=-1)

        q_mean = pyro.param(
            "q_mean",
            gfs_forecast + self.bias_net(x).detach(),
        )
        q_std = pyro.param(
            "q_std",
            self.noise_net(x).detach(),
            constraint=dist.constraints.positive,
        )

        with pyro.plate("data", batch_size):
            pyro.sample("theta", dist.Normal(q_mean, q_std).to_event(1))

    def predict(
        self,
        gfs_forecast: torch.Tensor,
        spatial_embed: torch.Tensor,
        n_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo posterior predictive: returns (mean, std) over n_samples.
        This is the method exported to ExecuTorch for on-device inference.
        """
        samples = []
        for _ in range(n_samples):
            x = torch.cat([gfs_forecast, spatial_embed], dim=-1)
            bias = self.bias_net(x)
            noise = self.noise_net(x)
            sample = gfs_forecast + bias + torch.randn_like(noise) * noise
            samples.append(sample)

        stacked = torch.stack(samples, dim=0)
        return stacked.mean(dim=0), stacked.std(dim=0)


def build_svi(model: BMAModel, lr: float = 1e-3) -> SVI:
    optimizer = Adam({"lr": lr})
    return SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
