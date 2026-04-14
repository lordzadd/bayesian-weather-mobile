"""
Bayesian Calibration Layer — Source Fusion variant.

Instead of residual correction on the LSTM alone, this learns a per-variable
blending weight between the base model prediction and the GFS forecast,
plus a bias correction and calibrated uncertainty.

At each horizon the network sees both sources and context, then outputs:
  - alpha:  sigmoid blend weight per variable (0 = trust GFS, 1 = trust base model)
  - bias:   additive correction on the blended result
  - logvar: calibrated log-variance

The architecture doc's key claim: the Bayesian layer should implicitly learn
that the LSTM carries more skill at short horizons and GFS at long horizons.
A learned alpha per horizon directly tests this.
"""

import torch
import torch.nn as nn


class BayesianCalibration(nn.Module):
    def __init__(
        self,
        n_vars: int = 6,
        hidden_dim: int = 64,
        n_spatial: int = 2,
        n_temporal: int = 4,
        n_horizons: int = 5,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_vars = n_vars

        # Input: base_mu (6) + base_logvar (6) + gfs (6) + spatial (2) + temporal (4) = 24
        inp_dim = n_vars * 3 + n_spatial + n_temporal

        # Blend weight network: outputs per-variable alpha in (0, 1)
        # alpha=1 means trust base model, alpha=0 means trust GFS
        self.alpha_net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_vars),
        )

        # Bias correction on the blended result
        self.bias_net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_vars),
        )

        # Calibrated uncertainty
        self.logvar_net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_vars),
        )

    def forward(self, base_mu, base_logvar, gfs_targets, spatial, temporal):
        """
        Args:
            base_mu:     (B, H, 6) base model predicted mean
            base_logvar: (B, H, 6) base model predicted log-variance
            gfs_targets: (B, H, 6) GFS forecast at each horizon
            spatial:     (B, 2)    lat/lon
            temporal:    (B, H, 4) sin/cos hour+doy at each horizon
        Returns:
            mu:     (B, H, 6) fused & calibrated mean
            logvar: (B, H, 6) calibrated log-variance
        """
        mus, logvars = [], []
        for i in range(self.n_horizons):
            x = torch.cat([
                base_mu[:, i],
                base_logvar[:, i],
                gfs_targets[:, i],
                spatial,
                temporal[:, i],
            ], dim=-1)

            # Learned per-variable blend weight
            alpha = torch.sigmoid(self.alpha_net(x))  # (B, V), in (0,1)

            # Fuse: weighted combination of base model and GFS
            blended = alpha * base_mu[:, i] + (1 - alpha) * gfs_targets[:, i]

            # Additive bias correction
            mu = blended + self.bias_net(x)

            logvar = self.logvar_net(x)
            mus.append(mu)
            logvars.append(logvar)

        return torch.stack(mus, dim=1), torch.stack(logvars, dim=1)

    def get_blend_weights(self, base_mu, base_logvar, gfs_targets, spatial, temporal):
        """Return the learned alpha weights for analysis. alpha=1 → trust base model."""
        alphas = []
        for i in range(self.n_horizons):
            x = torch.cat([
                base_mu[:, i],
                base_logvar[:, i],
                gfs_targets[:, i],
                spatial,
                temporal[:, i],
            ], dim=-1)
            alphas.append(torch.sigmoid(self.alpha_net(x)))
        return torch.stack(alphas, dim=1)  # (B, H, V)


class PerHorizonBayCal(nn.Module):
    """
    Per-horizon Bayesian calibration — one independent fusion model per horizon.

    From the architecture doc: "Per-hour Bayesian models are used because the
    nature of prediction errors changes at each lead time. Hour 1 errors are
    dominated by observation lag. Hour 6 errors are dominated by mesoscale
    features. Each hour has a distinct error signature that a dedicated model
    can learn precisely."
    """

    def __init__(
        self,
        n_vars: int = 6,
        hidden_dim: int = 64,
        n_spatial: int = 2,
        n_temporal: int = 4,
        n_horizons: int = 5,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_vars = n_vars

        inp_dim = n_vars * 3 + n_spatial + n_temporal

        # Independent networks per horizon
        self.alpha_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inp_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_vars),
            ) for _ in range(n_horizons)
        ])
        self.bias_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inp_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_vars),
            ) for _ in range(n_horizons)
        ])
        self.logvar_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inp_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_vars),
            ) for _ in range(n_horizons)
        ])

    def forward(self, base_mu, base_logvar, gfs_targets, spatial, temporal):
        mus, logvars = [], []
        for i in range(self.n_horizons):
            x = torch.cat([
                base_mu[:, i],
                base_logvar[:, i],
                gfs_targets[:, i],
                spatial,
                temporal[:, i],
            ], dim=-1)

            alpha = torch.sigmoid(self.alpha_nets[i](x))
            blended = alpha * base_mu[:, i] + (1 - alpha) * gfs_targets[:, i]
            mu = blended + self.bias_nets[i](x)
            logvar = self.logvar_nets[i](x)
            mus.append(mu)
            logvars.append(logvar)

        return torch.stack(mus, dim=1), torch.stack(logvars, dim=1)

    def get_blend_weights(self, base_mu, base_logvar, gfs_targets, spatial, temporal):
        alphas = []
        for i in range(self.n_horizons):
            x = torch.cat([
                base_mu[:, i],
                base_logvar[:, i],
                gfs_targets[:, i],
                spatial,
                temporal[:, i],
            ], dim=-1)
            alphas.append(torch.sigmoid(self.alpha_nets[i](x)))
        return torch.stack(alphas, dim=1)
