"""
LSTM forecaster for short-range weather prediction.

Uses the last 12 hours of observation history (from the 48h window) to
capture sequential momentum — pressure trends, temperature tendencies,
wind shifts — and predicts at multiple horizons.

The same LSTM hidden state is reused for all horizons, but each horizon
receives its own GFS anchor and temporal context through the prediction head.
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_vars: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_spatial: int = 2,
        n_temporal: int = 4,
        n_horizons: int = 5,
        lstm_lookback: int = 12,
        dropout: float = 0.1,
        include_gfs: bool = True,
    ):
        super().__init__()
        self.lstm_lookback = lstm_lookback
        self.n_horizons = n_horizons
        self.n_vars = n_vars
        self.include_gfs = include_gfs

        self.lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        ctx_dim = hidden_dim + n_spatial + n_temporal
        if include_gfs:
            ctx_dim += n_vars

        self.fc_mu = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_vars),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(ctx_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_vars),
        )

    def forward(self, obs_hist, gfs_targets, spatial, temporal):
        """
        Args:
            obs_hist:    (B, 48, 6) observation history — uses last 12 steps
            gfs_targets: (B, H, 6) GFS forecast at each horizon
            spatial:     (B, 2)    lat/lon
            temporal:    (B, H, 4) sin/cos hour+doy at each horizon
        Returns:
            mu:     (B, H, 6) predicted mean
            logvar: (B, H, 6) predicted log-variance
        """
        # Use only the most recent lstm_lookback hours
        h_in = obs_hist[:, -self.lstm_lookback :]
        _, (h_n, _) = self.lstm(h_in)
        h_last = h_n[-1]  # (B, hidden_dim)

        mus, logvars = [], []
        for i in range(self.n_horizons):
            if self.include_gfs:
                ctx = torch.cat([h_last, gfs_targets[:, i], spatial, temporal[:, i]], dim=-1)
            else:
                ctx = torch.cat([h_last, spatial, temporal[:, i]], dim=-1)
            mus.append(self.fc_mu(ctx))
            logvars.append(self.fc_logvar(ctx))

        return torch.stack(mus, dim=1), torch.stack(logvars, dim=1)
