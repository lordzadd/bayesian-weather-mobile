"""
PatchTST forecaster for full-trajectory weather prediction.

Channel-independent design: each weather variable is processed through the
same transformer backbone separately, then combined at the prediction head.
Input is patched into groups of 6 consecutive hours (8 patches from 48h).

This captures both local temporal patterns within each patch and long-range
dependencies across patches via self-attention, while keeping the model
compact enough for eventual mobile distillation.
"""

import torch
import torch.nn as nn
import math


class PatchTSTForecaster(nn.Module):
    def __init__(
        self,
        n_vars: int = 6,
        seq_len: int = 48,
        patch_len: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_spatial: int = 2,
        n_temporal: int = 4,
        n_horizons: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.d_model = d_model
        self.n_horizons = n_horizons

        # Shared patch embedding (applied independently per channel)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )
        self.input_dropout = nn.Dropout(dropout)

        # Transformer encoder (shared across channels)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Per-horizon prediction head
        ctx_dim = d_model * n_vars + n_vars + n_spatial + n_temporal

        self.fc_mu = nn.Sequential(
            nn.Linear(ctx_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_vars),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_vars),
        )

    def forward(self, obs_hist, gfs_targets, spatial, temporal):
        """
        Args:
            obs_hist:    (B, 48, 6) observation history
            gfs_targets: (B, H, 6) GFS forecast at each horizon
            spatial:     (B, 2)    lat/lon
            temporal:    (B, H, 4) sin/cos hour+doy at each horizon
        Returns:
            mu:     (B, H, 6) predicted mean
            logvar: (B, H, 6) predicted log-variance
        """
        B = obs_hist.shape[0]

        # Channel-independent: reshape (B, seq, V) → (B*V, seq)
        x = obs_hist.permute(0, 2, 1).reshape(B * self.n_vars, -1)

        # Patch: (B*V, n_patches, patch_len)
        x = x.reshape(B * self.n_vars, self.n_patches, self.patch_len)

        # Embed patches → (B*V, n_patches, d_model)
        x = self.patch_embed(x) + self.pos_embed
        x = self.input_dropout(x)

        # Transformer encode
        x = self.encoder(x)
        x = self.norm(x)

        # Pool over patches → (B*V, d_model)
        x = x.mean(dim=1)

        # Reshape back → (B, V * d_model)
        x = x.reshape(B, self.n_vars * self.d_model)

        # Per-horizon prediction
        mus, logvars = [], []
        for i in range(self.n_horizons):
            ctx = torch.cat([x, gfs_targets[:, i], spatial, temporal[:, i]], dim=-1)
            mus.append(self.fc_mu(ctx))
            logvars.append(self.fc_logvar(ctx))

        return torch.stack(mus, dim=1), torch.stack(logvars, dim=1)
