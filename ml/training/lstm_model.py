"""
LSTM sequence model for weather bias correction.

Takes a sequence of T past hourly [gfs_norm(6), spatial(2)] observations
and predicts the posterior mean and log-std for the next hour's ERA5 values.

Architecture:
    LSTM(input=8, hidden=64, layers=2)
    → final hidden state h_T
    → mean_head:    Linear(64 → 6)
    → log_std_head: Linear(64 → 6)  (exponentiated → std)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.mean_head = nn.Linear(hidden_size, output_size)
        self.log_std_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        """
        x: [batch, seq_len, input_size]
        returns: (mean[batch, output_size], std[batch, output_size])
        """
        _, (h_n, _) = self.lstm(x)
        # h_n: [num_layers, batch, hidden]
        h = h_n[-1]  # last layer's hidden state: [batch, hidden]

        mean = self.mean_head(h)
        std = torch.exp(self.log_std_head(h)).clamp(min=1e-4)
        return mean, std
