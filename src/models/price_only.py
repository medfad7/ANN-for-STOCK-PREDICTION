# src/models/price_only.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PriceOnlyMLP(nn.Module):
    def __init__(self, window_size: int, num_features: int,
                 hidden_sizes=(64, 32)):
        super().__init__()
        input_dim = window_size * num_features

        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # binary logit output

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, W, F)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # flatten window
        logit = self.net(x).squeeze(-1)  # (batch,)
        return logit  # use BCEWithLogitsLoss outside
