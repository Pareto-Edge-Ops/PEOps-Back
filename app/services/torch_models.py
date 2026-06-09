"""Tiny torch architectures for synthesized import models.

Module-level classes so `torch.save(model)` pickles them (.pt imports).
This module imports torch — only ever loaded lazily from model_factory.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 24, out_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TinyCNN(nn.Module):
    def __init__(self, channels: int = 8, n_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels * 2, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.conv2(x))
        x = self.gap(x).flatten(1)
        return self.fc(x)


class TinyAttention(nn.Module):
    """Single-head self-attention block → exports MatMul/Softmax/MatMul."""

    def __init__(self, dim: int = 16, n_classes: int = 4):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Linear(dim, dim * 2)
        self.act = nn.ReLU()
        self.head = nn.Linear(dim * 2, n_classes)
        self.scale = dim ** -0.5

    def forward(self, x):  # x: [B, T, D]
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        z = self.norm(attn @ v + x)
        z = self.act(self.ffn(z))
        return self.head(z.mean(dim=1))


class TinyLSTM(nn.Module):
    def __init__(self, in_dim: int = 12, hidden: int = 16, n_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: [B, T, D]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
