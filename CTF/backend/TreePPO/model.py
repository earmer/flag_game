from __future__ import annotations

import math

import torch
import torch.nn as nn


ACTION_COUNT = 5  # AI_Design.md: 0 up, 1 down, 2 left, 3 right, 4 stay


class ConvFeatureEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),  # 20x20 -> 10x10
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),  # 10x10 -> 5x5
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.GELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class SoftDecisionTree(nn.Module):
    def __init__(self, feature_dim: int, depth: int, output_dim: int) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be >= 1")
        self.feature_dim = int(feature_dim)
        self.depth = int(depth)
        self.output_dim = int(output_dim)

        self.num_leaves = 2**self.depth
        self.num_nodes = self.num_leaves - 1

        self.gate = nn.Linear(self.feature_dim, self.num_nodes)

        bound = 1.0 / math.sqrt(max(1, self.num_leaves))
        self.leaf_logits = nn.Parameter(torch.empty(self.num_leaves, self.output_dim).uniform_(-bound, bound))
        self.leaf_values = nn.Parameter(torch.empty(self.num_leaves).uniform_(-bound, bound))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        features: (B, feature_dim)
        Returns:
          logits: (B, output_dim)
          value: (B,)
        """
        batch_size = features.shape[0]
        gate = torch.sigmoid(self.gate(features))  # (B, num_nodes), p(right)

        probs = torch.ones((batch_size, 1), device=features.device, dtype=features.dtype)
        node_offset = 0
        for depth in range(self.depth):
            count = 2**depth
            g = gate[:, node_offset:node_offset + count]  # (B, count)
            node_offset += count
            probs = torch.cat([probs * (1.0 - g), probs * g], dim=1)  # (B, 2*count)

        leaf_probs = probs  # (B, num_leaves)
        logits = leaf_probs @ self.leaf_logits  # (B, output_dim)
        value = (leaf_probs * self.leaf_values.unsqueeze(0)).sum(dim=1)
        return logits, value


class TreePPOPolicy(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 36,
        feature_dim: int = 128,
        depth: int = 10,
        action_count: int = ACTION_COUNT,
    ) -> None:
        super().__init__()
        self.encoder = ConvFeatureEncoder(in_channels=in_channels, feature_dim=feature_dim)
        self.tree = SoftDecisionTree(feature_dim=feature_dim, depth=depth, output_dim=action_count)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        return self.tree(features)

