from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOModel(nn.Module):
    """Lightweight CNN actor-critic model for multi-agent (3 players) actions."""

    def __init__(self, *, num_entity_ids: int = 20, num_players: int = 3, num_actions: int = 5) -> None:
        super().__init__()

        self.num_entity_ids = int(num_entity_ids)
        self.num_players = int(num_players)
        self.num_actions = int(num_actions)

        # CNN backbone (expects one-hot encoded channels)
        self.conv1 = nn.Conv2d(self.num_entity_ids, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)

        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

        # 20x20 -> pool => 10x10 -> stride2 => 4x4
        self.fc_hidden = nn.Linear(256 * 4 * 4, 512)

        self.actor = nn.Linear(512, self.num_players * self.num_actions)
        self.critic = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def _to_one_hot(self, state: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
        - (B, H, W) integer IDs
        - (B, 1, H, W) integer IDs
        - (B, C, H, W) already-one-hot / continuous features (C must be num_entity_ids)
        Returns:
        - (B, num_entity_ids, H, W) float
        """
        if state.dim() == 3:
            ids = state
        elif state.dim() == 4:
            if state.size(1) == self.num_entity_ids:
                return state.float()
            if state.size(1) == 1:
                ids = state.squeeze(1)
            else:
                raise ValueError(f"Unexpected state shape: {tuple(state.shape)}")
        else:
            raise ValueError(f"Unexpected state dims: {state.dim()}")

        ids = ids.to(dtype=torch.long)
        oh = F.one_hot(ids.clamp(min=0, max=self.num_entity_ids - 1), num_classes=self.num_entity_ids)
        return oh.permute(0, 3, 1, 2).to(dtype=torch.float32)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._to_one_hot(state)

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.flatten(x)
        x = self.relu(self.fc_hidden(x))

        action_logits = self.actor(x).view(-1, self.num_players, self.num_actions)
        value = self.critic(x)
        return action_logits, value

