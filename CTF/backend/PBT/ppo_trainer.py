from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass(slots=True)
class PPOMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float


class PPOTrainer:
    """Minimal PPO trainer for the multi-player (3x discrete action) policy."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.clip_eps = float(clip_eps)
        self.gamma = float(gamma)
        self.lam = float(lam)

        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            advantages: (T,)
            returns: (T,)
        """
        if not (len(rewards) == len(values) == len(dones)):
            raise ValueError("rewards/values/dones must have same length")

        advantages: list[float] = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == (len(rewards) - 1) else float(values[t + 1])
            not_done = 0.0 if bool(dones[t]) else 1.0
            delta = float(rewards[t]) + self.gamma * next_value * not_done - float(values[t])
            gae = delta + self.gamma * self.lam * not_done * gae
            advantages.insert(0, gae)

        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        returns_t = advantages_t + values_t
        return advantages_t, returns_t

    def train_step(
        self,
        states: torch.Tensor,  # (B, H, W)
        actions: torch.Tensor,  # (B, num_players)
        old_log_probs: torch.Tensor,  # (B,)
        advantages: torch.Tensor,  # (B,)
        returns: torch.Tensor,  # (B,)
    ) -> PPOMetrics:
        action_logits, values = self.model(states)

        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)  # (B,)

        adv = advantages
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(-1), returns)

        entropy = dist.entropy().mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return PPOMetrics(
            loss=float(loss.item()),
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            entropy=float(entropy.item()),
        )

