from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ppo_env_adapter import PPOEnvAdapter
from ppo_model import PPOModel
from ppo_trainer import PPOTrainer


def _random_opponent_actions(team: str, num_players: int) -> Dict[str, Optional[str]]:
    opponent = "R" if team == "L" else "L"
    choices = ["up", "down", "left", "right", None]
    return {f"{opponent}{i}": np.random.choice(choices) for i in range(num_players)}


def collect_trajectory(
    env: PPOEnvAdapter,
    model: PPOModel,
    *,
    max_steps: int = 500,
    device: torch.device,
) -> Dict[str, Any]:
    states: list[np.ndarray] = []
    actions: list[torch.Tensor] = []
    rewards: list[float] = []
    values: list[float] = []
    log_probs: list[float] = []
    dones: list[bool] = []

    state = env.reset()

    for _ in range(max_steps):
        state_tensor = torch.from_numpy(state).to(device=device).unsqueeze(0)

        with torch.no_grad():
            action_logits, value = model(state_tensor)
            dist = torch.distributions.Categorical(logits=action_logits[0])  # (num_players, num_actions)
            action = dist.sample()  # (num_players,)
            log_prob = dist.log_prob(action).sum()

        opponent_actions = _random_opponent_actions(env.team, env.num_players)
        next_state, reward, done, info = env.step(action.cpu().numpy(), opponent_actions)

        states.append(state)
        actions.append(action.cpu())
        rewards.append(float(reward))
        values.append(float(value.item()))
        log_probs.append(float(log_prob.item()))
        dones.append(bool(done))

        state = next_state

        if done:
            break

    return {
        "states": np.stack(states, axis=0),
        "actions": torch.stack(actions, dim=0),
        "rewards": rewards,
        "values": values,
        "log_probs": torch.tensor(log_probs, dtype=torch.float32),
        "dones": dones,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PPOEnvAdapter(team="L")
    model = PPOModel(num_players=env.num_players, num_actions=env.num_actions).to(device)
    trainer = PPOTrainer(model)

    checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # --- Load latest checkpoint if exists ---
    start_episode = 0
    ckpt_files = list(checkpoints_dir.glob("*.pt"))
    if ckpt_files:
        # 按修改时间排序，取最新的
        latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading latest checkpoint: {latest_ckpt.name}")
        try:
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
            # 尝试从文件名中提取起始集数 (ppo_ep100_...)
            match = re.search(r"ppo_ep(\d+)_", latest_ckpt.name)
            if match:
                start_episode = int(match.group(1)) + 1
                print(f"Resuming from episode {start_episode}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    num_episodes = 1000

    for episode in range(start_episode, num_episodes):
        traj = collect_trajectory(env, model, device=device)

        advantages, returns = trainer.compute_gae(traj["rewards"], traj["values"], traj["dones"])

        states = torch.from_numpy(traj["states"]).to(device=device)
        actions = traj["actions"].to(device=device)
        old_log_probs = traj["log_probs"].to(device=device)
        advantages = advantages.to(device=device)
        returns = returns.to(device=device)

        metrics = trainer.train_step(states, actions, old_log_probs, advantages, returns)

        if episode % 10 == 0:
            total_reward = sum(traj["rewards"])
            print(
                f"Episode {episode}: reward={total_reward:.2f} loss={metrics.loss:.4f} "
                f"policy={metrics.policy_loss:.4f} value={metrics.value_loss:.4f} ent={metrics.entropy:.4f}"
            )

        if episode % 100 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = checkpoints_dir / f"ppo_ep{episode}_{timestamp}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path.name}")


if __name__ == "__main__":
    main()

