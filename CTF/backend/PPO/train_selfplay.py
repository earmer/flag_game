import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from model import PPOTransformerPolicy
from selfplay_env import CTFFrontendRulesEnv
from state_encoder import StateEncoder, PLAYER_FEATURE_INDEX


CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def load_config(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def build_movable_mask(player_features, max_players):
    my_features = player_features[:, :max_players, :]
    valid = my_features[:, :, PLAYER_FEATURE_INDEX["valid"]]
    in_prison = my_features[:, :, PLAYER_FEATURE_INDEX["in_prison"]]
    return (valid == 1) & (in_prison == 0)


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def count_has_flag(status):
    return sum(1 for p in status.get("myteamPlayer", []) if p.get("hasFlag"))


def count_in_prison(status):
    return sum(1 for p in status.get("myteamPlayer", []) if p.get("inPrison"))


def compute_reward(prev_status, curr_status, weights):
    prev_score = prev_status.get("myteamScore", 0.0)
    curr_score = curr_status.get("myteamScore", prev_score)
    prev_flags = count_has_flag(prev_status)
    curr_flags = count_has_flag(curr_status)
    prison_count = count_in_prison(curr_status)

    score_delta = curr_score - prev_score
    flag_delta = curr_flags - prev_flags

    return (
        weights["score"] * score_delta
        + weights["flag_pickup"] * flag_delta
        + weights["prison"] * prison_count
    )


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid_ids = []
        self.player_features = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, grid_ids, player_features, actions, logprob, value, reward, done):
        self.grid_ids.append(grid_ids)
        self.player_features.append(player_features)
        self.actions.append(actions)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(float(done))

    def as_tensors(self, device):
        grid_ids = torch.tensor(np.array(self.grid_ids), dtype=torch.long, device=device)
        player_features = torch.tensor(np.array(self.player_features), dtype=torch.long, device=device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=device)
        logprobs = torch.tensor(np.array(self.logprobs), dtype=torch.float32, device=device)
        values = torch.tensor(np.array(self.values), dtype=torch.float32, device=device)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        return grid_ids, player_features, actions, logprobs, values, rewards, dones


def select_actions(model, grid_ids, player_features, device, max_players):
    grid_tensor = torch.from_numpy(grid_ids).unsqueeze(0).to(device)
    player_tensor = torch.from_numpy(player_features).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, values = model(grid_tensor, player_tensor)

    mask = build_movable_mask(player_tensor, max_players).float()
    dist = Categorical(logits=logits)
    actions = dist.sample()
    actions = torch.where(mask.bool(), actions, torch.zeros_like(actions))
    logprob = (dist.log_prob(actions) * mask).sum(dim=1)
    return actions.squeeze(0).cpu().numpy(), logprob.item(), values.squeeze(0).item()


def evaluate_actions(model, grid_ids, player_features, actions, max_players):
    logits, values = model(grid_ids, player_features)
    dist = Categorical(logits=logits)
    mask = build_movable_mask(player_features, max_players).float()
    logprobs = (dist.log_prob(actions) * mask).sum(dim=1)
    entropy = (dist.entropy() * mask).sum(dim=1)
    return logprobs, entropy, values


def load_checkpoint(model, device, path):
    ckpt_path = Path(__file__).resolve().parent / path
    if not ckpt_path.exists():
        return
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"[PPO] loaded checkpoint from {ckpt_path}")


def main():
    config = load_config(CONFIG_PATH)
    env_cfg = config.get("env", {})
    ppo_cfg = config.get("ppo", {})
    transformer_cfg = config.get("transformer", {})
    reward_cfg = config.get("reward", {})

    device = torch.device(os.environ.get("CTF_PPO_DEVICE", "cpu"))
    env = CTFFrontendRulesEnv(env_cfg)

    encoder_L = StateEncoder(width=env.width, height=env.height, max_players=env.num_players)
    encoder_R = StateEncoder(width=env.width, height=env.height, max_players=env.num_players)
    encoder_L.start_game(env.init_req["L"])
    encoder_R.start_game(env.init_req["R"])

    model = PPOTransformerPolicy(
        width=env.width,
        height=env.height,
        max_players=env.num_players,
        vector_dim=config.get("vector_dim", 64),
        num_layers=transformer_cfg.get("num_layers", 4),
        num_heads=transformer_cfg.get("num_heads", 4),
        dropout=transformer_cfg.get("dropout", 0.1),
        mlp_dim=transformer_cfg.get("mlp_dim", 128),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.get("learning_rate", 3e-4))

    load_checkpoint(model, device, config.get("checkpoint_path", "checkpoints/latest.pt"))

    rollout_steps = ppo_cfg.get("rollout_steps", 256)
    epochs = ppo_cfg.get("epochs", 4)
    batch_size = ppo_cfg.get("batch_size", 128)
    gamma = ppo_cfg.get("gamma", 0.99)
    gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
    clip_ratio = ppo_cfg.get("clip_ratio", 0.2)
    value_coef = ppo_cfg.get("value_coef", 0.5)
    entropy_coef = ppo_cfg.get("entropy_coef", 0.01)
    save_every = config.get("training", {}).get("save_every_updates", 5)

    reward_weights = {
        "score": reward_cfg.get("score", 1.0),
        "flag_pickup": reward_cfg.get("flag_pickup", 0.1),
        "prison": reward_cfg.get("prison", -0.01),
    }

    update_idx = 0
    obs_L, obs_R = env.reset()
    buffer = RolloutBuffer()

    while True:
        buffer.reset()
        for _ in range(rollout_steps):
            grid_L, players_L = encoder_L.encode(obs_L)
            grid_R, players_R = encoder_R.encode(obs_R)
            if grid_L is None or grid_R is None:
                break

            actions_L, logprob_L, value_L = select_actions(
                model, grid_L, players_L, device, env.num_players
            )
            actions_R, logprob_R, value_R = select_actions(
                model, grid_R, players_R, device, env.num_players
            )
            next_obs_L, next_obs_R, done = env.step(actions_L, actions_R)

            reward_L = compute_reward(obs_L, next_obs_L, reward_weights)
            reward_R = compute_reward(obs_R, next_obs_R, reward_weights)

            buffer.add(grid_L, players_L, actions_L, logprob_L, value_L, reward_L, done)
            buffer.add(grid_R, players_R, actions_R, logprob_R, value_R, reward_R, done)

            obs_L, obs_R = next_obs_L, next_obs_R
            if done:
                obs_L, obs_R = env.reset()

        grid_ids, player_features, actions, old_logprobs, values, rewards, dones = buffer.as_tensors(device)
        advantages, returns = compute_gae(
            rewards, values.cpu().numpy(), dones, gamma, gae_lambda
        )
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_steps = grid_ids.shape[0]
        indices = np.arange(total_steps)
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, total_steps, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_grid = grid_ids[mb_idx]
                mb_players = player_features[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_logprobs, entropy, new_values = evaluate_actions(
                    model, mb_grid, mb_players, mb_actions, env.num_players
                )
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = (mb_returns - new_values).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        update_idx += 1
        if save_every and update_idx % save_every == 0:
            ckpt_path = Path(__file__).resolve().parent / config.get("checkpoint_path", "checkpoints/latest.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"[PPO] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
