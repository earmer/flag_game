import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

import pick_flag_ai

from model import PPOTransformerPolicy
from selfplay_env import CTFFrontendRulesEnv
from state_encoder import StateEncoder, PLAYER_FEATURE_INDEX


CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def load_config(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def render_progress(step, total, prefix="rollout"):
    bar_len = 28
    filled = int(bar_len * step / total)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix} [{bar}] {step}/{total}", end="", flush=True)


def build_movable_mask(player_features, max_players):
    my_features = player_features[:, :max_players, :]
    valid = my_features[:, :, PLAYER_FEATURE_INDEX["valid"]]
    in_prison = my_features[:, :, PLAYER_FEATURE_INDEX["in_prison"]]
    return (valid == 1) & (in_prison == 0)

def compute_theta(update_idx, start, end, decay_updates):
    if decay_updates <= 0:
        return end
    progress = min(update_idx / float(decay_updates), 1.0)
    return start + (end - start) * progress


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

def count_opponent_in_prison(status):
    return sum(1 for p in status.get("opponentPlayer", []) if p.get("inPrison"))

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def nearest_distance(pos, targets):
    if not targets:
        return None
    return min(manhattan(pos, t) for t in targets)

def flag_positions(status):
    return [
        (f["posX"], f["posY"])
        for f in status.get("opponentFlag", [])
        if f.get("canPickup")
    ]

def distance_delta(prev_status, curr_status, target_positions):
    prev_flags = flag_positions(prev_status)
    curr_flags = flag_positions(curr_status)
    prev_players = {p["name"]: p for p in prev_status.get("myteamPlayer", [])}
    curr_players = {p["name"]: p for p in curr_status.get("myteamPlayer", [])}

    total = 0.0
    for name, curr_p in curr_players.items():
        if curr_p.get("inPrison"):
            continue
        prev_p = prev_players.get(name, curr_p)
        curr_pos = (curr_p["posX"], curr_p["posY"])
        prev_pos = (prev_p["posX"], prev_p["posY"])

        if curr_p.get("hasFlag"):
            prev_dist = nearest_distance(prev_pos, target_positions)
            curr_dist = nearest_distance(curr_pos, target_positions)
        else:
            prev_dist = nearest_distance(prev_pos, prev_flags)
            curr_dist = nearest_distance(curr_pos, curr_flags)

        if prev_dist is None or curr_dist is None:
            continue
        total += (prev_dist - curr_dist)
    return total

def compute_reward(prev_status, curr_status, weights, target_positions):
    prev_score = prev_status.get("myteamScore", 0.0)
    curr_score = curr_status.get("myteamScore", prev_score)
    prev_flags = count_has_flag(prev_status)
    curr_flags = count_has_flag(curr_status)
    prison_count = count_in_prison(curr_status)
    capture_delta = count_opponent_in_prison(curr_status) - count_opponent_in_prison(prev_status)

    score_delta = curr_score - prev_score
    flag_delta = curr_flags - prev_flags
    dist_delta = distance_delta(prev_status, curr_status, target_positions)

    return (
        weights["score"] * score_delta
        + weights["flag_pickup"] * flag_delta
        + weights["prison"] * prison_count
        + weights["distance"] * dist_delta
        + weights["capture"] * capture_delta
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

def select_actions_with_theta(model, grid_ids, player_features, device, max_players, theta, rng):
    grid_tensor = torch.from_numpy(grid_ids).unsqueeze(0).to(device)
    player_tensor = torch.from_numpy(player_features).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, values = model(grid_tensor, player_tensor)

    mask = build_movable_mask(player_tensor, max_players).float()
    dist = Categorical(logits=logits)
    if rng.random() < theta:
        actions = torch.randint(0, 5, dist.logits.shape[:-1], device=device)
    else:
        actions = dist.sample()
    actions = torch.where(mask.bool(), actions, torch.zeros_like(actions))
    logprob = (dist.log_prob(actions) * mask).sum(dim=1)
    return actions.squeeze(0).cpu().numpy(), logprob.item(), values.squeeze(0).item()

def select_actions_greedy(model, grid_ids, player_features, device, max_players):
    grid_tensor = torch.from_numpy(grid_ids).unsqueeze(0).to(device)
    player_tensor = torch.from_numpy(player_features).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, values = model(grid_tensor, player_tensor)

    mask = build_movable_mask(player_tensor, max_players)
    actions = torch.argmax(logits, dim=-1)
    actions = torch.where(mask.bool(), actions, torch.zeros_like(actions))
    return actions.squeeze(0).cpu().numpy()


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

def moves_to_actions(move_dict, players):
    action_map = {"up": 1, "down": 2, "left": 3, "right": 4, "": 0, None: 0}
    actions = []
    for player in players:
        move = move_dict.get(player["name"])
        actions.append(action_map.get(move, 0))
    return actions

def evaluate_against_pick_flag(model, env_cfg, device, episodes, max_players, eval_theta):
    eval_env = CTFFrontendRulesEnv(env_cfg)
    encoder_L = StateEncoder(width=eval_env.width, height=eval_env.height, max_players=max_players)
    encoder_L.start_game(eval_env.init_req["L"])

    rng = np.random.default_rng()

    wins = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs_L, obs_R = eval_env.reset()
            pick_flag_ai.start_game(eval_env.init_req["R"])

            done = False
            while not done:
                grid_L, players_L = encoder_L.encode(obs_L)
                if grid_L is None:
                    break
                actions_L, _, _ = select_actions_with_theta(
                    model, grid_L, players_L, device, max_players, eval_theta, rng
                )
                moves_R = pick_flag_ai.plan_next_actions(obs_R) or {}
                players_R = sorted(obs_R.get("myteamPlayer", []), key=lambda p: p.get("name", ""))
                actions_R = moves_to_actions(moves_R, players_R)
                obs_L, obs_R, done = eval_env.step(actions_L, actions_R)

            if eval_env.score["L"] > eval_env.score["R"]:
                wins += 1.0
            elif eval_env.score["L"] == eval_env.score["R"]:
                wins += 0.5
            pick_flag_ai.game_over({"action": "finished"})

    model.train()
    return wins / max(1, episodes)


def main():
    config = load_config(CONFIG_PATH)
    env_cfg = config.get("env", {})
    ppo_cfg = config.get("ppo", {})
    transformer_cfg = config.get("transformer", {})
    reward_cfg = config.get("reward", {})

    device = select_device()
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

    print(f"[PPO] training on device: {device}")

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
    eval_episodes = config.get("training", {}).get("eval_episodes", 5)
    theta_start = config.get("training", {}).get("theta_start", 0.2)
    theta_end = config.get("training", {}).get("theta_end", 0.02)
    theta_decay_updates = config.get("training", {}).get("theta_decay_updates", 200)
    eval_theta = config.get("training", {}).get("eval_theta", 0.05)

    reward_weights = {
        "score": reward_cfg.get("score", 1.0),
        "flag_pickup": reward_cfg.get("flag_pickup", 0.1),
        "prison": reward_cfg.get("prison", -0.01),
        "distance": reward_cfg.get("distance", 0.01),
        "capture": reward_cfg.get("capture", 0.05),
    }

    update_idx = 0
    obs_L, obs_R = env.reset()
    buffer = RolloutBuffer()
    rng = np.random.default_rng(env_cfg.get("seed", 0))

    while True:
        theta = compute_theta(update_idx, theta_start, theta_end, theta_decay_updates)
        buffer.reset()
        update_reward_L = 0.0
        update_reward_R = 0.0
        for step_idx in range(rollout_steps):
            grid_L, players_L = encoder_L.encode(obs_L)
            grid_R, players_R = encoder_R.encode(obs_R)
            if grid_L is None or grid_R is None:
                break

            actions_L, logprob_L, value_L = select_actions_with_theta(
                model, grid_L, players_L, device, env.num_players, theta, rng
            )
            actions_R, logprob_R, value_R = select_actions_with_theta(
                model, grid_R, players_R, device, env.num_players, theta, rng
            )
            next_obs_L, next_obs_R, done = env.step(actions_L, actions_R)

            reward_L = compute_reward(obs_L, next_obs_L, reward_weights, env.targets["L"])
            reward_R = compute_reward(obs_R, next_obs_R, reward_weights, env.targets["R"])
            update_reward_L += reward_L
            update_reward_R += reward_R

            buffer.add(grid_L, players_L, actions_L, logprob_L, value_L, reward_L, done)
            buffer.add(grid_R, players_R, actions_R, logprob_R, value_R, reward_R, done)

            obs_L, obs_R = next_obs_L, next_obs_R
            if done:
                obs_L, obs_R = env.reset()
            render_progress(step_idx + 1, rollout_steps, prefix=f"update {update_idx + 1}")
        print()

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
        print(
            f"[PPO] update {update_idx} net reward "
            f"L={update_reward_L:.3f} R={update_reward_R:.3f}"
        )
        if save_every and update_idx % save_every == 0:
            ckpt_path = Path(__file__).resolve().parent / config.get("checkpoint_path", "checkpoints/latest.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"[PPO] saved checkpoint to {ckpt_path}")
            win_rate = evaluate_against_pick_flag(
                model, env_cfg, device, eval_episodes, env.num_players, eval_theta
            )
            print(f"[PPO] eval vs pick_flag_ai win_rate={win_rate:.2%} over {eval_episodes} eps")


if __name__ == "__main__":
    main()
