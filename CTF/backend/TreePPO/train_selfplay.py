from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from PPO.selfplay_env import CTFFrontendRulesEnv  # noqa: E402

from model import ACTION_COUNT, TreePPOPolicy  # noqa: E402
from reward import RewardTracker  # noqa: E402
from state_encoder import TeamHistoryStateEncoder  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().with_name("config.json")

# log helpers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("TreePPO.train")

# AI_Design.md action indices -> env.step indices.
AI_TO_ENV_ACTION = {
    0: 1,  # up
    1: 2,  # down
    2: 3,  # left
    3: 4,  # right
    4: 0,  # stay
}


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def render_progress(step: int, total: int, *, prefix: str) -> None:
    bar_len = 28
    filled = int(bar_len * step / max(1, total))
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix} [{bar}] {step}/{total}", end="", flush=True)


def ai_actions_to_env(actions_ai: np.ndarray, *, encoder: TeamHistoryStateEncoder) -> list[int]:
    out: list[int] = []
    for a in actions_ai.tolist():
        a = int(a)
        a = encoder.denormalize_action(a)
        out.append(int(AI_TO_ENV_ACTION.get(a, 0)))
    return out


@dataclass
class Rollout:
    obs: list[np.ndarray]
    actions: list[np.ndarray]
    logprobs: list[np.ndarray]
    values: list[np.ndarray]
    rewards: list[np.ndarray]
    active_masks: list[np.ndarray]
    dones: list[float]

    def __init__(self) -> None:
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.active_masks = []
        self.dones = []

    def add(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        active_masks: np.ndarray,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.active_masks.append(active_masks)
        self.dones.append(1.0 if done else 0.0)

    def as_arrays(self) -> tuple[np.ndarray, ...]:
        obs = np.stack(self.obs, axis=0)  # (T, N, C, H, W)
        actions = np.stack(self.actions, axis=0)  # (T, N)
        logprobs = np.stack(self.logprobs, axis=0)  # (T, N)
        values = np.stack(self.values, axis=0)  # (T, N)
        rewards = np.stack(self.rewards, axis=0)  # (T, N)
        active_masks = np.stack(self.active_masks, axis=0)  # (T, N)
        dones = np.array(self.dones, dtype=np.float32)  # (T,)
        return obs, actions, logprobs, values, rewards, active_masks, dones


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    rewards: (T, N)
    values:  (T, N)
    dones:   (T,)
    """
    t_steps, n_agents = rewards.shape
    advantages = np.zeros((t_steps, n_agents), dtype=np.float32)
    last_adv = np.zeros((n_agents,), dtype=np.float32)

    for t in reversed(range(t_steps)):
        next_values = values[t + 1] if (t + 1) < t_steps else np.zeros((n_agents,), dtype=np.float32)
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


def sample_actions(
    model: TreePPOPolicy,
    obs_batch: np.ndarray,
    active_mask: np.ndarray,
    *,
    device: torch.device,
    epsilon: float,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_t = torch.from_numpy(obs_batch).to(device)
    with torch.no_grad():
        logits, values = model(obs_t)

    dist = Categorical(logits=logits)
    actions = dist.sample()  # (B,)
    if epsilon > 0.0:
        explore = torch.tensor([rng.random() < epsilon for _ in range(actions.shape[0])], device=device)
        random_actions = torch.randint(0, ACTION_COUNT, actions.shape, device=device)
        actions = torch.where(explore, random_actions, actions)

    active = torch.from_numpy(active_mask.astype(np.float32)).to(device)
    stay = torch.full_like(actions, 4)  # AI_Design stay
    actions = torch.where(active > 0.0, actions, stay)

    logprobs = dist.log_prob(actions)
    return actions.cpu().numpy(), logprobs.cpu().numpy(), values.cpu().numpy()


def evaluate_actions(
    model: TreePPOPolicy,
    obs: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values = model(obs)
    dist = Categorical(logits=logits)
    logprobs = dist.log_prob(actions)
    entropy = dist.entropy()
    return logprobs, entropy, values


def main() -> None:
    config = load_config(CONFIG_PATH)
    env_cfg = config.get("env", {})
    ppo_cfg = config.get("ppo", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    seed = int(training_cfg.get("seed", 0))
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = select_device()
    env = CTFFrontendRulesEnv(env_cfg)

    LOGGER.info(
        "Launcher config: env=%sx%s players=%s depth=%s history=%s seed=%s",
        env.width,
        env.height,
        env.num_players,
        model_cfg.get("depth"),
        model_cfg.get("history_len"),
        training_cfg.get("seed"),
    )

    encoder_L = TeamHistoryStateEncoder(
        width=env.width,
        height=env.height,
        max_players=env.num_players,
        history_len=int(model_cfg.get("history_len", 3)),
        normalize_side=bool(model_cfg.get("normalize_side", True)),
    )
    encoder_R = TeamHistoryStateEncoder(
        width=env.width,
        height=env.height,
        max_players=env.num_players,
        history_len=int(model_cfg.get("history_len", 3)),
        normalize_side=bool(model_cfg.get("normalize_side", True)),
    )
    encoder_L.start_game(env.init_req["L"])
    encoder_R.start_game(env.init_req["R"])

    tracker_L = RewardTracker(max_players=env.num_players)
    tracker_R = RewardTracker(max_players=env.num_players)
    tracker_L.start_game(env.init_req["L"])
    tracker_R.start_game(env.init_req["R"])

    in_channels = int(model_cfg.get("history_len", 3)) * 12
    model = TreePPOPolicy(
        in_channels=in_channels,
        feature_dim=int(model_cfg.get("feature_dim", 128)),
        depth=int(model_cfg.get("depth", 10)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo_cfg.get("learning_rate", 3e-4)))

    ckpt_path = Path(__file__).resolve().parent / str(config.get("checkpoint_path", "checkpoints/latest.pt"))
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model", state))
        print(f"[TreePPO] loaded checkpoint: {ckpt_path}")

    rollout_steps = int(ppo_cfg.get("rollout_steps", 1024))
    epochs = int(ppo_cfg.get("epochs", 4))
    batch_size = int(ppo_cfg.get("batch_size", 256))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_ratio = float(ppo_cfg.get("clip_ratio", 0.2))
    value_coef = float(ppo_cfg.get("value_coef", 0.5))
    entropy_coef = float(ppo_cfg.get("entropy_coef", 0.01))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 1.0))

    save_every = int(training_cfg.get("save_every_updates", 5))
    epsilon = float(training_cfg.get("epsilon", 0.05))
    reward_div = float(training_cfg.get("reward_divide_by_players", 1.0))

    print(f"[TreePPO] training on device: {device}")
    LOGGER.info("Beginning training on device=%s (epsilon=%s)", device, epsilon)

    update_idx = 0
    obs_L, obs_R = env.reset()
    tracker_L.reset_episode(obs_L)
    tracker_R.reset_episode(obs_R)
    encoder_L.reset_history()
    encoder_R.reset_history()

    while True:
        model.train()
        rollout = Rollout()
        update_reward_L = 0.0
        update_reward_R = 0.0
        LOGGER.info(
            "Update %d: collecting %d steps (epsilon=%.3f)",
            update_idx + 1,
            rollout_steps,
            epsilon,
        )

        for step_idx in range(rollout_steps):
            obs_L_batch, players_L, active_L = encoder_L.encode_team(obs_L)
            obs_R_batch, players_R, active_R = encoder_R.encode_team(obs_R)
            if obs_L_batch.size == 0 or obs_R_batch.size == 0:
                break

            actions_ai_L, logp_L, v_L = sample_actions(
                model, obs_L_batch, active_L, device=device, epsilon=epsilon, rng=rng
            )
            actions_ai_R, logp_R, v_R = sample_actions(
                model, obs_R_batch, active_R, device=device, epsilon=epsilon, rng=rng
            )

            actions_env_L = ai_actions_to_env(actions_ai_L, encoder=encoder_L)
            actions_env_R = ai_actions_to_env(actions_ai_R, encoder=encoder_R)

            next_obs_L, next_obs_R, done = env.step(actions_env_L, actions_env_R)

            reward_L = tracker_L.compute(obs_L, next_obs_L, done=done, actions_env=actions_env_L)
            reward_R = tracker_R.compute(obs_R, next_obs_R, done=done, actions_env=actions_env_R)
            update_reward_L += reward_L
            update_reward_R += reward_R

            if reward_div > 0:
                reward_L = reward_L / reward_div
                reward_R = reward_R / reward_div

            n = env.num_players
            obs_all = np.concatenate([obs_L_batch, obs_R_batch], axis=0)
            actions_all = np.concatenate([actions_ai_L, actions_ai_R], axis=0)
            logp_all = np.concatenate([logp_L, logp_R], axis=0)
            v_all = np.concatenate([v_L, v_R], axis=0)
            active_all = np.concatenate([active_L, active_R], axis=0)
            rewards_all = np.concatenate(
                [np.full((n,), reward_L, dtype=np.float32), np.full((n,), reward_R, dtype=np.float32)],
                axis=0,
            )

            rollout.add(
                obs=obs_all.astype(np.float32, copy=False),
                actions=actions_all.astype(np.int64, copy=False),
                logprobs=logp_all.astype(np.float32, copy=False),
                values=v_all.astype(np.float32, copy=False),
                rewards=rewards_all,
                active_masks=active_all.astype(np.float32, copy=False),
                done=done,
            )

            obs_L, obs_R = next_obs_L, next_obs_R

            if done:
                LOGGER.info(
                    "Episode done at step %d (env time=%.1f) score L=%d R=%d",
                    step_idx + 1,
                    env.time,
                    env.score["L"],
                    env.score["R"],
                )
                obs_L, obs_R = env.reset()
                tracker_L.reset_episode(obs_L)
                tracker_R.reset_episode(obs_R)
                encoder_L.reset_history()
                encoder_R.reset_history()

            render_progress(step_idx + 1, rollout_steps, prefix=f"update {update_idx + 1}")
        print()

        LOGGER.info(
            "Update %d finished: reward_sum L=%.3f R=%.3f",
            update_idx + 1,
            update_reward_L,
            update_reward_R,
        )
        obs_arr, actions_arr, old_logp_arr, values_arr, rewards_arr, active_arr, dones_arr = rollout.as_arrays()
        adv_arr, ret_arr = compute_gae(rewards_arr, values_arr, dones_arr, gamma=gamma, gae_lambda=gae_lambda)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        t_steps, n_agents = actions_arr.shape
        flat_obs = obs_arr.reshape(t_steps * n_agents, *obs_arr.shape[2:])
        flat_actions = actions_arr.reshape(t_steps * n_agents)
        flat_old_logp = old_logp_arr.reshape(t_steps * n_agents)
        flat_adv = adv_arr.reshape(t_steps * n_agents)
        flat_ret = ret_arr.reshape(t_steps * n_agents)
        flat_active = active_arr.reshape(t_steps * n_agents)

        obs_t = torch.from_numpy(flat_obs).to(device)
        actions_t = torch.from_numpy(flat_actions).to(device)
        old_logp_t = torch.from_numpy(flat_old_logp).to(device)
        adv_t = torch.from_numpy(flat_adv.astype(np.float32)).to(device)
        ret_t = torch.from_numpy(flat_ret.astype(np.float32)).to(device)
        active_t = torch.from_numpy(flat_active.astype(np.float32)).to(device)

        total = obs_t.shape[0]
        indices = np.arange(total)
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, total, batch_size):
                mb_idx = indices[start:start + batch_size]
                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]
                mb_active = active_t[mb_idx]

                new_logp, entropy, values = evaluate_actions(model, mb_obs, mb_actions)
                ratio = torch.exp(new_logp - mb_old_logp)

                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
                policy_loss = -(torch.min(unclipped, clipped) * mb_active).sum() / (mb_active.sum() + 1e-8)

                value_loss = (mb_ret - values).pow(2).mean()
                entropy_loss = -(entropy * mb_active).sum() / (mb_active.sum() + 1e-8)

                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        update_idx += 1
        print(f"[TreePPO] update {update_idx} reward_sum L={update_reward_L:.3f} R={update_reward_R:.3f}")

        if save_every and update_idx % save_every == 0:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, ckpt_path)
            LOGGER.info("Saved checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
