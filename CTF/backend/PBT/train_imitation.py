from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mock_env_vnew import MockEnvVNew
from ppo_model import PPOModel
from state_normalizer import StateNormalizer
from strategic_opponent import StrategicOpponent

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from lib.matrix_util import CTFMatrixConverter  # noqa: E402


def collect_expert_trajectory(
    env: MockEnvVNew,
    expert: StrategicOpponent,
    converter: CTFMatrixConverter,
    normalizer: StateNormalizer,
    *,
    max_steps: int = 500,
) -> Dict:
    """
    收集专家轨迹

    关键：专家始终在 L 队（左侧），对手在 R 队（右侧）
    """
    states = []
    actions = []

    # 重置环境
    full_state = env.reset()

    # 初始化矩阵转换器
    init_payload = env.get_init_payload("L")
    converter.initialize_static_map(init_payload)

    for step in range(max_steps):
        # 获取专家动作（L 队）
        expert_actions = expert.get_actions(full_state)

        # 获取随机对手动作（R 队）
        random_actions = {
            f"R{i}": np.random.choice(["up", "down", "left", "right", None])
            for i in range(env.num_players)
        }

        # 获取当前状态矩阵
        status = full_state["L"]
        state_matrix = converter.convert_to_matrix(status)
        if state_matrix is None:
            break

        # 标准化状态（L 队已经在左侧，无需翻转）
        normalized_state = normalizer.normalize_state(state_matrix, "L")

        # 将专家动作转换为模型输出格式
        action_ids = []
        for i in range(env.num_players):
            player_name = f"L{i}"
            direction = expert_actions.get(player_name)
            action_id = _direction_to_id(direction)
            action_ids.append(action_id)

        states.append(normalized_state)
        actions.append(action_ids)

        # 执行动作
        full_state, done, info = env.step(expert_actions, random_actions)

        if done:
            print(f"  Episode ended at step {step}, winner: {info.get('winner')}")
            break

    return {
        "states": np.array(states),
        "actions": np.array(actions),
    }


def _direction_to_id(direction: Optional[str]) -> int:
    """将方向转换为动作 ID"""
    mapping = {None: 0, "up": 1, "down": 2, "left": 3, "right": 4}
    return mapping.get(direction, 0)


def train_imitation(
    model: PPOModel,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> float:
    """
    模仿学习训练步骤

    Args:
        states: (B, H, W) 状态矩阵
        actions: (B, num_players) 专家动作

    Returns:
        loss: 交叉熵损失
    """
    states = states.to(device)
    actions = actions.to(device)

    # 前向传播
    action_logits, _ = model(states)  # (B, num_players, num_actions)

    # 计算交叉熵损失（对每个玩家）
    loss = 0.0
    for player_idx in range(actions.shape[1]):
        player_logits = action_logits[:, player_idx, :]  # (B, num_actions)
        player_actions = actions[:, player_idx]  # (B,)
        loss += F.cross_entropy(player_logits, player_actions)

    loss = loss / actions.shape[1]  # 平均损失

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建环境
    env = MockEnvVNew(num_flags=9, seed=None)

    # 创建专家（L 队）
    expert = StrategicOpponent(team="L", env=env)

    # 创建工具
    converter = CTFMatrixConverter(width=env.width, height=env.height)
    normalizer = StateNormalizer(swap_ids=False)

    # 创建模型
    model = PPOModel(num_players=env.num_players, num_actions=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Checkpoint 目录
    checkpoints_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = 1000

    print(f"\nStarting imitation learning for {num_episodes} episodes...")
    print(f"Expert team: L (left), Opponent team: R (right, random)\n")

    for episode in range(num_episodes):
        # 收集专家轨迹
        trajectory = collect_expert_trajectory(
            env, expert, converter, normalizer, max_steps=500
        )

        if len(trajectory["states"]) == 0:
            print(f"Episode {episode}: No data collected, skipping")
            continue

        # 转换为 Tensor
        states = torch.from_numpy(trajectory["states"]).long()
        actions = torch.from_numpy(trajectory["actions"]).long()

        # 训练
        loss = train_imitation(model, optimizer, states, actions, device)

        if episode % 10 == 0:
            print(f"Episode {episode}: loss={loss:.4f}, steps={len(states)}")

        if episode % 100 == 0 and episode > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = checkpoints_dir / f"imitation_ep{episode}_{timestamp}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path.name}")


if __name__ == "__main__":
    main()
