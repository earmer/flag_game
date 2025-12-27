# 基础 PPO 训练系统详细实现计划

> **更新**: 基于现有 `mock_env_vnew.py` 环境，专注于 PPO 核心组件实现

---

## 现有资源

### ✅ 已有组件

1. **`mock_env_vnew.py`** - 完整的游戏模拟器
   - 实现了 Phaser 前端的所有游戏逻辑
   - 支持 Self-Play（双方同时行动）
   - API: `reset()`, `step(actions_l, actions_r)`, `get_team_status(team)`
   - 输出标准的 WebSocket 格式状态

2. **`lib/matrix_util.py`** - 状态矩阵转换器
   - `CTFMatrixConverter` - 将 JSON 状态转换为 (20, 20) 矩阵
   - 20 种实体类型编码

3. **`lib/game_engine.py`** - 游戏引擎工具类
   - `GameMap` - 地图管理、路径规划

---

## 实现计划

### Stage 1: 环境适配层

#### [NEW] [ppo_env_adapter.py](file:///c:/Users/Earmer/flag_game/CTF/backend/PBT/ppo_env_adapter.py)

**目标**: 将 `mock_env_vnew.py` 包装为 RL 训练友好的接口

```python
class PPOEnvAdapter:
    """PPO 训练环境适配器"""
    
    def __init__(self, team='L'):
        self.env = MockEnvVNew(num_flags=9, seed=None)
        self.team = team  # 当前训练的队伍
        self.opponent_team = 'R' if team == 'L' else 'L'
        
        self.converter = CTFMatrixConverter()
        self.normalizer = StateNormalizer()  # 地图标准化
        
    def reset(self) -> np.ndarray:
        """重置环境，返回标准化的观测矩阵"""
        full_state = self.env.reset()
        
        # 初始化静态地图
        init_payload = self.env.get_init_payload(self.team)
        self.converter.initialize_static_map(init_payload)
        
        # 获取当前状态
        status = full_state[self.team]
        state_matrix = self.converter.convert_to_matrix(status)
        
        # 地图标准化（己方始终在左侧）
        normalized = self.normalizer.normalize_state(state_matrix, self.team)
        return normalized
    
    def step(self, actions: np.ndarray, opponent_actions: Dict[str, str]) -> Tuple:
        """
        执行一步
        Args:
            actions: (3,) 数组，每个玩家的动作 [0-4]
            opponent_actions: 对手动作字典
        Returns:
            (next_state, reward, done, info)
        """
        # 转换动作格式
        action_map = {0: None, 1: 'up', 2: 'down', 3: 'left', 4: 'right'}
        my_actions = {}
        for i, action_id in enumerate(actions):
            player_name = f"{self.team}{i}"
            my_actions[player_name] = action_map[action_id]
        
        # 反标准化动作（如果是 R 队）
        my_actions = self.normalizer.denormalize_actions(my_actions, self.team)
        
        # 执行环境步进
        if self.team == 'L':
            full_state, done, info = self.env.step(my_actions, opponent_actions)
        else:
            full_state, done, info = self.env.step(opponent_actions, my_actions)
        
        # 获取标准化状态
        status = full_state[self.team]
        state_matrix = self.converter.convert_to_matrix(status)
        normalized = self.normalizer.normalize_state(state_matrix, self.team)
        
        # 计算奖励
        reward = self.compute_reward(full_state, done, info)
        
        return normalized, reward, done, info
    
    def compute_reward(self, full_state, done, info) -> float:
        """计算即时奖励（详见 reward.py）"""
        # 简化版，后续在 reward.py 中实现
        my_score = full_state[self.team]['myteamScore']
        opp_score = full_state[self.opponent_team]['myteamScore']
        
        reward = 0.0
        if done:
            reward = 100.0 if info['winner'] == self.team else -100.0
        else:
            reward = (my_score - opp_score) * 10.0
        
        return reward
```

---

### Stage 2: 状态标准化工具

#### [NEW] [state_normalizer.py](file:///c:/Users/Earmer/flag_game/CTF/backend/PBT/state_normalizer.py)

**目标**: 实现地图标准化（复用架构文档设计）

```python
class StateNormalizer:
    """状态标准化器 - 始终将己方放在左侧"""
    
    def normalize_state(self, state_matrix: np.ndarray, team_name: str) -> np.ndarray:
        """
        标准化输入状态
        Args:
            state_matrix: (20, 20) 原始状态矩阵
            team_name: "L" 或 "R"
        Returns:
            normalized_state: (20, 20) 标准化后的状态
        """
        if team_name == "R":
            # 左右翻转整个地图
            state_matrix = np.flip(state_matrix, axis=1)
            
            # 交换 L/R 实体 ID
            # 00-05 (我方) <-> 06-11 (敌方)
            # 13-14 (我方 Home) <-> 15 (敌方 Home)
            # 18 (我方旗帜) <-> 19 (敌方旗帜)
            id_swap_map = {
                0: 6, 1: 7, 2: 8,
                3: 9, 4: 10, 5: 11,
                6: 0, 7: 1, 8: 2,
                9: 3, 10: 4, 11: 5,
                13: 15, 14: 15,  # Home 映射
                15: 13,
                18: 19,
                19: 18,
            }
            
            # 应用 ID 映射
            new_matrix = state_matrix.copy()
            for old_id, new_id in id_swap_map.items():
                new_matrix[state_matrix == old_id] = new_id
            
            return new_matrix
        
        return state_matrix
    
    def denormalize_actions(self, actions: Dict[str, str], team_name: str) -> Dict[str, str]:
        """
        将标准化的动作映射回真实坐标系
        Args:
            actions: {player_name: direction}
            team_name: "L" 或 "R"
        Returns:
            real_actions: 真实坐标系下的动作
        """
        if team_name == "R":
            action_map = {
                'left': 'right',
                'right': 'left',
                'up': 'up',
                'down': 'down',
                None: None,
            }
            return {name: action_map[act] for name, act in actions.items()}
        
        return actions
```

---

### Stage 3: PPO 神经网络

#### [NEW] [ppo_model.py](file:///c:/Users/Earmer/flag_game/CTF/backend/PBT/ppo_model.py)

**目标**: 轻量级 CNN 策略网络

```python
import torch
import torch.nn as nn

class PPOModel(nn.Module):
    """PPO Actor-Critic 网络"""
    
    def __init__(self, input_channels=20, num_players=3, num_actions=5):
        super().__init__()
        
        # CNN Backbone
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        
        # 计算展平后的维度: 20x20 -> 10x10 (pool) -> 5x5 (stride=2) -> 256*5*5
        self.fc_hidden = nn.Linear(256 * 5 * 5, 512)
        
        # Actor Head (3个玩家，每个5个动作)
        self.actor = nn.Linear(512, num_players * num_actions)
        
        # Critic Head
        self.critic = nn.Linear(512, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, state):
        """
        Args:
            state: (B, 20, 20) 或 (B, 1, 20, 20)
        Returns:
            action_logits: (B, 3, 5)
            value: (B, 1)
        """
        # 确保输入是 4D
        if state.dim() == 3:
            state = state.unsqueeze(1)  # (B, 1, 20, 20)
        
        # CNN 特征提取
        x = self.relu(self.conv1(state))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # 展平
        x = self.flatten(x)
        x = self.relu(self.fc_hidden(x))
        
        # Actor: (B, 15) -> (B, 3, 5)
        action_logits = self.actor(x).view(-1, 3, 5)
        
        # Critic: (B, 1)
        value = self.critic(x)
        
        return action_logits, value
```

---

### Stage 4: PPO 训练器

#### [NEW] [ppo_trainer.py](file:///c:/Users/Earmer/flag_game/CTF/backend/PBT/ppo_trainer.py)

**目标**: 实现 PPO 算法核心

```python
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(self, model, lr=3e-4, clip_eps=0.2, gamma=0.99, lam=0.95):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        
        self.value_coef = 0.5
        self.entropy_coef = 0.01
    
    def compute_gae(self, rewards, values, dones):
        """计算 GAE 优势估计"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def train_step(self, states, actions, old_log_probs, advantages, returns):
        """一次 PPO 更新"""
        # Forward pass
        action_logits, values = self.model(states)
        
        # 计算新的 log_probs
        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)  # (B,)
        
        # PPO Clipped Loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value Loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy Bonus
        entropy = dist.entropy().mean()
        
        # Total Loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }
```

---

### Stage 5: 训练入口

#### [NEW] [train_ppo.py](file:///c:/Users/Earmer/flag_game/CTF/backend/PBT/train_ppo.py)

**目标**: Self-Play 训练循环

```python
import torch
import numpy as np
from ppo_env_adapter import PPOEnvAdapter
from ppo_model import PPOModel
from ppo_trainer import PPOTrainer

def collect_trajectory(env, model, max_steps=500):
    """收集一条轨迹"""
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
    
    state = env.reset()
    
    for _ in range(max_steps):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value = model(state_tensor)
            dist = torch.distributions.Categorical(logits=action_logits[0])
            action = dist.sample()  # (3,)
            log_prob = dist.log_prob(action).sum()
        
        # 对手使用随机策略（初期）
        opponent_actions = {f"R{i}": np.random.choice(['up', 'down', 'left', 'right', None]) 
                           for i in range(3)}
        
        next_state, reward, done, info = env.step(action.numpy(), opponent_actions)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value.item())
        log_probs.append(log_prob.item())
        dones.append(done)
        
        state = next_state
        
        if done:
            break
    
    return {
        'states': np.array(states),
        'actions': torch.stack(actions),
        'rewards': rewards,
        'values': values,
        'log_probs': torch.tensor(log_probs),
        'dones': dones,
    }

def main():
    # 创建环境和模型
    env = PPOEnvAdapter(team='L')
    model = PPOModel()
    trainer = PPOTrainer(model)
    
    num_episodes = 1000
    
    for episode in range(num_episodes):
        # 收集轨迹
        traj = collect_trajectory(env, model)
        
        # 计算优势
        advantages = trainer.compute_gae(
            traj['rewards'], 
            traj['values'], 
            traj['dones']
        )
        returns = advantages + torch.tensor(traj['values'])
        
        # PPO 更新
        states = torch.from_numpy(traj['states']).float()
        metrics = trainer.train_step(
            states,
            traj['actions'],
            traj['log_probs'],
            advantages,
            returns
        )
        
        # 日志
        if episode % 10 == 0:
            total_reward = sum(traj['rewards'])
            print(f"Episode {episode}: Reward={total_reward:.2f}, Loss={metrics['loss']:.4f}")
        
        # 保存模型
        if episode % 100 == 0:
            torch.save(model.state_dict(), f'checkpoints/ppo_ep{episode}.pt')

if __name__ == '__main__':
    main()
```

---

## 文件结构

```
CTF/backend/PBT/
├── mock_env_vnew.py           # ✅ 已有 - 游戏模拟器
├── AI_STUCTURE_PLAN.md        # ✅ 已有 - 架构文档
├── DETAIL_PLAN.md             # 📝 本文件
│
├── ppo_env_adapter.py         # 🆕 环境适配层
├── state_normalizer.py        # 🆕 状态标准化
├── ppo_model.py               # 🆕 PPO 网络
├── ppo_trainer.py             # 🆕 PPO 训练器
├── train_ppo.py               # 🆕 训练入口
│
└── checkpoints/               # 🆕 模型保存目录
```

---

## 实现顺序

1. **`state_normalizer.py`** - 最基础的工具类
2. **`ppo_model.py`** - 独立的网络定义
3. **`ppo_env_adapter.py`** - 环境适配（依赖 normalizer）
4. **`ppo_trainer.py`** - 训练算法（依赖 model）
5. **`train_ppo.py`** - 训练入口（整合所有组件）

---

## 验证计划

### 单元测试
```bash
# 测试环境适配
python -c "from ppo_env_adapter import PPOEnvAdapter; env = PPOEnvAdapter(); print(env.reset().shape)"

# 测试模型前向传播
python -c "from ppo_model import PPOModel; import torch; m = PPOModel(); print(m(torch.randn(1, 20, 20))[0].shape)"
```

### 训练测试
```bash
# 启动训练（1000 episodes）
python train_ppo.py
```

**观察指标**:
- Episode reward 趋势
- Policy loss 下降
- Entropy 适度下降（不要太快归零）

---

## 关键设计决策

> [!IMPORTANT]
> ### 复用现有 mock_env_vnew.py
> 不需要重新实现环境，只需要适配层将其包装为 RL 友好的接口。

> [!NOTE]
> ### 地图标准化策略
> 在 `ppo_env_adapter.py` 中实现，确保无论训练哪个队伍，AI 输入始终将己方映射到左侧。

> [!TIP]
> ### 初期对手策略
> 先用**对手随机移动**训练，验证 Pipeline 正常工作后再引入 Self-Play。
