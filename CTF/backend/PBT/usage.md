• 已在 CTF/backend/PBT/ 按 DETAIL_PLAN.md 落地 PPO 训练核心组件（基于现有 mock_env_vnew.py + lib/matrix_util.py）：

  - 新增环境适配层：CTF/backend/PBT/ppo_env_adapter.py:1
  - 新增状态标准化：CTF/backend/PBT/state_normalizer.py:1
  - 新增 PPO Actor-Critic 模型（自动 one-hot 到 20 通道）：CTF/backend/PBT/ppo_model.py:1
  - 新增 PPO 训练器（GAE + clipped loss）：CTF/backend/PBT/ppo_trainer.py:1
  - 新增训练入口（随机对手 self-play baseline + checkpoints）：CTF/backend/PBT/train_ppo.py:1

  本环境的 python3 缺 numpy/torch，所以只能做语法编译检查；你本机依赖已装好时用这些命令验证/运行：

  - cd CTF/backend/PBT && python -c "from ppo_env_adapter import PPOEnvAdapter;
    print(PPOEnvAdapter(team='L').reset().shape)"
  - cd CTF/backend/PBT && python -c "import torch; from ppo_model import PPOModel; m=PPOModel();
    print(m(torch.zeros(1,20,20,dtype=torch.long))[0].shape)"
  - cd CTF/backend/PBT && python train_ppo.py


## Terminal mock-env 外层套件（用于逻辑一致性验证）

`terminal_suite.py` 会直接驱动 `mock_env_vnew.py` 的 `init/status/finished` 循环，并在终端渲染网格（墙/障碍/目标区/监狱/玩家/旗帜），方便和 `CTF/frontend` 的表现做对照。

### 运行

- L0 键盘控制 + R 随机对手：
  - `cd CTF/backend/PBT && python3 terminal_suite.py --l keyboard --r random`
- 自动播放（每回合延迟 120ms）：
  - `cd CTF/backend/PBT && python3 terminal_suite.py --delay-ms 120`

快捷键：`space` 暂停/继续，`n` 单步，`q` 退出；键盘控制时 `WASD -> L0`，`IJKL -> R0`。

### 接入现有 AI（无 WebSocket，本地直调用）

如果你的 AI 脚本暴露了与 WebSocket backend 一致的函数：

- `start_game(req)`
- `plan_next_actions(req)`（返回 `{player_name: direction}` 或 `{"players": {...}}`）
- 可选：`game_over(req)`

那么可以用 `module:` 直接接入，例如：

- `cd CTF/backend/PBT && python3 terminal_suite.py --l module:pick_flag_ai --r module:pick_closest_flag`

### 接入任意程序（子进程 JSONL 协议）

也可以把控制器作为子进程启动：父进程按行发送 JSON（`init/status/finished`），子进程按行返回 JSON（`{"players": {...}}` 或直接 `{...}`）。

- `cd CTF/backend/PBT && python3 terminal_suite.py --l 'cmd:["python3","my_bot.py"]' --r random`
