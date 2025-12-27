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