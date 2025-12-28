"""
测试脚本：验证轨迹收集功能
"""
import sys
from pathlib import Path

from train_imitation import collect_expert_trajectory
from strategic_opponent import StrategicOpponent
from mock_env_vnew import MockEnvVNew
from state_normalizer import StateNormalizer

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from lib.matrix_util import CTFMatrixConverter

print("=" * 60)
print("Test: Expert Trajectory Collection")
print("=" * 60)

# 创建环境和组件
env = MockEnvVNew(num_flags=9, seed=42)
expert = StrategicOpponent('L', env)
converter = CTFMatrixConverter(width=env.width, height=env.height)
normalizer = StateNormalizer(swap_ids=False)

# 收集轨迹
print("\nCollecting expert trajectory (max 50 steps)...")
traj = collect_expert_trajectory(env, expert, converter, normalizer, max_steps=50)

print(f"\nResults:")
print(f"  States shape: {traj['states'].shape}")
print(f"  Actions shape: {traj['actions'].shape}")
print(f"  Total steps collected: {len(traj['states'])}")

print(f"\nSample actions (first 5 steps):")
for i, actions in enumerate(traj['actions'][:5]):
    action_names = []
    for action_id in actions:
        names = {0: 'None', 1: 'up', 2: 'down', 3: 'left', 4: 'right'}
        action_names.append(names[action_id])
    print(f"  Step {i}: {action_names}")

print("\n✓ Trajectory collection test passed!")
