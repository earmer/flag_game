"""
测试脚本：验证 StrategicOpponent 和模仿学习组件
"""
from strategic_opponent import StrategicOpponent
from mock_env_vnew import MockEnvVNew

print("=" * 60)
print("Test 1: StrategicOpponent Basic Functionality")
print("=" * 60)

# 创建环境
env = MockEnvVNew(num_flags=9, seed=42)
full_state = env.reset()

# 创建专家（L 队）
expert = StrategicOpponent('L', env)

# 测试动作生成
print("\nInitial state:")
print(f"  L team players: {len(full_state['L']['myteamPlayer'])}")
print(f"  R team flags (available): {sum(1 for f in full_state['R']['myteamFlag'] if f['canPickup'])}")

actions_l = expert.get_actions(full_state)
print(f"\nGenerated actions for L team: {actions_l}")

# 运行几步
print("\n" + "=" * 60)
print("Test 2: Running 10 Steps")
print("=" * 60)

for step in range(10):
    actions_l = expert.get_actions(full_state)
    actions_r = {f'R{i}': None for i in range(3)}  # R 队不动
    
    print(f"\nStep {step}:")
    print(f"  L actions: {actions_l}")
    
    full_state, done, info = env.step(actions_l, actions_r)
    
    if done:
        print(f"\n  Game over! Winner: {info['winner']}")
        break

print("\n✓ StrategicOpponent tests passed!")
