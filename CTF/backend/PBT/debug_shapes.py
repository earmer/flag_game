import torch
import numpy as np
from ppo_env_adapter import PPOEnvAdapter
from ppo_model import PPOModel

# Create environment and model
env = PPOEnvAdapter(team="L")
model = PPOModel(num_players=env.num_players, num_actions=env.num_actions)

# Reset environment
state = env.reset()
print(f"State shape from env.reset(): {state.shape}")
print(f"State dtype: {state.dtype}")

# Convert to tensor
state_tensor = torch.from_numpy(state).unsqueeze(0)
print(f"State tensor shape: {state_tensor.shape}")
print(f"State tensor dtype: {state_tensor.dtype}")

# Try forward pass
try:
    action_logits, value = model(state_tensor)
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Value shape: {value.shape}")
    print("Forward pass successful!")
except Exception as e:
    print(f"Error during forward pass: {e}")
    import traceback
    traceback.print_exc()
