# train_ppo.py Debug Summary

## Issues Found and Fixed

### 1. Missing PyTorch Dependency ✅ FIXED
**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:** Installed PyTorch using:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. CNN Dimension Mismatch ✅ FIXED
**Problem:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x4096 and 6400x512)`

**Root Cause:** The `fc_hidden` layer in `ppo_model.py` expected the wrong input dimension.

**Analysis:**
- Input state: 20x20
- After one-hot encoding: (B, 20, 20, 20) - 20 channels
- After conv1 (kernel=3, padding=1): (B, 64, 20, 20)
- After MaxPool2d(2): (B, 64, 10, 10)
- After conv2 (kernel=3, padding=1): (B, 128, 10, 10)
- After conv3 (kernel=3, stride=2, no padding): (B, 256, 4, 4)
  - Formula: floor((10 - 3) / 2) + 1 = 4
- After flatten: 256 × 4 × 4 = **4096 features**

**Solution:** Changed line 27 in `ppo_model.py`:
```python
# Before:
self.fc_hidden = nn.Linear(256 * 5 * 5, 512)  # 6400 -> 512

# After:
self.fc_hidden = nn.Linear(256 * 4 * 4, 512)  # 4096 -> 512
```

## Training Status

✅ **Training is now running successfully!**

Sample output:
```
Episode 0: reward=0.00 loss=-0.0161 policy=-0.0000 value=0.0000 ent=1.6091
Episode 10: reward=0.00 loss=-0.0159 policy=-0.0000 value=0.0000 ent=1.5937
Episode 20: reward=0.00 loss=-0.0154 policy=0.0000 value=0.0007 ent=1.5759
```

## How to Run

```bash
cd CTF/backend/PBT
python train_ppo.py
```

The script will:
- Train for 1000 episodes
- Print metrics every 10 episodes
- Save checkpoints every 100 episodes to `checkpoints/ppo_ep{episode}.pt`

## Files Modified

1. `ppo_model.py` - Fixed CNN dimension calculation (line 27)

## Debug Scripts Created

1. `debug_shapes.py` - Tests the model forward pass with environment states
2. `calc_dims.py` - Calculates CNN output dimensions step-by-step
