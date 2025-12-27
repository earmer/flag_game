import torch
import torch.nn as nn

# Simulate the forward pass
input_size = (1, 20, 20, 20)  # B, C, H, W

print("Input:", input_size)

# Conv1: kernel=3, padding=1, no stride
# Output: same size = (1, 64, 20, 20)
print("After conv1: (1, 64, 20, 20)")

# MaxPool2d(2)
# Output: (1, 64, 10, 10)
print("After pool: (1, 64, 10, 10)")

# Conv2: kernel=3, padding=1
# Output: (1, 128, 10, 10)
print("After conv2: (1, 128, 10, 10)")

# Conv3: kernel=3, stride=2, no padding
# Output: floor((10 - 3) / 2) + 1 = floor(7/2) + 1 = 3 + 1 = 4
# So: (1, 256, 4, 4)
print("After conv3: (1, 256, 4, 4)")

# Flatten
# Output: 256 * 4 * 4 = 4096
print("After flatten: 4096")
print("\nThe fc_hidden layer should expect 4096 input features, not 6400")
