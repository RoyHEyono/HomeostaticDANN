import numpy as np
import json

# Define ranges for random parameters (log scale!)
lr_min, lr_max = -3, 0# 1e-3, 1e-1
hidden_layer_width_min, hidden_layer_width_max = 100, 500

# Number of random configurations
num_random_configs = 100

# Generate random configurations
random_configs = []
for _ in range(num_random_configs):
    config = {
        'lr': 10 ** np.random.uniform(lr_min, lr_max)
    }
    random_configs.append(config)

# Save to file
with open('random_nondann_configs.json', 'w') as f:
    json.dump(random_configs, f)
