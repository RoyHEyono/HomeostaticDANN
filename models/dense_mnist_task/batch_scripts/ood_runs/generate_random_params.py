import numpy as np
import json

# Define ranges for random parameters (log scale!)
lr_min, lr_max = -3, -2# 1e-3, 1e-1
lr_wei_min, lr_wei_max = -5, -2 #1e-5, 1e-2
lr_wix_min, lr_wix_max = -2, 0 #1e-2, 1e0
hidden_layer_width_min, hidden_layer_width_max = 100, 500
bf_min, bf_max = 0, 0.75
bf_eval_min, bf_max_eval = -1, 1


# Number of random configurations
num_random_configs = 100

# Generate random configurations
random_configs = []
for _ in range(num_random_configs):
    config = {
        'lr': 10 ** np.random.uniform(lr_min, lr_max),
        'lr_wei': 10 ** np.random.uniform(lr_wei_min, lr_wei_max),
        'lr_wix': 10 ** np.random.uniform(lr_wix_min, lr_wix_max),
        'hidden_layer_width': int(np.random.uniform(hidden_layer_width_min, hidden_layer_width_max)),
        'brightness_factor': round(np.random.uniform(bf_min, bf_max),2),
        'brightness_factor_eval': round(np.random.uniform(bf_eval_min, bf_max_eval),2)
    }
    random_configs.append(config)

# Save to file
with open('random_configs.json', 'w') as f:
    json.dump(random_configs, f)
