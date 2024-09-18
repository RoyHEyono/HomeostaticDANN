import numpy as np
import json
import wandb
import re


api = wandb.Api(timeout=19)

def fetch_runs(api, entity, project_name, filters, order=None):
    if order:
        runs = api.runs(f"{entity}/{project_name}", filters=filters, order=order)
    else:
        runs = api.runs(f"{entity}/{project_name}", filters=filters)
    #print(f"Runs for project '{project_name}':")
    return runs

EI = fetch_runs(api, entity='project_danns', project_name='Luminosity_DeepDANN_ExhaustiveSearch', filters={"config.dataset": "fashionmnist", 
                                                                        "config.brightness_factor": 0.75, "config.homeostasis": 1, "config.normtype": 0,
                                                                        "config.implicit_homeostatic_loss": None, "config.use_testset": True, "config.task_opt_inhib": 1, "config.use_sep_bias_gain_lrs": 0,
                                                                        "config.homeostatic_annealing": 0 , "config.lambda_homeo": 300}, order="-summary_metrics.test_acc")

EI = EI[:10]

# Regex pattern to extract wei and wix
pattern = r'wei=([-\d.eE]+), wix=([-\d.eE]+)'



# Define ranges for random parameters (log scale!)
lambda1_min, lambda1_max = -2, 2 #1e-5, 1e-2
lambda2_min, lambda2_max = -1, 1 #1e-5, 1e-2

# Number of random configurations
num_random_configs = 10

# Generate random configurations
random_configs = []
for run in EI:
    match = re.search(pattern, run.config['inhib_lrs'])

    if match:
        wei = float(match.group(1))
        wix = float(match.group(2))

    for _ in range(num_random_configs):
        config = {
            'lr': run.config['lr'],
            'lr_wei': wei,
            'lr_wix': wix,
            'hidden_layer_width': run.config['hidden_layer_width'],
            'lambda1': 10 ** np.random.uniform(lambda1_min, lambda1_max),
            'lambda2': 10 ** np.random.uniform(lambda2_min, lambda2_max),
        }
        random_configs.append(config)

# Save to file
with open('random_configs_decoupled.json', 'w') as f:
    json.dump(random_configs, f)
