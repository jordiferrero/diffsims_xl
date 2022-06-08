import os.path
import shutil

import tqdm
from dagster import job, op, static_partitioned_config
from ml_difsims.simulators.simulator_diffsims_dagster_ops import *
import json

from paramspace import yaml

root = os.path.abspath(r"C:\Users\Sauron\Documents\GitHub\strankslab\ml_difsims")
path_to_save_jsons = os.path.join(root, "data\simulations\json_files")

# Create folder
save_folder_path = os.path.join(root, path_to_save_jsons)
if os.path.exists(save_folder_path):
    shutil.rmtree(save_folder_path)

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

yaml_path = os.path.join(root, "ml_difsims/simulators/param_space_yaml.yml")
with open(yaml_path, mode='r') as cfg_file:
    cfg = yaml.load(cfg_file)

# Get the ParamSpace object and print some information
pspace = cfg['sim_params']
print("Received parameter space with volume", pspace.volume)
print(pspace.get_info_str())

# Now perform the iteration and run the simulations
print("Starting simulation")
JSON_FILES = []
for i, params in tqdm.tqdm(enumerate(pspace)):
    fname = f'json_{i:03}.json'
    fpath = os.path.join(path_to_save_jsons, fname)
    with open(fpath, 'w') as f:
        json.dump(params, f)
    JSON_FILES.append(json.dumps(params))

print(len(JSON_FILES))
print(JSON_FILES[-1])
