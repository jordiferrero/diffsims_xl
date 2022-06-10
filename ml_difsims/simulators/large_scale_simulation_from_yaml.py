import os.path

import tqdm
from dagster import job, op, static_partitioned_config
from ml_difsims.simulators.simulator_diffsims_dagster_ops import *
import json

from paramspace import yaml

root = os.path.abspath(r"C:\Users\Sauron\Documents\GitHub\strankslab\ml_difsims")
path_to_save_jsons = os.path.join(root, "data\simulations\json_files")

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
    JSON_FILES.append(json.dumps(params))


@static_partitioned_config(partition_keys=JSON_FILES)
def json_md_config(partition_key: str):
    return {'ops': {'simulate_diffraction': {'ops': {'load_json_metadata_to_dict': {'config': {'json_string': partition_key}}}}},
            "execution": {"config": {"multiprocess": {"max_concurrent": 10,}}}
            }


# Write the job
@job(op_retry_policy=default_policy, config=json_md_config)
def simulate_diffraction_partitioned():
    simulate_diffraction()


if __name__ == "__main__":
    result = simulate_diffraction_partitioned.execute_in_process()


