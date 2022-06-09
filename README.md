# ML_DIFSIMS

Welcome to your new Dagster repository.

### Contents

| Name             | Description                                                                      |
|------------------|----------------------------------------------------------------------------------|
| `README.md`      | A description and guide for this code repository                                 |
| `workspace.yaml` | A file that specifies the location of the user code for Dagit and the Dagster CLI |
| `ml_difsims/`    | A Python directory that contains code for your Dagster repository                |
| `models/`        | A Python directory that contains crystal structures and orientation files       |
| `notebooks/`     | A Python directory that contains notebooks to process experimental data          |
| `writing/`       | A Python directory that contains reports of results, figures and references      |



## Getting up and running

1. Create a new Python environment and activate.

**Conda**
```bash
export PYTHON_VERSION=X.Y.Z
conda create --name difsims python=PYTHON_VERSION
conda activate difsims
```

2. Once you have activated your Python environment, install your repository as a Python package. By
using the `--editable` flag, `pip` will install your repository in
["editable mode"](https://pip.pypa.io/en/latest/reference/pip_install/?highlight=editable#editable-installs)
so that as you develop, local code changes will automatically apply.

```bash
pip install --editable .
```

## Local Development

1. Set the `DAGSTER_HOME` environment variable. Dagster will store run history in this directory.

```base
mkdir FULL_PATH_TO_PROJECT/dagster_logs
export DAGSTER_HOME=FULL_PATH_TO_PROJECT/dagster_logs
```

2. Start the [Dagit process](https://docs.dagster.io/overview/dagit). This will start a Dagit web
server that, by default, is served on http://localhost:3000.

```bash
set DAGSTER_HOME=FULL_PATH_TO_PROJECT/dagster_logs
dagit -f FULL_PATH_TO_PROJECT\ml_difsims\ml_difsims\simulators\large_scale_simulation_from_yaml.py"
```

3. (Optional) If you want to enable Dagster
[Schedules](https://docs.dagster.io/overview/schedules-sensors/schedules) or
[Sensors](https://docs.dagster.io/overview/schedules-sensors/sensors) for your jobs, start the
[Dagster Daemon process](https://docs.dagster.io/overview/daemon#main) **in a different shell or terminal**:

```bash
set DAGSTER_HOME=FULL_PATH_TO_PROJECT/dagster_logs
dagster-daemon run
```