# DIFFSIMS XL

Welcome to diffsims XL, a platform to simulate large amounts of electron diffraction data using Dagster.
Continue reading to get started!

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


# General comments on I/O file formats

**Simulation**

I have now create 6.6GB of data (approximately it is 3GB of actual data, as I also saved the full simulated dataset. This should be around 2.5M datapoints

They are saved in the shared_space/jf631/simulations folder as .npz  files (compressed numpy arrays). Each of these files has the following keys:
- ['1d']['x_all'] and ['1d']['y_all'] for the full dataset (do not use this)
- ['1d']['x_px'] and ['1d']['y_px'] for the cropped and processed dataset (use this to train). In the x_px you have the actual (300k, 147) data arrays (300k points, each 147 datapoints). In the y_px you have the labelled data (300k, 1) as categorical data (either being 0 for phase 1, 1 for phase 2....). You can convert it to one hot notation if needed.
- 
You can load these arrays using the np.load(path_to_npz)['1d'][x_px]

In the simulation files you also have some useful metadata (it would be great to copy it also as metadata to any of your output files):
- ['1d']['x_all_q_axis'] has the calibration array for the x axis.
- ['metadata']['phases'] has the name of what phase A, B ... are
- ['metadata']['id'] is the unique id (uuid4.id) that links to the simulation dataset


**Experimental files**

I have now processed 3 different experimental files to be easily integrated in your cnn to predict their phase once trained.
These files you can find in the shared_space/jf631/experimental folder as .npz files too.

These files contain the following keys:
- ['x'] has the calibrated x axis (not needed for the CNN, it's more of a metadata)
- ['y'] has the actual data (the diffraction pattern of shape (x, y, 147) . Note that it is an array of rank 3, meaning that it is a map of datapoints. I think you will need to reshape it to (x*y, 147) shape before inputing it into the predict function in tensorflow. It would be great if you could reshape it back after the prediction to the original 3D array shape.

In short, for each CNN model you train, you can predict 3 different experimental files (so create 3 different predicted datasets).

**CNN and predictions**

It would be nice to have a unique identifier for each neural network you create and train, so it is easy to trace back. I would normally use the uuid4.id python function.

Once you have some models trained, could you make sure to save the output prediction files as follows:

- Save each predicted file for each experimental datafile (see above) as  a .npz file with the name *_probabilty_pred.npz so the name has the original experimental file name in the *) (e.g. roi_3_rebin_radial_crop_probability_pred.npz if the original file is roi_3_rebin_radial_crop.npz).
- 
      Each npz file should contain the following keys:
        - the predictions for each experimental file as a shape of (y, x, prob_n_phases) saved as ['pred'] key
        -- the predicted labels (you can find them from the metadata in the simulation files) saved as ['phases'] key
        -- the unique uuid4 for the neural network that was used to get the predictions, saved as ['cnn_id'] key.

If you save the prediction for each experimental file in this way, then we can run the evaluation pipeline automatically after predictions (see next section).

**Evaluation pipeline**

Apart from the training/testing metrics that can be produced from the labelled (fully simulated) data (such as the confusion metrics, loss...), I have now also created a script that can quantify how well a prediction model is working on experimental data (which is never labelled).

For that, I have created the following script (https://github.com/jordiferrero/ml_difsims/blob/master/ml_difsims/predictors/predictions_evaluator.py). If you run the function  evaluate_all() it will find peaks and compare the predicted phases to simulated one and give an error on the difference in their peak positions.

The details don't matter that much. The only thing I want to clarify is what type of file it needs for this evaluator to work:
Input files it takes:
- A list of the paths to all the predicted *_probabilty_pred.npz numpy array files (the name needs to have the original experimental file name in the *). Each npz file should contain the 'pred', 'phases' and 'cnn_id' keys (see above)
- The relative path to the folder were the original experimental *_radial_crop.npz files are (with the original experimental file name in the *)
- The relative path to the folder where the .cif files are (these are the structure files required to simulate diffraction patterns). I have included all these files in my GitHub but also have uploaded them into the shared_space/jf631/models/crystal_files
I think it would be great if you could include this final evaluation pipeline after you train each model, so we can get a quantitative idea of how the model is performing on real experimental data.
