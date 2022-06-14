# Import packages
import os, glob
from datetime import datetime

import hyperspy.api as hs
import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
import matplotlib.pyplot as plt

#%%
# Operations
from ml_difsims.utils.external_connects import connect_to_mongo_database


def get_diffraction_1d(link_to_cif, wavelength = 0.61992, plot=False):
    # Wavelength in amstrongs
    xrd_calculator = XRDCalculator(wavelength=wavelength)
    cif_file = Structure.from_file(link_to_cif)
    xrd = xrd_calculator.get_pattern(structure=cif_file,)

    def two_theta_to_a_inv(two_theta_array, waveln):
        # No 2 * pi factor
        q = (2. / waveln) * np.sin(np.deg2rad(two_theta_array) / 2.)
        return np.abs(q)

    q_sim = two_theta_to_a_inv(xrd.x, wavelength)

    # Set attributes
    setattr(xrd, 'q', q_sim)
    setattr(xrd, 'k', q_sim * 2 * np.pi)
    setattr(xrd, 'two_theta', xrd.x)

    if plot:
        plt.vlines(q_sim, ymax=xrd.y, ymin=0, label=f"{os.path.basename(link_to_cif).split('.')[0]}")
        plt.show()
        plt.legend()
    return xrd

def get_sim_diffraction_1d_dict(list_of_cif_files, phase_labels, **kwargs):
    # Returns a dictionary of each phase with its corresponding xrd diffraction pattern
    diffraction_dict = {}
    for cif, phase in zip(list_of_cif_files, phase_labels):
        xrf = get_diffraction_1d(cif, **kwargs)
        diffraction_dict[phase] = xrf

    return diffraction_dict

# Get mean diffraction pattern for each phase
def get_mean_diffraction_per_labelled_prediction(s_experimental, hspy_mask_predictions, phase_labels, q_axis):
    # Takes a hyperspy object with the signal axis labelled as its argmax phase (e.g. 0 or 1 or 2...)
    # Takes an array with the labels

    # Check s_predictions is a hyperspy object
    if not hasattr(hspy_mask_predictions, "_signal_type"):
        hspy_mask_predictions = hs.signals.Signal2D(hspy_mask_predictions)

    if not hasattr(s_experimental, "_signal_type"):
        s_experimental = hs.signals.Signal2D(s_experimental)

    mean_diffraction = []
    for i, phase in enumerate(phase_labels):
        phase_mask = hspy_mask_predictions.data == i
        phase_mask = phase_mask[:,:, np.newaxis]
        dp = s_experimental.data * phase_mask
        dp = hs.signals.Signal1D(dp).mean()
        mean_diffraction.append(dp)

    mean_diffraction = hs.stack(mean_diffraction)
    ax = mean_diffraction.axes_manager.signal_axes[0]
    ax.offset = q_axis[0]
    ax.scale = np.mean(np.diff(q_axis))
    ax.units = 'A^{-1}'
    ax.name = 'q'

    mean_diffraction.metadata.set_item("Phases", phase_labels)
    return mean_diffraction

# Get main peaks for each phase
def get_peaks(mean_diffraction):
    # Takes a hyperspy Signal1D object and finds the peaks
    # Returns a structured array containing fields (‘position’, ’width’, and ‘height’) for each peak.
    exp_peaks = mean_diffraction.find_peaks1D_ohaver(
        slope_thresh=0,
        amp_thresh=None,
        medfilt_radius=3,
        maxpeakn=25,
        parallel=True,)
    return exp_peaks



def calculate_error_from_closest_peak(peaks_sim, peaks_exp, method='x_pos_dif_error'):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    error = []
    # Iterate though each experimental peak position
    for p_exp in peaks_exp:
        if method == 'x_pos_dif_error':
            # Find peak index in sim closest to each of the experimental peaks
            x_sim = find_nearest(peaks_sim, p_exp)
            # Compute error in x position between experimental and simulated x position
            err = np.abs(x_sim - p_exp)
        else:
            raise NotImplementedError("Only the `x_pos_dif_error` method has been added.")
            # TODO: Add other algorithms to compare similarity/closeness between array values
            # https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9,
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html,
            # cosine similarity
        error.append(err)

    # Return the mean error
    return np.mean(error)

# Compare simulated with experimental peaks
def evaluate_peak_position_similarity(experimental_peaks, simulated_peaks_dict):
    # Takes an array of found peaks in the exp data for each phase and a dictionary with the xrd class object with the simulated pattern for each phase
    total_metric = []
    for i, (phase, xrd) in enumerate(simulated_peaks_dict.items()):
        peaks_sim = xrd.q
        peaks_exp = experimental_peaks[i]['position']
        error = calculate_error_from_closest_peak(peaks_sim, peaks_exp)
        total_metric.append(error)

    return total_metric, np.sum(total_metric)

#%%
def load_prediction_and_data_files(path_to_prediction_file, folder_with_exp_data, root):

    pred_dat = np.load(path_to_prediction_file)['pred']
    phase_labels = np.load(path_to_prediction_file)['phases']
    try:
        cnn_id = np.load(path_to_prediction_file)['cnn_id']
    except KeyError:
        cnn_id = None

    # Find the exp file that matches the prediction name file
    exp_files_paths = glob.glob(os.path.join(root, folder_with_exp_data, '*.npz'))
    general_name = os.path.basename(path_to_prediction_file).split('_probab')[0]

    exp_file_path = [f for f in exp_files_paths if general_name in f][0]
    q_axis = np.load(exp_file_path)['x']
    exp_dat = np.load(exp_file_path)['y']

    return pred_dat, exp_dat, phase_labels, q_axis, cnn_id


def convert_dat_to_mask(pred_dat, masking_method):
    """
    Takes a numpy array with the predictions as probabilities. Takes a masking method. Default is "argmax".
    Returns a hyperspy object with the signal axis labelled as its argmax phase (e.g. 0 or 1 or 2...)
    """
    if masking_method == "argmax":
        hspy_mask_pred = hs.signals.Signal2D(pred_dat.argmax(axis=-1))

    else:
        raise NotImplementedError("The `masking_method` is only implemented for 'argmax' method.")
    return hspy_mask_pred


def log_metrics(metric_per_phase, mean_metric, phase_labels, f_pred):
    print("---------")
    print("---------")
    print(f"Evaluation metrics for prediciton of: {os.path.basename(f_pred)}")
    print("Evaluation method: Peak X position difference")
    print("---------")
    for i, phase in enumerate(phase_labels):
        print(f"{phase} -- error {metric_per_phase[i]}")
    print("---------")
    print(f"Mean error -- {mean_metric}")
    return

def save_metadata_to_mongodb(metric_per_phase, mean_metric, phase_labels, f_pred, cnn_id):
    db_collection = connect_to_mongo_database('pred_evaluation', f'{cnn_id}')
    md = {}
    md['pred_evaluation_id'] = f'{cnn_id}'
    md['predicted_exp_file'] = str(os.path.basename(f_pred))
    md['evaluation_method'] = "peak_x_pos_difference"

    md["metric_per_phase"] = {}
    for i, phase in enumerate(phase_labels):
        md["metric_per_phase"][phase] = metric_per_phase[i]
    md["metric_mean"] = mean_metric
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md["timestamp"] = time_stamp

    db_collection.insert_one(md)
    return


def evaluate_data(f_pred, data_folder, cif_files_folder, root_path):
    """
    Takes both the raw experimental diffraction file and the predictions probability file.
    """

    # Load data
    pred_dat, exp_dat, phase_labels, q_axis, cnn_id = load_prediction_and_data_files(f_pred, data_folder, root_path)

    cif_files = [os.path.join(root_path, cif_files_folder, f"{phase}.cif") for phase in phase_labels]

    sim_dif_dict = get_sim_diffraction_1d_dict(cif_files, phase_labels,)

    pred_masks = convert_dat_to_mask(pred_dat, masking_method = "argmax")
    # make sure it is a hyperspy object when needed
    mean_dp_per_label = get_mean_diffraction_per_labelled_prediction(exp_dat, pred_masks, phase_labels, q_axis)
    peaks = get_peaks(mean_dp_per_label)

    metric_per_phase, mean_metric = evaluate_peak_position_similarity(peaks, sim_dif_dict)
    log_metrics(metric_per_phase, mean_metric, phase_labels, f_pred)
    save_metadata_to_mongodb(metric_per_phase, mean_metric, phase_labels, f_pred, cnn_id)
    return mean_metric

