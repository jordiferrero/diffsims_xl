#%% md
# Predicting phases for experimental dataset
# For the large filter dataset
#%%
# Import packages
import os, glob
import hyperspy.api as hs
import numpy as np
from diffsims.utils.sim_utils import get_electron_wavelength
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
import matplotlib.pyplot as plt

#%%
# Operations

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
def get_mean_diffraction_per_labelled_prediction(s_experimental, s_predictions, phase_labels, q_axis):
    # Takes a hyperspy object with the signal axis labelled as its argmax phase (e.g. 0 or 1 or 2...)
    # Takes an array with the labels

    # Check s_predictions is a hyperspy object
    if not hasattr(s_predictions, "_signal_type"):
        s_predictions = hs.signals.Signal2D(s_predictions)

    mean_diffraction = []
    for i, phase in enumerate(phase_labels):
        phase_mask = s_predictions.data == i
        dp = s_experimental.data * phase_mask
        dp = hs.signals.Signal1D(np.mean(dp))
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
        medfilt_radius=2,
        maxpeakn=25,
        parallel=True,)
    return exp_peaks


# Compare simulated with experimental peaks
def evaluate_peak_position_similarity(experimental_peaks, simulated_peaks_dict):
    # Takes an array of found peaks in the exp data for each phase and a dictionary with the xrd class object with the simulated pattern for each phase
    total_metric = []
    for i, (phase, xrd) in enumerate(simulated_peaks_dict.items()):
        print(f"Evaluating phase {phase}")
        xs_sim = xrd.q
        xs_exp = experimental_peaks[i]['position']
        # Algorithm to compare similarity/closeness between array values
        # https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9, https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html, cosine similarity
        # metric = f()
        #total_metric.append(metric)

    return np.sum(total_metric)

#%%
def load_prediction_and_data_files(path_to_prediction_file, folder_with_exp_data, root):

    pred_dat = np.load(path_to_prediction_file)['pred']
    phase_labels = np.load(path_to_prediction_file)['phase_list']

    # Find the exp file that matches the prediction name file
    exp_files_paths = glob.glob(os.path.join(root, folder_with_exp_data, '*npz'))
    general_name = os.path.basename(path_to_prediction_file).split('probab')[0]

    exp_file_path = [f for f in exp_files_paths if general_name in f][0]
    q_axis = np.load(exp_file_path)['x']
    exp_dat = np.load(exp_file_path)['y']

    return pred_dat, exp_dat, phase_labels, q_axis


def evaluate_data(f_pred, data_folder, cif_files_folder, root_path):
    """
    Takes both the raw experimental diffraction file and the predictions probability file.
    """

    # Load data
    pred_dat, exp_dat, phase_labels, q_axis = load_prediction_and_data_files(f_pred, data_folder, root_path)

    cif_files = [os.path.join(root_path, cif_files_folder, f"{phase}.cif") for phase in phase_labels]

    sim_dif_dict = get_sim_diffraction_1d_dict(cif_files, phase_labels,)

    # TODO: see below
    # Select which method to use (to convert to one-hot notation)
    # make sure it is a hyperspy object when needed
    mean_dp_per_label = get_mean_diffraction_per_labelled_prediction(exp_dat, pred_dat, phase_labels, q_axis)
    peaks = get_peaks(mean_dp_per_label)

    metric = evaluate_peak_position_similarity(peaks, sim_dif_dict)
    return metric

#%%

root = r'G:\My Drive\PhD\projects\external_measurements\ml_difsims'

pred_folder = r'data\predictions\test'
data_folder = r'data\experimental\npz_files'
cif_files_folder = r'models\crystal_phases'
prediction_list_to_evaluate = glob.glob(os.path.join(root, pred_folder, '*_probability_pred.npz'))
prediction_list_to_evaluate.sort()

prediction_list_to_evaluate

#%%

def evaluate_all(prediction_list_to_evaluate, data_folder, cif_files_folder, root_path):
    """
    Input files needed:
    - `*_probabilty_pred.npz` numpy array containing the predictions for each experimental file as a shape of (y, x, prob_n_phases) saved as 'pred' key AND the predicted labels as 'phase_list' key.
    - Link to the original `*_radial_crop.npz` files containing the 1D cropped diffraction patterns and the original q_axis range.
    - Link to the `.cif` files to use to simulate diffraction patterns from
    """

    evaluations = {}
    # Loop through all the files
    for f_pred in prediction_list_to_evaluate:
        evaluation = evaluate_data(f_pred, data_folder, cif_files_folder, root_path)
        evaluations[f_pred] = evaluation

    return evaluations


if __name__ == "__main__":
    result = evaluate_all.execute_in_process()
