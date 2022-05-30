# Packages

import numpy as np
import pandas as pd
import hyperspy.api as hs
import pyxem as pxm
import diffpy.structure
from matplotlib import pyplot as plt
from tempfile import TemporaryFile
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator, VectorLibraryGenerator

import tqdm
import gc
import os
import random

#from secrets import SLACK_WEBHOOK_URL

# import jupyter_slack
# os.environ['SLACK_WEBHOOK_URL'] = SLACK_WEBHOOK_URL

# Randomisation
seed = 10
random.seed(seed)
np.random.seed(seed)
# Unique identifier
import uuid

unique_id = uuid.uuid4()

from pyxem.detectors import Medipix515x515Detector

detector = Medipix515x515Detector()

save_hspy_files = False
plot_hspy_files = False

# %%

# Variables as json structure
json_vars = {
    "root_path": r'G:\My Drive\PhD\projects\external_measurements\ml_difsims',
    "structure_parameters": {
        "phase_files_location_from_root": r"models/crystal_phases",
        "phase_files": ['p4mbm_scaled_mixed_halide.cif', 'gratia_2h.cif', 'pbi2_2h.cif'],
        "add_bkg_phase": False,
        # Do you want to add a bkg/just noise phase at the end? If True, the final datasets will be phases + 1 shape.
    },
    "calibration_parameters": {
        "calibration_value": [0.00588, ],
        #  List of 1 value only for now
        "calibration_modify_percent": 5,
        # It will disturb the calibration value by % when cropping in the q space. In None, nothing happens.
    },
    "orientations_parameters": {
        "n_points": 5,
        "use_orix_sampling": True,
        "ori_files_location_from_root": r"models/orix_orientation_full_lists",
        "orientation_files_list": ['orientations_pg422_3_xxxx.npy',
                                   'orientations_pg622_3_xxxx.npy',
                                   'orientations_pg32_3_xxxx.npy'],
        "orientation_sampling_mode": 'cubo',
    },
    "data_augmentation_parameters": {
        "peak_removal": {
            "remove_peaks_from_diffraction_library": False,
            "n_intensity_peaks": 20,
            # Finds n brightest peaks to remove from. n_intensity_peaks >= n_peaks_to_remove
            "num_peaks_to_remove": 'random',
            # Removes n amount of peaks of the brightest ones.
            # If 'random' is passed, it will randomise this value between 0 and n_intensity_peaks.
        },
        "noise_addition": {
            "add_noise": False,
            "include_also_non_noisy_simulation": False,
            # If add noise, do you want to also have the non-noisy data?
            "snrs": [0.9, 0.99],
            "intensity_spikes": [0.25, ],
        },
        "background_parameters": {
            "add_background_to": 'none',
            # Select from '1d' or 'none'
            "a_vals": [1., 5.],
            # A: pre-exp factor, tau: decay time constant
            "tau_vals": [0.5, 1.5],
        },
    },
    "relrod_parameters": {
        "randomise_relrod": True,
        # If true, will randomly pick one. If false, it will compute all.
        "relrod_list": [0.02, 0.2, 0.4, 0.6, 1, 2, 5, 10],
    },
    "sigma_parameters": {
        "randomise_sigmas": True,
        # If true, will randomly pick one. If false, it will compute all.
        "sigma_2d_gaussian_list": np.arange(3, 10, 1).tolist(),
    },
    "detector_geometry": {
        # In px, keV, m, m
        "detector_size": 515,
        "beam_energy": 200.0,
        "wavelength": 2.5079e-12,
        "detector_pix_size": 55e-6,
        "detector_type": str(detector),
        "radial_integration_1d": True,
        "radial_integration_2d": True,
        "save_peak_position_library": True,
    },
    "scattering_params": 'lobato',
    "simulated_direct_beam_bool": False,
    "postprocessing_parameters": {
        "cropping_start_k": 0.11,
        "cropping_stop_k": 1.30,
        "cropped_signal_k_points": 147,
        # To rebin signal, if necessary (when using k_units)
        "cropping_start_px": 13,
        "cropping_stop_px": 160,
        "sqrt_signal": False,
    },
    "id": f"sim-{unique_id}",
    "random_seed": 10,
    "save_relpath": r'data/simulations',
}

# %%
import json
from types import SimpleNamespace

json_vars_dump = json.dumps(json_vars)
vs = json.loads(json_vars_dump, object_hook=lambda d: SimpleNamespace(**d))

# %%

### Variables

# Paths
root = vs.root_path

# Phases
structures_path = os.path.join(root, vs.structure_parameters.phase_files_location_from_root)
phase_files = vs.structure_parameters.phase_files
add_bkg_phase = vs.structure_parameters.add_bkg_phase

# Calibration values
calibrations = vs.calibration_parameters.calibration_value
calibration_modify_percent = vs.calibration_parameters.calibration_modify_percent

# Orientation values
n_points = vs.orientations_parameters.n_points
use_orix_sampling = vs.orientations_parameters.use_orix_sampling

orientation_list_path = os.path.join(root, vs.orientations_parameters.ori_files_location_from_root)
orientation_files_list = vs.orientations_parameters.orientation_files_list
orientation_sampling_mode = vs.orientations_parameters.orientation_sampling_mode
orientation_list = [s.replace('xxxx', orientation_sampling_mode) for s in orientation_files_list]

# Peak removal tool
remove_peaks_from_diffraction_library = vs.data_augmentation_parameters.peak_removal.remove_peaks_from_diffraction_library
n_intensity_peaks = vs.data_augmentation_parameters.peak_removal.n_intensity_peaks
num_peaks_to_remove = vs.data_augmentation_parameters.peak_removal.num_peaks_to_remove

# Domain amplification
simulated_direct_beam_bool = vs.simulated_direct_beam_bool
randomise_relrod = vs.relrod_parameters.randomise_relrod
relrod_list = vs.relrod_parameters.relrod_list
randomise_sigmas = vs.sigma_parameters.randomise_sigmas
sigma_2d_gaussian_list = vs.sigma_parameters.sigma_2d_gaussian_list
scattering_params = vs.scattering_params

# Noise addition values
add_noise = vs.data_augmentation_parameters.noise_addition.add_noise
include_also_non_noisy_simulation = vs.data_augmentation_parameters.noise_addition.include_also_non_noisy_simulation
snrs = vs.data_augmentation_parameters.noise_addition.snrs
intensity_spikes = vs.data_augmentation_parameters.noise_addition.intensity_spikes

# Background parameterisation values
add_background_to = vs.data_augmentation_parameters.background_parameters.add_background_to
a_vals = vs.data_augmentation_parameters.background_parameters.a_vals
tau_vals = vs.data_augmentation_parameters.background_parameters.tau_vals

# Simulation microscope values (for azimuthal integration)
detector_size = vs.detector_geometry.detector_size
beam_energy = vs.detector_geometry.beam_energy
wavelength = vs.detector_geometry.wavelength
detector_pix_size = vs.detector_geometry.detector_pix_size
radial_integration_1d = vs.detector_geometry.radial_integration_1d
radial_integration_2d = vs.detector_geometry.radial_integration_2d
save_peak_position_library = vs.detector_geometry.save_peak_position_library

# Cropping and post-processing
cropping_start_k = vs.postprocessing_parameters.cropping_start_k
cropping_stop_k = vs.postprocessing_parameters.cropping_stop_k
cropped_signal_k_points = vs.postprocessing_parameters.cropped_signal_k_points

cropping_start_px = vs.postprocessing_parameters.cropping_start_px
cropping_stop_px = vs.postprocessing_parameters.cropping_stop_px
sqrt_signal = vs.postprocessing_parameters.sqrt_signal

# %% md

## Simulate Data
# Check inputs are valid for simulation to succeed
if len(calibrations) == 1:
    calibration = calibrations[0]
else:
    calibration = np.mean(calibrations)

# Get phase labels
phase_dict = {}
for phase in phase_files[:]:
    name = phase.split(".")[0]
    phase_dict[name] = diffpy.structure.loadStructure(os.path.join(structures_path, phase))

# Get euler angle lists
from ml_difsims.simulators.simulation_utils import *

if not use_orix_sampling:
    euler_list_n = get_random_euler(n_points, len(phase_dict.keys()))
else:
    euler_list_n = load_orientation_list(orientation_list, orientation_list_path, n_points)

n_angle_points = int(euler_list_n.shape[1])

# If add bkg phase, add a last phase with any cif file (the atoms will be ignored later)
if add_bkg_phase:
    phase_dict['background'] = phase_dict[name]
    euler_list_n = np.vstack((euler_list_n, [euler_list_n[-1]]))

# %%
# Log all parms
print('n_phases = {}'.format(len(phase_dict)))
print([key for key in phase_dict.keys()])
print('Calibration val = ', calibration)
print(euler_list_n.shape, "phases, n angle points, 3coordinates")

# jupyter_slack.notify_self(f"Simulation started for {len(phase_files)} n phases")

# %%

import dask.array as da
from ml_difsims.utils.batch_processing_utils import chunker

n_ori_chunk = 4

camera_length = detector_pix_size / (wavelength * calibration * 1e10)
unit = "k_A^-1"
radial_steps = int(np.ceil((int(detector_size / 2) - 1) / 2) * 2)

data = da.array([])
data_2d = da.array([])
data_peak_pos = {}

random.seed(seed)
# Iterate through each phase
for i, (key, phase) in enumerate(phase_dict.items()):

    phase_data = da.array([])
    phase_data_2d = da.array([])
    phase_peak_pos = []
    euler_list = euler_list_n[i]

    # TODO: Parallelise here
    for ii, euler_chunk in enumerate(chunker(euler_list, n_ori_chunk, None)):

        # Process each chunk of orientation angles (for each phase individually)
        chunk_data = da.array([])
        chunk_peak_pos = []

        # Simulate diffraction patterns from library
        for euler in euler_chunk:
            # Skip None euler angles (from non-perfect chunking)
            if euler is None:
                continue
            # Iterate through relrods
            reciprocal_radius = get_reciprocal_radius(detector_size, calibration)
            len_relrodlist = 1 if randomise_relrod else len(relrod_list)
            for j in range(len_relrodlist):
                if randomise_relrod:
                    relrod_length = random.choice(relrod_list)
                else:
                    relrod_length = relrod_list[j]

                phase_dict_temp = {key: phase}
                library = create_diffraction_library(phase_dict_temp, [[euler]], beam_energy, scattering_params,
                                                     relrod_length, calibration, detector_size,
                                                     simulated_direct_beam_bool)

                # Get coordinates from simulation
                coords_library = get_coordinates_dict_from_library(library)
                chunk_peak_pos.append(coords_library)

                # Generate simulation
                lib_entry = library.get_library_entry(phase=key, angle=euler)['Sim']

                # Data augmentation
                if remove_peaks_from_diffraction_library:
                    lib_entry_mod = remove_peaks(lib_entry, n_intensity_peaks, num_peaks_to_remove)
                else:
                    lib_entry_mod = lib_entry

                # Iterate through sigmas
                len_sigmaslist = 1 if randomise_sigmas else len(sigma_2d_gaussian_list)
                for k in range(len_sigmaslist):
                    if randomise_sigmas:
                        sigma_2d_gaus = random.choice(sigma_2d_gaussian_list)
                    else:
                        sigma_2d_gaus = sigma_2d_gaussian_list[k]

                    pattern = lib_entry_mod.get_diffraction_pattern(sigma=sigma_2d_gaus)
                    try:
                        chunk_data = da.vstack((chunk_data, [pattern]))
                    except ValueError:
                        chunk_data = [pattern]

                    # plt.figure()
                    # plt.imshow(pattern, cmap='viridis')
                    # plt.show()

        # Chunk simulated (for a n_chunk_size of one phase only)
        # Create hyperspy signal and recenter
        chunk_s = pxm.signals.LazyElectronDiffraction2D(chunk_data)
        try:
            chunk_s.compute()
        except AttributeError:
            chunk_s = pxm.signals.ElectronDiffraction2D(chunk_data)

        chunk_s.set_diffraction_calibration(calibration)
        shifts_list = np.ones((chunk_s.data.shape[0], 2)) * 0.5
        chunk_s.align2D(shifts=shifts_list, crop=False, fill_value=0., parallel=True)

        if key == 'background':
            chunk_s.data *= 0

        # Data amplification with noise
        if add_noise:
            from ml_difsims.simulators.noise_utils import *

            training_data_noisy = []
            # Include the non-corrupted data in the dataset?
            if include_also_non_noisy_simulation:
                training_data_noisy.append(chunk_s)

            # Append noisy data
            for snr in snrs:
                for int_spike in intensity_spikes:
                    signal_noisy = chunk_s.map(add_noise_to_simulation,
                                               snr=snr, int_salt=int_spike,
                                               inplace=False, parallel=True)

                    training_data_noisy.append(signal_noisy)

            chunk_s = hs.stack(training_data_noisy, axis=0)

        # Radial integration
        chunk_s.unit = unit
        chunk_s.set_experimental_parameters(beam_energy=beam_energy)
        chunk_s.set_ai(center=([detector_size / 2, detector_size / 2]))

        if radial_integration_1d:
            chunk_s_radial = chunk_s.get_azimuthal_integral1d(npt=radial_steps, )
            qx_axis = chunk_s_radial.axes_manager.signal_axes[0].axis
            # Convert to numpy and append
            if ii == 0:
                phase_data = chunk_s_radial.data
            else:
                phase_data = da.vstack((phase_data, chunk_s_radial.data))

        if radial_integration_2d:
            chunk_s_cake = chunk_s.get_azimuthal_integral2d(npt=radial_steps, )
            qx_axis = chunk_s_cake.axes_manager.signal_axes[0].axis
            # Convert to numpy and append
            if ii == 0:
                phase_data_2d = chunk_s_cake.data
            else:
                phase_data_2d = da.vstack((phase_data_2d, chunk_s_cake.data))


        if save_peak_position_library:
            # Add peak positions to phase as a dictionary entry
            phase_peak_pos.append(chunk_peak_pos)
            data_peak_pos[key] = {}
            i_dict = 0
            for chunk_group in phase_peak_pos:
                for dict_peak_pos in chunk_group:
                    data_peak_pos[key][i_dict] = dict_peak_pos
                    i_dict += 1

    if radial_integration_1d:
        if i == 0:
            data = [phase_data]
        else:
            data = da.vstack((data, [phase_data]))

    if radial_integration_2d:
        if i == 0:
            data_2d = [phase_data_2d]
        else:
            data_2d = da.vstack((data_2d, [phase_data_2d]))

data_peak_pos = json.dumps(data_peak_pos)

print(data)
print(data_2d)
print(data_peak_pos)

# %%
from ml_difsims.utils.postprocessing_utils import *

# Post processing
if radial_integration_1d:
    # Sqrt signal (if wanted)
    if sqrt_signal:
        data = da.sqrt(data)

    # Add simulated background
    # Approximate background as a $A * exp ^ {(-tau \: q)}$ value.
    if add_background_to == '1d':
        # Normalise
        dpmax = data.max(2).compute()
        data_norm = data / dpmax[:, :, np.newaxis]
        # Correct any nan value
        nan_mask = np.isnan(data_norm)
        data_norm[nan_mask] = 0

        for a in a_vals:
            for i, tau in enumerate(tau_vals):
                bkg_data = add_background_to_signal_array(data_norm, qx_axis, a, tau)
                if i == 0:
                    data = np.hstack((data_norm, bkg_data))
                else:
                    data = np.hstack((data, bkg_data))

    ## Crop, rebin and normalise on pixel coords
    # Crop in pixel units:
    data_px = data[:, :, cropping_start_px : cropping_stop_px]
    # Renormalise
    dpmax = data_px.max(-1)
    data_px = data_px / dpmax[:, :, np.newaxis]
    # Correct any nan value
    nan_mask = np.isnan(data_px)
    data_px[nan_mask] = 0

    ## Crop, rebin and normalise on k coords
    # Crop in k units:
    data_s = pxm.signals.LazyElectronDiffraction1D(data)
    data_s.compute()
    # TODO: Parallelise here too
    if calibration_modify_percent == (None or 0):
        data_s = data_s.crop_signal1D(cropping_start_k, cropping_stop_k)
        data_k = rebin_signal(data_s, cropped_signal_k_points).data
    else:
        range_percents_modification = np.linspace(-calibration_modify_percent, calibration_modify_percent, 20)
        signals_temp = da.array([])
        for i, nav in enumerate(data_s.inav[:]):
            percent = random.choice(range_percents_modification) / 100
            cropping_start_k_temp = (1 + percent) * cropping_start_k
            cropping_stop_k_temp = (1 + percent) * cropping_stop_k
            try:
                nav.crop_signal1D(cropping_start_k_temp, cropping_stop_k_temp)
            except ValueError:
                nav.crop_signal1D(cropping_start_k, cropping_stop_k)

            nav_k = rebin_signal(nav, cropped_signal_k_points).data
            if i == 0:
                signals_temp = da.array(nav_k)
            else:
                signals_temp = da.vstack((signals_temp, nav_k))

        # Reshape back
        shape_ori = list(data.shape)
        shape_ori[-1] = cropped_signal_k_points
        data_k = np.reshape(signals_temp, shape_ori)

    data_k = da.array(data_k)
    # Renormalise
    dpmax = data_k.max(-1)
    data_k = data_k / dpmax[:, :, np.newaxis]
    # Correct any nan value
    nan_mask = np.isnan(data_k)
    data_k[nan_mask] = 0

    # NN Requirements: Reshape and Labelling
    data_px = data_px.reshape(-1, data_px.shape[-1])
    data_k = data_k.reshape(-1, data_k.shape[-1])
    data = data.reshape(-1, data.shape[-1])

    # Create labels for 1D
    phase_names = list(phase_dict.keys())
    n_phases = len(phase_dict)
    labels = np.zeros((n_phases, int(data_px.shape[0] / n_phases)))
    for i in range(n_phases):
        labels[i, :] = i

    labels = labels.flatten()

if radial_integration_2d:
    # Sqrt signal (if wanted)
    if sqrt_signal:
        data_2d = da.sqrt(data_2d)

    # Add simulated background
    # TODO: Add 2d background simulation

    ## Crop, rebin and normalise on pixel coords
    # # Crop in pixel units:
    if calibration_modify_percent == (None or 0):
        data_2d_px = data_2d[:, :, cropping_start_px: cropping_stop_px, :]
    else:
        # Calculate how many pixels the calibration tolerance factor corresponds to
        one_px_q = qx_axis.max() / radial_steps
        max_q_range = calibration_modify_percent/100 * qx_axis.max()
        max_px_shift = max_q_range / one_px_q
        range_percents_modification = np.arange(-max_px_shift, max_px_shift+1, 1)
        signals_temp_2d = da.array([])
        for phase_2d in data_2d:
            for ori_2d in phase_2d:
                px_shift = random.choice(range_percents_modification) / 100
                temp = ori_2d[np.ceil(cropping_start_px + px_shift): np.ceil(cropping_stop_px + px_shift), :]
                try:
                    signals_temp_2d = da.vstack((signals_temp_2d, temp))
                except ValueError:
                    signals_temp_2d = da.array(temp)

        # Reshape back
        shape_ori = list(data_2d.shape)
        shape_ori[-2] = cropped_signal_k_points
        data_2d_px = np.reshape(signals_temp_2d, shape_ori)

    # Renormalise
    dpmax = data_2d_px.max([-2,-1])
    data_2d_px = data_2d_px / dpmax[:, :, np.newaxis, np.newaxis]
    # Correct any nan value
    nan_mask = np.isnan(data_2d_px)
    data_2d_px[nan_mask] = 0

    # NN Requirements: Reshape and Labelling
    data_2d_px = data_2d_px.reshape(-1, data_2d_px.shape[-2], data_2d_px.shape[-1])

    # Create labels for 2D
    phase_names = list(phase_dict.keys())
    n_phases = len(phase_dict)
    labels_2d = np.zeros((n_phases, int(data_2d_px.shape[0] / n_phases)))
    for i in range(n_phases):
        labels_2d[i, :] = i

    labels_2d = labels.flatten()

# %%
# Saving
calibration_str = f"{calibration}".replace('.', 'p')
if calibration_modify_percent is not None:
    calibration_str += f"mod{calibration_modify_percent}percent".replace('.', 'p')

relrod_str = f"{relrod_list}".replace('[', '').replace(']', '').replace(', ', '-').replace('.', 'p')
sigma_2d_str = f"{sigma_2d_gaussian_list}".replace('[', '').replace(']', '').replace(', ', '-').replace('.', 'p')

full_name = '1D_simulated_data_cal{}_cropK_randrelrod{}_randsigma2d{}_{}classes_orix_{}neuler_peakremoval_{}npeaks_random'.format(
    calibration_str,
    relrod_str,
    sigma_2d_str,
    n_phases,
    n_angle_points,
    n_intensity_peaks)

calibration_range_str = f"Calibrations range : {calibrations[0]} pm {calibrations[0] * calibration_modify_percent / 100}"
print(calibration_range_str)
print(full_name)

# Save all sorts of useful files
import h5py
id_name = f'sim-{unique_id}.hdf5'
save_path = os.path.join(root, vs.save_relpath, id_name)
with h5py.File(save_path, 'w') as f:
    if radial_integration_1d:
        g = f.create_group('1d')
        g.create_dataset('x_all', data=data)
        g.create_dataset('y_all', data=labels)
        g.create_dataset('x_q', data=data_k)
        g.create_dataset('y_q', data=labels)
        g.create_dataset('x_px', data=data_px)
        g.create_dataset('y_px', data=labels)
        g.create_dataset('x_all_q_axis', data=qx_axis)

    if radial_integration_2d:
        g = f.create_group('2d')
        g.create_dataset('x_px', data=data_2d_px)
        g.create_dataset('y_px', data=labels_2d)

    if save_peak_position_library:
        g = f.create_group('peaks_positions')
        g.create_dataset('peaks_positions_json', data=data_peak_pos)

    g = f.create_group('metadata')
    g.attrs['phases'] = phase_names
    g.attrs['summary'] = full_name
    g.create_dataset('metadata_json', data=json_vars_dump)

from mongodb.pymongo_connect import connect_to_mongo_database

# Send json database to mongodb
db_collection = connect_to_mongo_database('models', f'sim-{unique_id}')
db_collection.insert_one(json_vars)

# jupyter_slack.notify_self(f"Simulation finished for:")
# jupyter_slack.notify_self(full_name)

# %% md

### Plotting

# %% raw

# i = 0
# plt.figure()
# # plt.plot(training_data_1D_px[i], label='px')
# plt.plot(training_data_1D_q[i], label='q')
# plt.legend()
# plt.savefig('1D_plt_compare_k_q_cropping_i{}.png'.format(i))

# del training_data_1D_px
# del training_data_1D_q

# %%