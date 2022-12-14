# Packages
import math
from datetime import datetime
import dask.array as da
import hyperspy.api as hs
import pyxem as pxm
import diffpy.structure
from dagster import op, get_dagster_logger, Out, DynamicOut, DynamicOutput, graph, GraphOut, RetryPolicy
from scp import SCPClient, SCPException

from ml_difsims.simulators.simulation_utils import *
from ml_difsims.simulators.noise_utils import *
from ml_difsims.utils.postprocessing_utils import *
from ml_difsims.utils.batch_processing_utils import chunker
import json
from types import SimpleNamespace
import os
import random
import uuid

from ml_difsims.utils.external_connects import connect_to_mongo_database, start_shh_connection

default_policy = RetryPolicy(max_retries=3)

# %%
@op(out={"vs": Out(), "json_vars_dump": Out()}, config_schema={"json_string": str})
def load_json_metadata_to_dict(context):
    # Variables as json structure (Only input)
    json_dump = context.op_config["json_string"]
    vs = json.loads(json_dump, object_hook=lambda d: SimpleNamespace(**d))
    return vs, json_dump

@op(out={"calibration": Out(), "vs": Out()})
def get_dif_calibration_value(vs):
    calibrations = vs.calibration_parameters.calibration_value
    if len(calibrations) == 1:
        calibration = calibrations[0]
    else:
        calibration = np.mean(calibrations)
    get_dagster_logger().info(f'Calibration val = {calibration}')
    setattr(vs, 'calibration', calibration)
    return calibration, vs

@op
def get_phases_keys(vs):
    phase_files = vs.structure_parameters.phase_files
    return [phase.split(".")[0] for phase in phase_files]

# Get phase labels
@op(out={"phase_dict": Out(), "vs": Out()})
def create_phase_dict(vs):
    root = vs.root_path
    phase_files = vs.structure_parameters.phase_files
    structures_path = os.path.join(root, vs.structure_parameters.phase_files_location_from_root)

    phase_names = get_phases_keys(vs)
    phase_dict = {}
    for phase, name in zip(phase_files, phase_names):
        phase_dict[name] = diffpy.structure.loadStructure(os.path.join(structures_path, phase))
    get_dagster_logger().info(f'n_phases = {len(phase_dict)}')
    get_dagster_logger().info(f'{[key for key in phase_dict.keys()]}')
    # setattr(vs, 'phase_dict', phase_dict)
    return phase_dict, vs

# Get euler angle lists
@op(out={"euler_list_n": Out(), "vs": Out()})
def get_euler_angle_lists(vs):
    use_orix_sampling = vs.orientations_parameters.use_orix_sampling
    n_points = vs.orientations_parameters.n_points
    root = vs.root_path
    orientation_list_path = os.path.join(root, vs.orientations_parameters.ori_files_location_from_root)
    orientation_files_list = vs.orientations_parameters.orientation_files_list
    orientation_sampling_mode = vs.orientations_parameters.orientation_sampling_mode
    phase_names = get_phases_keys(vs)
    seed = vs.random_seed

    if not use_orix_sampling:
        euler_list_n = get_random_euler(n_points, len(phase_names))
    else:
        euler_list_n = load_orientation_list(orientation_files_list, orientation_sampling_mode, orientation_list_path, n_points, seed)

    get_dagster_logger().info(f'{euler_list_n.shape}, phases, n angle points, 3coordinates')
    # setattr(vs, 'euler_list_n', euler_list_n)
    return euler_list_n, vs

# If add bkg phase, add a last phase with any cif file (the atoms will be ignored later)
@op(out={"phase_dict": Out(), "euler_list_n": Out(), "vs": Out()})
def make_changes_to_add_bkg_phase(vs, phase_dict, euler_list_n):
    add_bkg_phase = vs.structure_parameters.add_bkg_phase
    if add_bkg_phase:
        phase_dict['background'] = list(phase_dict.values())[-1]
        euler_list_n = np.vstack((euler_list_n, [euler_list_n[-1]]))
        # setattr(vs, 'phase_dict', phase_dict)
        # setattr(vs, 'euler_list_n', euler_list_n)
    return phase_dict, euler_list_n, vs

# Randomisation
@op
def start_random_seeds(vs):
    seed = vs.random_seed
    random.seed(seed)
    np.random.seed(seed)
    return vs

@op(out={"id": Out(), "vs": Out()})
def get_unique_id(vs):
    id = uuid.uuid4()
    setattr(vs, 'id', f'sim-{id}')
    return id, vs

@op(out={"dp": Out(), "vs": Out()})
def set_detector_on_dp_object(vs, dp):

    beam_energy = vs.beam_energy
    unit = "k_A^-1"
    calibration = vs.calibration
    detector_size = dp.axes_manager.signal_axes[0].size
    assert detector_size == dp.axes_manager.signal_axes[1].size

    radial_steps = int(np.ceil((int(detector_size / 2) - 1) / 2) * 2)
    setattr(vs, 'radial_steps', radial_steps)
    center = [detector_size / 2, detector_size / 2]

    # # PyFAI way (without knowing the calibration but the camera_length
    # @op
    # def get_diffsims_detector_object(vs):
    #     detector_type = vs.detector_geometry.detector_type
    #     detector = getattr(pxm.detectors, detector_type)
    #     return detector
    # from diffsims.utils.sim_utils import get_electron_wavelength
    # wavelength = get_electron_wavelength(beam_energy)
    # detector_pix_size = vs.detector_geometry.detector_pix_size
    # detector = get_diffsims_detector_object(vs)
    # camera_length = detector_pix_size / (wavelength * calibration * 1e10)
    # ai = AzimuthalIntegrator(dist=camera_length, detector=detector, wavelength=wavelength)
    # ai.setFit2D(directDist=camera_length * 1000, centerX=center[1], centerY=center[0])
    # dp.metadata.set_item("Signal.ai", ai)

    # Old "pyxem" way
    dp.set_diffraction_calibration(calibration)
    dp.unit = unit
    dp.set_experimental_parameters(beam_energy=beam_energy)
    dp.set_ai(center=center)
    return dp, vs

# Modify crystal files
@op(out={"phase_mod": Out()})
def scale_crystal_phase(phase, key, scale_range_dict,):

    def scale_lattice(cif_file, scaling_factor):
        import copy
        cif = copy.deepcopy(cif_file)
        cif.lattice.a *= scaling_factor
        cif.lattice.b *= scaling_factor
        cif.lattice.c *= scaling_factor
        return cif

    try:
        scale_range = scale_range_dict[key]
        modify_scale = np.random.uniform(*scale_range)
        phase_mod = scale_lattice(phase, modify_scale)
        return phase_mod

    except KeyError:
        return phase

@op
def get_simulation_library(vs, i_relrod, key, phase, euler):
    randomise_relrod = vs.relrod_parameters.randomise_relrod
    relrod_list = vs.relrod_parameters.relrod_list
    beam_energy = vs.beam_energy
    scattering_params = vs.structure_parameters.scattering_params
    calibration = vs.calibration
    detector_size = vs.detector_geometry.detector_size
    simulated_direct_beam_bool = vs.data_augmentation_parameters.simulated_direct_beam_bool
    remove_peaks_from_diffraction_library = vs.data_augmentation_parameters.peak_removal.remove_peaks_from_diffraction_library
    n_intensity_peaks = vs.data_augmentation_parameters.peak_removal.n_intensity_peaks
    num_peaks_to_remove = vs.data_augmentation_parameters.peak_removal.num_peaks_to_remove
    calibration_modify_percent = vs.calibration_parameters.calibration_modify_percent
    scale_cif_files = vs.calibration_parameters.scale_cif_files
    scale_range_dict = vs.calibration_parameters.scale_range_dict

    if randomise_relrod:
        relrod_length = random.choice(relrod_list)
    else:
        relrod_length = relrod_list[i_relrod]

    if scale_cif_files:
        phase = scale_crystal_phase(phase, key, scale_range_dict)

    if calibration_modify_percent != (None or 0):
        modify_percent = np.random.uniform(low=-calibration_modify_percent, high=calibration_modify_percent)

        calibration = calibration * (1 + modify_percent / 100)

    phase_dict_temp = {key: phase}
    library = create_diffraction_library(phase_dict_temp, [[euler]], beam_energy, scattering_params,relrod_length, calibration, detector_size,simulated_direct_beam_bool)

    lib_entry = library.get_library_entry(phase=key, angle=euler)['Sim']

    # Data augmentation
    if remove_peaks_from_diffraction_library:
        lib_entry_mod = remove_peaks(lib_entry, n_intensity_peaks, num_peaks_to_remove)
    else:
        lib_entry_mod = lib_entry

    return library, lib_entry_mod

@op
def get_diffraction_pattern(lib_entry_mod, vs, i_sigma):
    randomise_sigmas = vs.sigma_parameters.randomise_sigmas
    sigma_2d_gaussian_list = vs.sigma_parameters.sigma_2d_gaussian_list

    if randomise_sigmas:
        sigma_2d_gaus = random.choice(sigma_2d_gaussian_list)
    else:
        sigma_2d_gaus = sigma_2d_gaussian_list[i_sigma]

    pattern = lib_entry_mod.get_diffraction_pattern(sigma=sigma_2d_gaus)
    return pattern

@op
def pyxem_setup_and_corrections(chunk_data, vs, key):
    calibration = vs.calibration

    # Create hyperspy signal and recenter
    dp = pxm.signals.LazyElectronDiffraction2D(chunk_data)
    try:
        dp.compute()
    except AttributeError:
        dp = pxm.signals.ElectronDiffraction2D(chunk_data)

    dp.set_diffraction_calibration(calibration)
    shifts_list = np.ones((dp.data.shape[0], 2)) * 0.5
    dp.align2D(shifts=shifts_list, crop=False, fill_value=0., parallel=True)

    if key == 'background':
        dp.data = dp.data * 0
    return dp

@op
def add_noise_to_pyxem_dp(dp, vs):
    add_noise = vs.data_augmentation_parameters.noise_addition.add_noise
    include_also_non_noisy_simulation = vs.data_augmentation_parameters.noise_addition.include_also_non_noisy_simulation
    snrs = vs.data_augmentation_parameters.noise_addition.snrs
    intensity_spikes = vs.data_augmentation_parameters.noise_addition.intensity_spikes

    # If add_noise is false:
    if not add_noise:
        return dp
    else:
        training_data_noisy = []
        # Include the non-corrupted data in the dataset?
        if include_also_non_noisy_simulation:
            training_data_noisy.append(dp)

        # Data amplification with a random subset of the param space
        if add_noise == 'random':
            nav_shape = dp.axes_manager.navigation_shape
            nav_size = math.prod(nav_shape)

            snr = random.choices(snrs, k=nav_size)
            int_spike = random.choices(intensity_spikes, k=nav_size)
            snr = np.reshape(snr, nav_shape)
            int_spike = np.reshape(int_spike, nav_shape)
            snr = hs.signals.BaseSignal(snr).T
            int_spike = hs.signals.BaseSignal(int_spike).T
            signal_noisy = dp.map(add_noise_to_simulation,
                                  snr=snr, int_salt=int_spike,
                                  inplace=False, parallel=True)
            training_data_noisy.append(signal_noisy)
        # Data amplification with all param space
        else:
            for snr in snrs:
                for int_spike in intensity_spikes:
                    signal_noisy = dp.map(add_noise_to_simulation,
                                          snr=snr, int_salt=int_spike,
                                          inplace=False, parallel=True)

                    training_data_noisy.append(signal_noisy)

        return hs.stack(training_data_noisy, axis=0, signal_type='electron_diffraction')

@op
def radially_integrate_1d(dp, vs):
    radial_steps = vs.radial_steps
    return dp.get_azimuthal_integral1d(npt=radial_steps, )

@op
def radially_integrate_2d(dp, vs):
    radial_steps = vs.radial_steps
    return dp.get_azimuthal_integral2d(npt=radial_steps, )

#%%
@op(out={"a": Out(), "b": Out(), "c": Out(), "d": Out()})
def get_metadata(vs):
    a = vs.simulation_orientations_chunk_size
    b = vs.detector_geometry.radial_integration_1d
    c = vs.detector_geometry.radial_integration_2d
    d = vs.detector_geometry.save_peak_position_library
    return a, b, c, d


@op(out={"keys": Out(), "vals": Out()})
def get_dictionary_items(dictionary):
    keys = [k for k in dictionary.keys()]
    vals = [v for v in dictionary.values()]
    return keys, vals


@op(out=DynamicOut())
def get_chunks(euler_list_n, phase_dict, n_ori_chunk):
    "Returns an list with iterable phase and angle chunk"
    idx = -1
    for i, (key, phase) in enumerate(phase_dict.items()):
        euler_list = euler_list_n[i]
        for euler_chunk in chunker(euler_list, n_ori_chunk, None):
            chunk = [key, phase, euler_chunk]
            idx += 1
            yield DynamicOutput(chunk, mapping_key=f"{idx}")

@op(out={"chunk_rad_data": Out(), "chunk_rad_data_2d": Out(), "chunk_peak_pos": Out(), "qx_axis": Out()},)
def process_each_chunk(chunk, vs):
    "Returns a dictionary with {key: dask_chunk_array}"
    key = chunk[0]
    phase = chunk[1]
    euler_chunk = chunk[2]

    # Process each chunk of orientation angles (for each phase individually)
    chunk_data = da.array([])
    chunk_peak_pos = []
    qx_axis = None

    # Simulate diffraction patterns from library
    for euler in euler_chunk:
        # Skip None euler angles (from non-perfect chunking)
        if euler is None:
            continue

        # Iterate through relrods
        randomise_relrod = vs.relrod_parameters.randomise_relrod
        relrod_list = vs.relrod_parameters.relrod_list
        len_relrodlist = 1 if randomise_relrod else len(relrod_list)
        for j in range(len_relrodlist):
            library, lib_entry_mod = get_simulation_library(vs, j, key, phase, euler)
            # Get coordinates from simulation
            coords_library = get_coordinates_dict_from_library(library)
            chunk_peak_pos.append(coords_library)

            # Iterate through sigmas
            randomise_sigmas = vs.sigma_parameters.randomise_sigmas
            sigma_2d_gaussian_list = vs.sigma_parameters.sigma_2d_gaussian_list
            len_sigmaslist = 1 if randomise_sigmas else len(sigma_2d_gaussian_list)
            for k in range(len_sigmaslist):

                pattern = get_diffraction_pattern(lib_entry_mod, vs, k)
                try:
                    chunk_data = da.vstack((chunk_data, [pattern]))
                except ValueError:
                    chunk_data = [pattern]

                # plt.figure()
                # plt.imshow(pattern, cmap='viridis')
                # plt.show()

    # Chunk simulated (for a n_chunk_size of one phase only)
    chunk_s = pyxem_setup_and_corrections(chunk_data, vs, key)
    chunk_s = add_noise_to_pyxem_dp(chunk_s, vs)

    # Radial integration
    chunk_s, vs = set_detector_on_dp_object(vs, chunk_s)

    if vs.detector_geometry.radial_integration_1d:
        chunk_s_radial = radially_integrate_1d(chunk_s, vs)
        qx_axis = chunk_s_radial.axes_manager.signal_axes[0].axis
        chunk_rad_data = {key: da.array(chunk_s_radial.data)}
    else:
        chunk_rad_data = None

    if vs.detector_geometry.radial_integration_2d:
        chunk_s_cake = radially_integrate_2d(chunk_s, vs)
        qx_axis = chunk_s_cake.axes_manager.signal_axes[1].axis
        chunk_rad_data_2d = {key: da.array(chunk_s_cake.data)}
    else:
        chunk_rad_data_2d = None

    if vs.detector_geometry.save_peak_position_library:
        # Add peak positions to phase as a dictionary entry
        chunk_peak_pos = {key: chunk_peak_pos}
    else:
        chunk_peak_pos = None

    return chunk_rad_data, chunk_rad_data_2d, chunk_peak_pos, qx_axis

@op()
def merge_dict_chunk_to_dask_arr(data_list, phase_dict, bool_proceed):
    if bool_proceed:
        key_list = [k for k in phase_dict.keys()]
        phase_dat_dict = {key: None for key in key_list}

        for dictionary in data_list:
            key = [k for k in dictionary.keys()][0]
            dat = dictionary[key]
            try:
                phase_dat_dict[key] = da.vstack((phase_dat_dict[key], dat))
            except ValueError:
                phase_dat_dict[key] = dat

        data = da.array([])
        for i, dat in enumerate(phase_dat_dict.values()):
            if i == 0:
                data = [dat]
            else:
                data = da.vstack((data, [dat]))
        return data
    else:
        return None
@op()
def merge_peak_position_dictionary(peak_position_dict_list, phase_dict, bool_proceed):
    if bool_proceed:
        key_list = [k for k in phase_dict.keys()]
        data_peak_pos = {key: [] for key in key_list}
        for chunk_element_dict in peak_position_dict_list:
            # Get a dictionary with "key" and list of pos_dict
            key = [k for k in chunk_element_dict.keys()][0]
            chunk_peak_pos_dict_list = chunk_element_dict[key]
            data_peak_pos[key].append(chunk_peak_pos_dict_list)

        data_peak_pos = json.dumps(data_peak_pos)
        return data_peak_pos
    return None

@op()
def get_qx_axis_array(qx_axis_list):
    return qx_axis_list[0]

# Post-processing ops
@op
def crop_and_rebin_individually_1d(vs, range, dp):
    cropping_start_k = vs.postprocessing_parameters.cropping_start_k
    cropping_stop_k = vs.postprocessing_parameters.cropping_stop_k
    cropped_signal_k_points = vs.postprocessing_parameters.cropped_signal_k_points

    percent = random.choice(range) / 100
    cropping_start_k_temp = (1 + percent) * cropping_start_k
    cropping_stop_k_temp = (1 + percent) * cropping_stop_k
    try:
        dp.crop_signal1D(cropping_start_k_temp, cropping_stop_k_temp)
    except ValueError:
        dp.crop_signal1D(cropping_start_k, cropping_stop_k)

    return rebin_signal(dp, cropped_signal_k_points).data

@op(out={"data": Out(), "data_k": Out(), "data_px": Out(), "labels": Out()})
def post_processing_1d(vs, data, qx_axis, phase_dict):
    sqrt_signal = vs.postprocessing_parameters.sqrt_signal
    add_background_to_simulation = vs.data_augmentation_parameters.background_parameters.add_background_to_simulation
    cropping_start_k = vs.postprocessing_parameters.cropping_start_k
    cropping_stop_k = vs.postprocessing_parameters.cropping_stop_k
    cropped_signal_k_points = vs.postprocessing_parameters.cropped_signal_k_points
    calibration_modify_percent = vs.calibration_parameters.calibration_modify_percent
    cropping_start_px = vs.postprocessing_parameters.cropping_start_px
    cropping_stop_px = vs.postprocessing_parameters.cropping_stop_px
    crop_in_k = vs.postprocessing_parameters.crop_in_k
    crop_in_px = vs.postprocessing_parameters.crop_in_px
    save_full_scan = vs.postprocessing_parameters.save_full_scan
    radial_integration_1d = vs.detector_geometry.radial_integration_1d
    include_also_non_bkg_simulation = vs.data_augmentation_parameters.background_parameters.include_also_non_bkg_simulation

    if radial_integration_1d:
        # Sqrt signal (if wanted)
        if sqrt_signal:
            data = da.sqrt(data)

        # Add simulated background
        # Approximate background as a $A * exp ^ {(-tau \: q)}$ value.
        if add_background_to_simulation is not False:
            # Normalise
            try:
                dpmax = data.max(2, keepdims=True).compute()
                dpmin = data.min(2, keepdims=True).compute()
            except AttributeError:
                dpmax = data.max(2, keepdims=True)
                dpmin = data.min(2, keepdims=True)
            data_norm = (data - dpmin) / (dpmax - dpmin)
            # Correct any nan value
            nan_mask = np.isnan(data_norm)
            data_norm[nan_mask] = 0

            a_vals = vs.data_augmentation_parameters.background_parameters.a_vals
            tau_vals = vs.data_augmentation_parameters.background_parameters.tau_vals
            # Data amplification with a random subset of the param space
            if add_background_to_simulation == 'random':
                nav_shape = data_norm.shape[:-1]
                nav_size = math.prod(nav_shape)

                tau = random.choices(tau_vals, k=nav_size)
                a = random.choices(a_vals, k=nav_size)
                tau = np.reshape(tau, nav_size)
                a = np.reshape(a, nav_size)

                data_norm_1dnav = np.reshape(data_norm, (nav_size, data_norm.shape[-1]))
                bkg_data = add_background_to_signal_array(data_norm_1dnav, qx_axis, a, tau)
                bkg_data = np.reshape(bkg_data, data_norm.shape)
                if include_also_non_bkg_simulation:
                    data = da.hstack((data_norm, bkg_data))
                else:
                    data = bkg_data
            else:
                for i, a in enumerate(a_vals):
                    for tau in tau_vals:
                        a, tau = np.array([a]), np.array([tau])
                        bkg_data = add_background_to_signal_array(data_norm, qx_axis, a, tau)
                        if i == 0:
                            data = da.hstack((data_norm, bkg_data))
                        else:
                            data = da.hstack((data, bkg_data))

        ## Crop, rebin and normalise on pixel coords
        # Crop in pixel units:
        if crop_in_px:
            data_px = data[:, :, cropping_start_px: cropping_stop_px]
            # Renormalise
            dpmax = data_px.max(-1, keepdims=True)
            dpmin = data_px.min(-1, keepdims=True)
            data_px = (data_px - dpmin) / (dpmax - dpmin)

            # Correct any nan value
            nan_mask = np.isnan(data_px)
            data_px[nan_mask] = 0

            # NN Requirements: Reshape and Labelling
            data_px = data_px.reshape(-1, data_px.shape[-1])
        else:
            data_px = None

        ## Crop, rebin and normalise on k coords
        # Crop in k units:
        if crop_in_k:
            data_s = pxm.signals.LazyElectronDiffraction1D(data)
            data_s.compute()

            if calibration_modify_percent == (None or 0):
                data_s = data_s.crop_signal1D(cropping_start_k, cropping_stop_k)
                data_k = rebin_signal(data_s, cropped_signal_k_points).data
            else:
                range_percents_modification = np.linspace(-calibration_modify_percent, calibration_modify_percent, 20)

                signals_temp = da.array([])
                for i, nav in enumerate(data_s.inav[:]):
                    nav_k = crop_and_rebin_individually_1d(vs, range_percents_modification, nav)
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
            dpmax = data_k.max(-1, keepdims=True)
            dpmin = data_k.min(-1, keepdims=True)
            data_k = (data_k - dpmin) / (dpmax - dpmin)

            # Correct any nan value
            nan_mask = np.isnan(data_k)
            data_k[nan_mask] = 0

            # NN Requirements: Reshape and Labelling
            data_k = data_k.reshape(-1, data_k.shape[-1])
        else:
            data_k = None

        # NN Requirements: Reshape and Labelling
        data = data.reshape(-1, data.shape[-1])

        # Create labels for 1D
        phase_names = list(phase_dict.keys())
        n_phases = len(phase_dict)
        labels = np.zeros((n_phases, int(data.shape[0] / n_phases)))
        for i in range(n_phases):
            labels[i, :] = i
        labels = labels.flatten()

        if not save_full_scan:
            data = None
    else:
        data, data_k, data_px, labels = None, None, None, None

    return data, data_k, data_px, labels

@op(out={"data_2d": Out(), "data_2d_px": Out(), "labels_2d": Out()})
def post_processing_2d(vs, data_2d, qx_axis, phase_dict):
    sqrt_signal = vs.postprocessing_parameters.sqrt_signal
    cropped_signal_k_points = vs.postprocessing_parameters.cropped_signal_k_points
    calibration_modify_percent = vs.calibration_parameters.calibration_modify_percent
    cropping_start_px = vs.postprocessing_parameters.cropping_start_px
    cropping_stop_px = vs.postprocessing_parameters.cropping_stop_px
    radial_integration_2d = vs.detector_geometry.radial_integration_2d
    crop_in_k = vs.postprocessing_parameters.crop_in_k
    crop_in_px = vs.postprocessing_parameters.crop_in_px
    save_full_scan = vs.postprocessing_parameters.save_full_scan
    add_background_to_simulation = vs.data_augmentation_parameters.background_parameters.add_background_to_simulation
    include_also_non_bkg_simulation = vs.data_augmentation_parameters.background_parameters.include_also_non_bkg_simulation


    if radial_integration_2d:
        # Sqrt signal (if wanted)
        if sqrt_signal:
            data_2d = da.sqrt(data_2d)

        # Add 2d background simulation
        # Approximate background as a $A * exp ^ {(-tau \: q)}$ value.
        if add_background_to_simulation is not False:
            # Normalise
            try:
                dpmax = data_2d.max((-2,-1), keepdims=True).compute()
                dpmin = data_2d.min((-2,-1), keepdims=True).compute()
            except AttributeError:
                dpmax = data_2d.max((-2, -1), keepdims=True)
                dpmin = data_2d.min((-2, -1), keepdims=True)
            data_2d_norm = (data_2d - dpmin) / (dpmax - dpmin)
            # Correct any nan value
            nan_mask = np.isnan(data_2d_norm)
            data_2d_norm[nan_mask] = 0

            a_vals = vs.data_augmentation_parameters.background_parameters.a_vals
            tau_vals = vs.data_augmentation_parameters.background_parameters.tau_vals
            # Data amplification with a random subset of the param space
            if add_background_to_simulation == 'random':
                nav_shape = data_2d_norm.shape[:-2]
                nav_size = math.prod(nav_shape)

                tau = random.choices(tau_vals, k=nav_size)
                a = random.choices(a_vals, k=nav_size)
                tau = np.reshape(tau, nav_size)
                a = np.reshape(a, nav_size)
                # Flatten the navigation axis
                data_norm_1dnav = np.reshape(data_2d_norm, (nav_size, data_2d_norm.shape[-2], data_2d_norm.shape[-1]))
                bkg_data = add_background_to_signal_array(data_norm_1dnav, qx_axis, a, tau, dimensions=2)
                bkg_data = np.reshape(bkg_data, data_2d_norm.shape)
                if include_also_non_bkg_simulation:
                    data_2d = da.hstack((data_2d_norm, bkg_data))
                else:
                    data_2d = bkg_data
            else:
                for i, a in enumerate(a_vals):
                    for tau in tau_vals:
                        a, tau = np.array([a]), np.array([tau])
                        bkg_data = add_background_to_signal_array(data_2d_norm, qx_axis, a, tau, dimensions=2)
                        if i == 0:
                            data_2d = da.hstack((data_2d_norm, bkg_data))
                        else:
                            data_2d = da.hstack((data_2d, bkg_data))

        # TODO: Add crop in k space
        if crop_in_k:
            raise NotImplementedError("Crop in k units in 2D is not implemented.")
            # ## Crop, rebin and normalise on pixel coords
            # # # Crop in pixel units:
            # if calibration_modify_percent == (None or 0):
            #     raise NotImplementedError("Crop in k units in 2D is not implemented.")
            # else:
            #     # Calculate how many pixels the calibration tolerance factor corresponds to
            #     detector_size = vs.detector_geometry.detector_size
            #     radial_steps = int(np.ceil((int(detector_size / 2) - 1) / 2) * 2)
            #     one_px_q = qx_axis.max() / radial_steps
            #     max_q_range = calibration_modify_percent / 100 * qx_axis.max()
            #     max_px_shift = max_q_range / one_px_q
            #     range_percents_modification = np.arange(-max_px_shift, max_px_shift + 1, 1)
            #
            #     signals_temp_2d = da.array([])
            #     for phase_2d in data_2d:
            #         for ori_2d in phase_2d:
            #
            #             px_shift = random.choice(range_percents_modification) / 100
            #             temp = ori_2d[np.ceil(cropping_start_px + px_shift): np.ceil(cropping_stop_px + px_shift), :]
            #             try:
            #                 signals_temp_2d = da.vstack((signals_temp_2d, temp))
            #             except ValueError:
            #                 signals_temp_2d = da.array(temp)
            #
            #     # Reshape back
            #     shape_ori = list(data_2d.shape)
            #     shape_ori[-2] = cropped_signal_k_points
            #     data_2d_px = da.reshape(signals_temp_2d, shape_ori)
            #
            # # Renormalise
            # dpmax = data_2d_px.max([-2, -1])
            # data_2d_px = data_2d_px / dpmax[:, :, np.newaxis, np.newaxis]
            # # Correct any nan value
            # nan_mask = np.isnan(data_2d_px)
            # data_2d_px[nan_mask] = 0
            #
            # # NN Requirements: Reshape and Labelling
            # data_2d_px = data_2d_px.reshape(-1, data_2d_px.shape[-2], data_2d_px.shape[-1])
        else:
            data_2d_k = None

        if crop_in_px:
            ## Crop, rebin and normalise on pixel coords
            data_2d_px = data_2d[:, :, cropping_start_px: cropping_stop_px, :]

            # Renormalise
            dpmax = data_2d_px.max((-2, -1), keepdims=True)
            dpmin = data_2d_px.min((-2, -1), keepdims=True)
            data_2d_px = (data_2d_px - dpmin) / (dpmax - dpmin)

            # Correct any nan value
            nan_mask = np.isnan(data_2d_px)
            data_2d_px[nan_mask] = 0

            # NN Requirements: Reshape and Labelling
            data_2d_px = data_2d_px.reshape(-1, data_2d_px.shape[-2], data_2d_px.shape[-1])
        else:
            data_2d_px = None

        # NN Requirements: Reshape and Labelling
        data_2d = data_2d.reshape(-1, data_2d.shape[-2], data_2d.shape[-1])

        # Create labels for 2D
        n_phases = len(phase_dict)
        labels_2d = np.zeros((n_phases, int(data_2d.shape[0] / n_phases)))
        for i in range(n_phases):
            labels_2d[i, :] = i
        labels_2d = labels_2d.flatten()

        if not save_full_scan:
            data_2d = None
    else:
        data_2d, data_2d_px, labels_2d = None, None, None

    return data_2d, data_2d_px, labels_2d


@op()
def save_metadata_to_mongodb(vs):
    save_md_to_mongodb = vs.postprocessing_parameters.save_md_to_mongodb
    id = vs.id
    setattr(vs, '_id', id)
    if save_md_to_mongodb:
        db_collection = connect_to_mongo_database('simulations_db', 'sim')
        # Pass in the json input file here
        db_collection.insert_one(json.loads(json.dumps(vs, default=lambda o: o.__dict__)))
    return

#%%
# Overall operations
@graph(out={"data": GraphOut(), "data_2d": GraphOut(), "data_peak_pos": GraphOut(), "qx_axis": GraphOut(),})
def simulate_diffraction_data(vs, phase_dict, euler_list_n):

    n_ori_chunk, radial_integration_1d, radial_integration_2d, save_peak_position_library = get_metadata(vs)

    data = da.array([])
    data_2d = da.array([])
    data_peak_pos = {}

    chunks = get_chunks(euler_list_n, phase_dict, n_ori_chunk)
    chunk_rad_data, chunk_rad_data_2d, chunk_peak_pos, qx_axis = chunks.map(lambda chk: process_each_chunk(chk, vs))

    qx_axis = get_qx_axis_array(qx_axis.collect())

    data = merge_dict_chunk_to_dask_arr(chunk_rad_data.collect(), phase_dict, radial_integration_1d)

    data_2d = merge_dict_chunk_to_dask_arr(chunk_rad_data_2d.collect(), phase_dict, radial_integration_2d)

    data_peak_pos = merge_peak_position_dictionary(chunk_peak_pos.collect(), phase_dict, save_peak_position_library)

    get_dagster_logger().info(f'Simulation of data completed. Shape of 1d : {data}')

    return {"data": data, "data_2d": data_2d, "data_peak_pos": data_peak_pos, "qx_axis": qx_axis,}


@graph(out={"data": GraphOut(), "data_k": GraphOut(), "data_px": GraphOut(), "labels": GraphOut(), "data_2d": GraphOut(), "data_2d_px": GraphOut(), "labels_2d": GraphOut()})
def post_processing(vs, data, data_2d, qx_axis, phase_dict):
    data, data_k, data_px, labels = post_processing_1d(vs, data, qx_axis, phase_dict)
    data_2d, data_2d_px, labels_2d = post_processing_2d(vs, data_2d, qx_axis, phase_dict)

    return {"data": data, "data_k": data_k, "data_px": data_px, "labels": labels, "data_2d": data_2d, "data_2d_px": data_2d_px, "labels_2d": labels_2d}

@op(out={"vs": Out()})
def save_simulation(vs, data, data_k, data_px, labels, data_2d, data_2d_px, labels_2d, data_peak_pos, phase_dict, qx_axis, json_vars_dump):
    calibration_modify_percent = vs.calibration_parameters.calibration_modify_percent
    radial_integration_1d = vs.detector_geometry.radial_integration_1d
    radial_integration_2d = vs.detector_geometry.radial_integration_2d
    relrod_list = vs.relrod_parameters.relrod_list
    sigma_2d_gaussian_list = vs.sigma_parameters.sigma_2d_gaussian_list
    n_phases = len(phase_dict)
    phase_names = [key for key in phase_dict.keys()]
    save_peak_position_library = vs.detector_geometry.save_peak_position_library
    save_full_scan = vs.postprocessing_parameters.save_full_scan
    crop_in_k = vs.postprocessing_parameters.crop_in_k
    crop_in_px = vs.postprocessing_parameters.crop_in_px

    # Create a name string with summary
    calibration_str = f"{vs.calibration}".replace('.', 'p')
    if calibration_modify_percent is not None:
        calibration_str += f"mod{calibration_modify_percent}percent".replace('.', 'p')
    relrod_str = f"{relrod_list}".replace('[', '').replace(']', '').replace(', ', '-').replace('.', 'p')
    sigma_2d_str = f"{sigma_2d_gaussian_list}".replace('[', '').replace(']', '').replace(', ', '-').replace('.',
                                                                                                            'p')
    full_name = '1D_simulated_data_cal{}_cropK_randrelrod{}_randsigma2d{}_{}classes_orix_{}neuler_peakremoval_{}npeaks_random'.format(
        calibration_str,
        relrod_str,
        sigma_2d_str,
        n_phases,
        vs.orientations_parameters.n_points,
        vs.data_augmentation_parameters.peak_removal.n_intensity_peaks)

    get_dagster_logger().info(full_name)

    # Get timestamp
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setattr(vs, 'timestamp', time_stamp)

    # Create folder
    save_folder_path = os.path.abspath(vs.save_abspath)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # Save all sorts of useful files
    import h5py
    id_name = f'{vs.id}.hdf5'
    save_path = os.path.join(save_folder_path, id_name)
    with h5py.File(save_path, 'w') as f:
        if radial_integration_1d:
            g = f.create_group('1d')
            if save_full_scan:
                g.create_dataset('x_all', data=data)
                g.create_dataset('y_all', data=labels)
            if crop_in_k:
                g.create_dataset('x_q', data=data_k)
                g.create_dataset('y_q', data=labels)
            if crop_in_px:
                g.create_dataset('x_px', data=data_px)
                g.create_dataset('y_px', data=labels)
            g.create_dataset('x_all_q_axis', data=qx_axis)

        if radial_integration_2d:
            g = f.create_group('2d')
            if save_full_scan:
                g.create_dataset('x_all', data=data_2d)
                g.create_dataset('y_all', data=labels_2d)
            if crop_in_px:
                g.create_dataset('x_px', data=data_2d_px)
                g.create_dataset('y_px', data=labels_2d)

        if save_peak_position_library:
            g = f.create_group('peaks_positions')
            g.create_dataset('peaks_positions_json', data=data_peak_pos)

        g = f.create_group('metadata')
        g.attrs['phases'] = phase_names
        g.attrs['summary'] = full_name
        g.attrs['id'] = f"{vs.id}"
        g.attrs['timestamp'] = time_stamp
        g.create_dataset('metadata_json', data=json_vars_dump)

        # Save in external drive
        ssh = start_shh_connection()
        with SCPClient(ssh.get_transport()) as scp:
            spc_folder = r"/rds/project/rds-hirYTW1FQIw/shared_space/jf631"
            spc_path = os.path.join(spc_folder, 'simulations', id_name)
            try:
                scp.put(save_path, spc_path)
            except SCPException:
                print("Could not save though SSH/SCP")
    return vs

@graph()
def simulate_diffraction():
    # Loading and reading all metadata
    vs, json_vars_dump = load_json_metadata_to_dict()
    vs = start_random_seeds(vs)
    calibration, vs = get_dif_calibration_value(vs)
    phase_dict, vs = create_phase_dict(vs)
    euler_list_n, vs = get_euler_angle_lists(vs)
    phase_dict, euler_list_n, vs = make_changes_to_add_bkg_phase(vs, phase_dict, euler_list_n)
    id, vs = get_unique_id(vs)

    # Simulate
    data, data_2d, data_peak_pos, qx_axis = simulate_diffraction_data(vs, phase_dict, euler_list_n)

    # Post-processing
    data, data_k, data_px, labels, data_2d, data_2d_px, labels_2d = post_processing(vs, data, data_2d, qx_axis, phase_dict)

    # Save files
    vs = save_simulation(vs, data, data_k, data_px, labels, data_2d, data_2d_px, labels_2d, data_peak_pos, phase_dict, qx_axis, json_vars_dump)
    # Send json database to mongodb
    save_metadata_to_mongodb(vs)

    get_dagster_logger().info(f"Simulation {id} complete")
