import os, glob
import warnings
from typing import Tuple
import pyxem as pxm
import h5py
from dagster import op, job, Out, graph, GraphOut, get_dagster_logger, fs_io_manager
import numpy as np
from types import SimpleNamespace

from scipy import interpolate

from ml_difsims.pre_processing.hspy_io_manager import hyperspy_io_manager
from ml_difsims.simulators.simulator_diffsims_dagster_ops import set_detector_on_dp_object, radially_integrate_1d, \
    radially_integrate_2d

@op(config_schema={"file_path": str}, out={"file_path": Out(io_manager_key="fs")})
def get_path(context) -> str:
    file_path = context.op_config["file_path"]
    context.log.info(file_path)
    return file_path

@op(config_schema={"md_dict": dict}, out={"file_path": Out(io_manager_key="fs")})
def get_metadata_dict(context) -> dict:
    md_dict = context.op_config["md_dict"]
    return md_dict

@op(out={"exp_name": Out(io_manager_key="fs"), "sample_name": Out(io_manager_key="fs"), "scan_id": Out(io_manager_key="fs"),})
def get_sample_names(file_path) -> Tuple[str, str, str]:
    # Get absolute path elements
    p_elements = os.path.abspath(file_path).split('\\')

    # Get experiment name
    exp_name = p_elements[-4]
    # Get sample type name
    sample_name = p_elements[-3]
    # Get the unique scan number of the raw file
    scan_id = p_elements[-2]

    return exp_name, sample_name, scan_id

@op(out={"dp_masked": Out(io_manager_key="hspy")})
def apply_mask(dp, md_dict):
    mask_path = md_dict['mask_path']
    if mask_path == "with_exp_datafiles":
        root = md_dict['exp_data_root']
        p = os.path.join(root, "**/mask*.npy")
        p = glob.glob(p, recursive=True)
        if len(p) != 1:
            raise AttributeError(f"{len(p)} masks found within the exp data directory. Only one should be. Consider providing the direct path link instead of 'with_exp_datafiles'")
        else:
            mask = np.load(p[0])
    else:
        try:
            mask = np.load(mask_path)
        except FileNotFoundError:
            raise FileNotFoundError

    print(dp)
    print(mask)
    return dp * mask

@op(out={"n_shifts": Out(io_manager_key="fs")})
def get_shifts_from_mean_dp(dp) -> np.ndarray:
    mean_dp = dp.mean()
    mean_dp = pxm.signals.electron_diffraction2d.ElectronDiffraction2D(mean_dp)
    centre = mean_dp.get_direct_beam_position(method='cross_correlate', radius_start=1, radius_finish=10)
    shifts = [[centre.data[0], centre.data[1]]]

    # Create shifts array and align and centre
    n_shifts = shifts * dp.axes_manager.navigation_shape[0] * dp.axes_manager.navigation_shape[1]
    n_shifts = np.array(n_shifts)
    new_shape = (dp.axes_manager.navigation_shape[1], dp.axes_manager.navigation_shape[0], 2)
    n_shifts = n_shifts.reshape(new_shape)

    return n_shifts

@op(out={"dp": Out(io_manager_key="hspy")})
def load_file(file_path) -> pxm.signals.electron_diffraction2d.ElectronDiffraction2D:
    # Load file
    f = h5py.File(file_path, 'r')['Experiments/__unnamed__/data']
    dp = pxm.signals.electron_diffraction2d.ElectronDiffraction2D(np.array(f))

    # Reset the offset
    dp.axes_manager.navigation_axes[0].offset = 0
    dp.axes_manager.navigation_axes[1].offset = 0
    dp.axes_manager.signal_axes[0].offset = 0
    dp.axes_manager.signal_axes[1].offset = 0

    return dp

@op(out={"dp": Out(io_manager_key="hspy")})
def align_and_fine_center(dp, n_shifts) -> pxm.signals.electron_diffraction2d.ElectronDiffraction2D:
    dp.align2D(shifts=-n_shifts, crop=False)
    dp.center_direct_beam(method='interpolate', sigma=5, upsample_factor=4, kind='linear', half_square_width=10)
    return dp

@op(out={"dp": Out(io_manager_key="hspy")})
def apply_affine_transform(dp, md_dict) -> pxm.signals.electron_diffraction2d.ElectronDiffraction2D:
    transform_mat = md_dict['affine_matrix']
    transform_mat = np.array(transform_mat)
    dp.apply_affine_transformation(transform_mat, keep_dtype=True)
    return dp

@op(out={"dp": Out(io_manager_key="hspy")})
def set_nav_calibration(dp, md_dict, file_path) -> pxm.signals.electron_diffraction2d.ElectronDiffraction2D:
    def get_magnification_from_file(metadata_fname):
        md = h5py.File(metadata_fname, 'r')
        mag = np.array(md['metadata/magnification'])
        return mag

    md_fname = file_path.replace("_data.hdf5", ".hdf")
    nav_axis_cal_dict = md_dict['nav_axis_cal_dict']
    mag = get_magnification_from_file(md_fname)
    nav_cal = nav_axis_cal_dict[str(mag)]
    dp.set_scan_calibration(nav_cal)
    return dp

@op(out={"dp": Out(io_manager_key="hspy")})
def set_dif_calibration(dp, md_dict) -> pxm.signals.electron_diffraction2d.ElectronDiffraction2D:
    recip_cal = md_dict['recip_cal']
    dp.set_diffraction_calibration(recip_cal)
    return dp

@op(out={"dp": Out(io_manager_key="hspy")})
def rotation_correction(dp, md_dict) -> pxm.signals.electron_diffraction2d.ElectronDiffraction2D:
    bool_correct_rot = md_dict['rotation_correction']
    rotation_angle = md_dict['rotation_angle']
    if bool_correct_rot:
        raise NotImplementedError(f"Rotation correction is not implemented yet. Rotation angle is set to {rotation_angle}")
    return dp

@op(out={"dp": Out(io_manager_key="hspy")})
def threshold_signal(dp, md_dict) -> pxm.signals.electron_diffraction2d.ElectronDiffraction2D:
    threshold_px_intensity = md_dict['threshold_px_intensity']
    if threshold_px_intensity > 0:
        # Threshold the pixels with counts 1
        dp.data[dp.data == threshold_px_intensity] = 0

    return dp

@op(out={"md": Out(io_manager_key="fs")})
def create_simplenamespace_object(md_dict) -> SimpleNamespace:
    beam_energy = md_dict['beam_energy']
    recip_cal = md_dict['recip_cal']

    md = SimpleNamespace()
    setattr(md, 'beam_energy', beam_energy)
    setattr(md, 'calibration', recip_cal)
    return md

@op(out={"dp_1d": Out(io_manager_key="hspy"), "dp_2d": Out(io_manager_key="hspy")})
def radially_integrate(dp, vs, md_dict):

    dp, vs = set_detector_on_dp_object(vs, dp)
    dp_1d, dp_2d = None, None

    bool_integration_1d = md_dict['radial_integration_1d']
    bool_integration_2d = md_dict['radial_integration_2d']

    if bool_integration_1d:
        dp_1d = radially_integrate_1d(dp, vs)
    if bool_integration_2d:
        dp_2d = radially_integrate_2d(dp, vs)

    return dp_1d, dp_2d


def interpolate_1d(signal_data, q_array, crop_range_q, crop_size):
    # Do interpolation
    x = q_array
    y = signal_data
    f = interpolate.interp1d(x, y, fill_value='extrapolate')

    # Generate new data
    x_new = np.linspace(crop_range_q[0], crop_range_q[1], crop_size)
    y_interpol = f(x_new)
    return y_interpol


def interpolate_2d(signal_data, q_array, crop_range_q, crop_size):
    # Transpose data (so we can iterate though angular axis of 360)
    signal_data = signal_data.T
    y_interpol_2d = \
        np.vstack([interpolate_1d(row, q_array, crop_range_q, crop_size)
                   for row in signal_data])
    # Transpose back
    return y_interpol_2d.T

@op(out={"dp_1d_crop": Out(io_manager_key="hspy"), "dp_2d_crop": Out(io_manager_key="hspy"), "q_new": Out(io_manager_key="fs")})
def crop_px_interpolate(dp_1d, dp_2d, md_dict):
    dp_1d_crop, dp_2d_crop = None, None

    crop_in_px = md_dict['crop_in_px']
    cropping_start_px = md_dict['cropping_start_px']
    cropping_stop_px = md_dict['cropping_stop_px']
    crop_range_q = md_dict['q_range_from_px_cropped_simulation']

    if crop_in_px:
        crop_size = cropping_stop_px - cropping_start_px # In pixels
        q_new = np.linspace(crop_range_q[0], crop_range_q[1], crop_size)

        if dp_1d is not None:
            # Get the experimental q range
            q_exp = dp_1d.axes_manager.signal_axes[0].axis
            if q_exp.min() > crop_range_q[0] or q_exp.max() < crop_range_q[1]:
                warnings.warn(
                    "The range at which signal was acquired is not large enough. Extrapolation will be used using scipy.interpolate.interp1d")

            dp_1d_crop = dp_1d.map(interpolate_1d, q_array=q_exp, crop_range_q=crop_range_q, crop_size=crop_size, parallel=True, inplace=False)

            # Correct for axes calibration
            sig_ax = dp_1d_crop.axes_manager.signal_axes[0]
            sig_ax.offset = crop_range_q[0]
            sig_ax.scale = (crop_range_q[1] - crop_range_q[0]) / crop_size

        if dp_2d is not None:
            # Get the experimental q range
            q_exp = dp_2d.axes_manager.signal_axes[1].axis
            if q_exp.min() > crop_range_q[0] or q_exp.max() < crop_range_q[1]:
                warnings.warn(
                    "The range at which signal was acquired is not large enough. Extrapolation will be used using scipy.interpolate.interp1d")

            dp_2d_crop = dp_2d.map(interpolate_2d, q_array=q_exp, crop_range_q=crop_range_q, crop_size=crop_size,
                                   parallel=True, inplace=False)

            # Correct for axes calibration
            sig_ax = dp_2d_crop.axes_manager.signal_axes[1]
            sig_ax.offset = crop_range_q[0]
            sig_ax.scale = (crop_range_q[1] - crop_range_q[0]) / crop_size

    return dp_1d_crop, dp_2d_crop, q_new


@op(out={"dp_1d_crop": Out(io_manager_key="hspy"), "dp_2d_crop": Out(io_manager_key="hspy")})
def normalise_data(dp_1d_crop, dp_2d_crop):
    if dp_1d_crop is not None:
        dpmax = dp_1d_crop.data.max(-1, keepdims=True)
        dpmin = dp_1d_crop.data.min(-1, keepdims=True)
        dp_1d_crop = (dp_1d_crop - dpmin) / (dpmax - dpmin)
        # Correct any nan value
        nan_mask = np.isnan(dp_1d_crop)
        dp_1d_crop[nan_mask] = 0

    if dp_2d_crop is not None:
        dpmax = dp_2d_crop.data.max((-2, -1), keepdims=True)
        dpmin = dp_2d_crop.data.min((-2, -1), keepdims=True)
        dp_2d_crop = (dp_2d_crop - dpmin) / ( dpmax - dpmin)
        # Correct any nan value
        nan_mask = np.isnan(dp_2d_crop)
        dp_2d_crop[nan_mask] = 0

    return dp_1d_crop, dp_2d_crop


@op(out={"dp_1d": Out(io_manager_key="hspy"), "dp_2d": Out(io_manager_key="hspy")})
def apply_sqrt_signal(dp_1d, dp_2d, md_dict):
    sqrt_signal = md_dict['sqrt_signal']
    if sqrt_signal:
        if dp_1d is not None:
            dp_1d.data = np.sqrt(dp_1d.data)
        if dp_2d is not None:
            dp_2d.data = np.sqrt(dp_2d.data)

    return dp_1d, dp_2d

@op(out={"dp_rebin": Out(io_manager_key="hspy"),})
def rebin_signal_dp(dp, md_dict):
    dp_crop_nav, dp_crop_dp = None, None

    rebin_nav = md_dict['rebin_nav']
    rebin_dp = md_dict['rebin_dp']

    # Crop dp into divisible shape
    if rebin_nav > 1:
        reminder = [a % rebin_nav for a in dp.axes_manager.navigation_shape]
        dp = dp.inav[reminder[0]:, reminder[1]:]

    if rebin_dp > 1:
        reminder = [a % rebin_dp for a in dp.axes_manager.signal_shape]
        dp = dp.isig[reminder[0]:, reminder[1]:]

    # Rebin all
    dp_rebin = dp.rebin(scale=[rebin_nav, rebin_nav, rebin_dp, rebin_dp])
    return dp_rebin

@op()
def save_files(file_path, md_dict, dp, dp_rebin, dp_1d, dp_2d, dp_1d_crop, dp_2d_crop, q_new) -> None:
    # Find out which files to save
    save_full_hspy_dp = md_dict["save_full_hspy_dp"]
    save_full_hspy_rebin_dp = md_dict["save_full_hspy_rebin_dp"]
    save_full_hspy_radial = md_dict["save_full_hspy_radial"]
    save_crop_npz_radial = md_dict["save_crop_npz_radial"]

    exp_name, sample_name, scan_id = get_sample_names(file_path)

    # Create save path for processed data
    root_processed = md_dict["processed_exp_data_root"]

    save_hspy_folder = os.path.join(root_processed, 'hspy_files', )
    # Make folder if not exists
    if not os.path.exists(save_hspy_folder):
        os.makedirs(save_hspy_folder)

    if save_full_hspy_dp:
        save_file =f"{exp_name}/{sample_name}_{scan_id}.hspy"
        dp.save(os.path.join(save_hspy_folder, save_file), overwrite=True)

    if save_full_hspy_rebin_dp:
        save_file =f"{exp_name}/{sample_name}_{scan_id}_rebin.hspy"
        dp_rebin.save(os.path.join(save_hspy_folder, save_file), overwrite=True)

    if save_full_hspy_radial:
        if dp_1d is not None:
            save_file =f"{exp_name}/{sample_name}_{scan_id}_radial_1d.hspy"
            dp_1d.save(os.path.join(save_hspy_folder, save_file), overwrite=True)

        if dp_2d is not None:
            save_file = f"{exp_name}/{sample_name}_{scan_id}_radial_2d.hspy"
            dp_2d.save(os.path.join(save_hspy_folder, save_file), overwrite=True)

    if save_crop_npz_radial:
        save_npz_folder = os.path.join(root_processed, 'npz_radial_crop',)
        # Make folder if not exists
        if not os.path.exists(save_npz_folder):
            os.makedirs(save_npz_folder)
        # Save files
        if dp_1d_crop is not None:
            save_file = f"{exp_name}/{sample_name}_{scan_id}_1d.npz"
            np.savez(os.path.join(save_npz_folder, save_file),
                     y=dp_1d_crop.data, x=q_new)

            save_file = f"{exp_name}/{sample_name}_{scan_id}_radial_crop_1d.hspy"
            dp_1d_crop.save(os.path.join(save_hspy_folder, save_file), overwrite=True)
        if dp_2d_crop is not None:
            save_file = f"{exp_name}/{sample_name}_{scan_id}_2d.npz"
            np.savez(os.path.join(save_npz_folder, save_file),
                     y=dp_2d_crop.data, x=q_new)

            save_file = f"{exp_name}/{sample_name}_{scan_id}_radial_crop_2d.hspy"
            dp_2d_crop.save(os.path.join(save_hspy_folder, save_file), overwrite=True)
    return

@graph(out={"dp": GraphOut()})
def load_and_beam_center_ops(file_path, md_dict):
    # Load file
    dp = load_file(file_path)

    # Apply mask
    dp = apply_mask(dp, md_dict)

    # Get mean diffraction pattern to centre from there
    n_shifts = get_shifts_from_mean_dp(dp)

    # Align and fine tune
    dp = align_and_fine_center(dp, n_shifts)
    return {"dp": dp}

@graph(out={"dp": GraphOut()})
def apply_corrections_and_calibrations_ops(dp, md_dict, file_path):

    # Apply affine transforms (affine)
    dp = apply_affine_transform(dp, md_dict)

    # Set calibrations
    dp = set_nav_calibration(dp, md_dict, file_path)
    dp = set_dif_calibration(dp, md_dict)

    dp = rotation_correction(dp, md_dict)

    # Threshold data
    dp = threshold_signal(dp, md_dict)

    return {"dp": dp}

@graph(out={"dp_1d": GraphOut(), "dp_2d": GraphOut(), "dp_1d_crop": GraphOut(), "dp_2d_crop": GraphOut(), "q_new": GraphOut()})
def radial_integration_ops(dp, md_dict):

    vs = create_simplenamespace_object(md_dict)

    # Radial integration
    dp_1d, dp_2d = radially_integrate(dp, vs, md_dict)

    # Apply sqrt of signal
    dp_1d, dp_2d = apply_sqrt_signal(dp_1d, dp_2d, md_dict)

    # Crop, interpolate and normalise
    dp_1d_crop, dp_2d_crop, q_new = crop_px_interpolate(dp_1d, dp_2d, md_dict)
    dp_1d_crop, dp_2d_crop = normalise_data(dp_1d_crop, dp_2d_crop)

    return {"dp_1d":dp_1d, "dp_2d":dp_2d, "dp_1d_crop":dp_1d_crop, "dp_2d_crop":dp_2d_crop, "q_new":q_new,}

@job(resource_defs={"hspy": hyperspy_io_manager,
                    "fs": fs_io_manager})
def pre_process_experimental_file():
    file_path = get_path()
    md_dict = get_metadata_dict()

    dp = load_and_beam_center_ops(file_path, md_dict)
    dp = apply_corrections_and_calibrations_ops(dp, md_dict, file_path)
    dp_1d, dp_2d, dp_1d_crop, dp_2d_crop, q_crop_new = radial_integration_ops(dp, md_dict)

    dp_rebin = rebin_signal_dp(dp, md_dict)

    # Save all files
    save_files(file_path, md_dict, dp, dp_rebin, dp_1d, dp_2d, dp_1d_crop, dp_2d_crop, q_crop_new)

    get_dagster_logger().info("Processed data completed")
    return
