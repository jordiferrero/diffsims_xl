import os, glob
from dagster import SkipReason, RunRequest, sensor, DefaultSensorStatus, repository, SensorDefinition
from ml_difsims.pre_processing.pre_processing_ops import *
import numpy as np

config_dict = {
    # Raw data files path (saved within folders of folders:
    # exp_data_root (exp_name)
    #   -> sample_type_name
    #       -> single scan number
    #           -> all raw files

    #'exp_data_root': 'D:/Data/jf631/simulations_diffsims_ml/experimental',
    'exp_data_root': "D:\Data\jf631\simulations_diffsims_ml\experimental",
    'file_wildcard': '**/2*_data.hdf5',

    # Save processed data (saved within folders of folders):
    # processed_exp_data_root
    #   -> processing step (npz, radial, centered_corrected_hspy)
    #      -> sample_type_name
    #           -> all processed files with their unique ID
    'processed_exp_data_root': "D:\Data\jf631\simulations_diffsims_ml\experimental_processed",

    # Additional files to save
    'save_full_hspy_dp': False,
    'save_full_hspy_rebin_dp': True,
    'save_full_hspy_radial': True,
    'save_crop_npz_radial': True,

    # Calibration files
    'recip_cal': 0.005154,
    'nav_axis_cal_dict' : {
                        '80000' : 9.58,
                        '100000' : 7.76,
                        '150000' : 5.13,
                        '200000' : 3.87,
                        '250000': 3.09,
                        '300000' : 2.57,
                        '600000' : 1.29,
                    },

    # Affine transform
    'affine_matrix': [[ 0.98051804, -0.01322819,  0.        ],
                      [-0.01322819,  0.9910181 ,  0.        ],
                      [ 0.        ,  0.        ,  1.        ]],

    'rotation_angle': 73.13, #deg
    'rotation_correction': False,

    # Load a mask to cover the dead pixels and the joints between detectors
    # Can be a path or a str:"with_exp_datafiles" (will search for it in the exp data folders.
    'mask_path':  "with_exp_datafiles",

    # Parameters
    # Threshold pixel intensity. It can be 0 or an integer.
    'threshold_px_intensity': 0,
    'sqrt_signal': False,

    # Rebin any of the axes?
    'rebin_nav': 1,
    'rebin_dp': 1,

    # Select which radial integrations to save
    'radial_integration_1d': True,
    'radial_integration_2d': True,
    'beam_energy': 200.,

    # Post processing parameters
    'crop_in_px': True,
    'cropping_start_px': 13,
    'cropping_stop_px': 160,
    # This is to correct for the mismatch in simulation/experimental radial integration
    'q_range_from_px_cropped_simulation': [0.10777668889613681, 1.318191810345058],

}

def pre_processing_sensor() -> SensorDefinition:
    """
    Returns a sensor that launches.
    """
    @sensor(job=pre_process_experimental_file, minimum_interval_seconds=30, default_status=DefaultSensorStatus.RUNNING)
    def my_directory_sensor():
        has_new_files = False
        ROOT = config_dict['exp_data_root']
        ROOT_PROCESSED = config_dict['processed_exp_data_root']
        file_wildcard = config_dict['file_wildcard']

        path = os.path.join(ROOT, file_wildcard)
        paths = glob.glob(path, recursive=True)
        paths.sort()

        for raw_file_path in paths[:1]:
            # Get the unique name of each raw file
            f_name = os.path.basename(raw_file_path).split('.')[0]

            # Search if the raw file has been fully processed/completed
            completed_file_path = os.path.join(ROOT_PROCESSED, 'npz_radial_crop', f"**/*{f_name}*.npz")
            completed_file_path = glob.glob(completed_file_path, recursive=True)
            if completed_file_path != []:
                continue
            else:
                yield RunRequest(
                    run_key=raw_file_path,
                    run_config={
                        "ops": {"get_path": {"config": {"file_path": raw_file_path}},
                                #"get_metadata_dict": {"config": {"md_dict": config_dict}}
                                }
                    },
                )
                has_new_files = True
        if not has_new_files:
            yield SkipReason(f"No files found in {ROOT} that are not processed already.")

    return my_directory_sensor

@repository
def pre_processing_prod():
    return [pre_processing_sensor()]
