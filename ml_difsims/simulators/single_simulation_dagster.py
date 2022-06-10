from dagster import job
from ml_difsims.simulators.simulator_diffsims_dagster_ops import *
import json

json_vars_example = json.dumps({
        "root_path": 'G:\My Drive\PhD\projects\external_measurements\ml_difsims',
        "structure_parameters": {
            "phase_files_location_from_root": "models/crystal_phases",
            "phase_files": ['p4mbm_scaled_mixed_halide.cif'], #, 'gratia_2h.cif', 'pbi2_2h.cif'],
            "add_bkg_phase": False,
            # Do you want to add a bkg/just noise phase at the end? If True, the final datasets will be phases + 1 shape.
            "scattering_params": 'lobato',
        },
        "calibration_parameters": {
            "calibration_value": [0.00588,],
            #  List of 1 value only for now
            "calibration_modify_percent": 5,
            # It will disturb the calibration value by % when cropping in the q space. In None, nothing happens.
        },
        "orientations_parameters": {
            "n_points": 10,
            "use_orix_sampling": True,
            "ori_files_location_from_root": "models/orix_orientation_full_lists",
            "orientation_files_list": ['orientations_pg422_3_xxxx.npy',
                                       'orientations_pg622_3_xxxx.npy',
                                       'orientations_pg32_3_xxxx.npy'],
            "orientation_sampling_mode": ['cubo', 'euler', 'quat']
        },
        "data_augmentation_parameters": {
            "peak_removal": {
                "remove_peaks_from_diffraction_library": True,
                "n_intensity_peaks": 20,
                # Finds n brightest peaks to remove from. n_intensity_peaks >= n_peaks_to_remove
                "num_peaks_to_remove": 'random',
                # Removes n amount of peaks of the brightest ones.
                # If 'random' is passed, it will randomise this value between 0 and n_intensity_peaks.
            },
            "noise_addition": {
                "add_noise": 'random',
                # Select from bool or 'random'
                "include_also_non_noisy_simulation": False,
                # If add noise, do you want to also have the non-noisy data?
                "snrs": [0.9, 0.99],
                "intensity_spikes": [0, 0.25, 0.50],
            },
            "background_parameters": {
                "add_background_to_1d": 'random',
                # Select from bool or 'random'
                "include_also_non_bkg_simulation": False,
                "a_vals": [0, 1., 5.],
                # A: pre-exp factor, tau: decay time constant
                "tau_vals": [0.5, 1.5],
            },
            "simulated_direct_beam_bool": False,
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
            "detector_type": "Medipix515x515Detector",
            "radial_integration_1d": True,
            "radial_integration_2d": False,
            "save_peak_position_library": False,
        },
        "postprocessing_parameters": {
            "crop_in_k": False,
            "crop_in_px": True,
            "save_full_scan": True,
            "cropping_start_k": 0.11,
            "cropping_stop_k": 1.30,
            "cropped_signal_k_points": 147,
            # To rebin signal, if necessary (when using k_units)
            "cropping_start_px": 13,
            "cropping_stop_px": 160,
            "sqrt_signal": True,
            "save_md_to_mongodb": True,
        },
        "random_seed": 10,
        "save_relpath": 'data/simulations',
        "simulation_orientations_chunk_size" : 5,
    })


@job(op_retry_policy=default_policy, config={'ops': {'simulate_diffraction': {'ops': {'load_json_metadata_to_dict': {'config': {'json_string': json_vars_example}}}}},
                                             #"execution": {"config": {"multiprocess": {"max_concurrent": 4,}}}
     },)
def simulate_diffraction_single():
    simulate_diffraction()


if __name__ == "__main__":
    result = simulate_diffraction_single.execute_in_process()

