import random, os
import numpy as np
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator
seed = 10

# Functions that are useful for the simulation of diffraction patterns

def get_random_euler(npoints, n_phases):
    radius = 1
    np.random.seed(seed)
    u = np.random.randint(-100, 100 + 1, size=(npoints,)) / 100
    u2 = 2 * np.pi * np.random.random(size=(npoints,))
    theta = 2 * np.pi * np.random.random(size=(npoints,))
    x = radius * np.sqrt(1 - u ** 2) * np.cos(theta)
    y = radius * np.sqrt(1 - u ** 2) * np.sin(theta)
    z = radius * u
    phi = np.arccos(z / radius)
    eulerAlpha = u2
    eulerBeta = phi
    eulerGamma = theta
    euler = np.array([np.rad2deg(eulerAlpha), np.rad2deg(eulerBeta), np.rad2deg(eulerGamma)]).T
    euler_n = np.tile(euler, (n_phases, 1, 1))
    return euler_n


def load_orientation_list(orientation_file_path_list, orientation_file_folder, npoints):
    for i, f in enumerate(orientation_file_path_list):
        random.seed(seed)
        oris = np.load(os.path.join(orientation_file_folder, f))
        rand_indeces = random.choices(np.arange(0, oris.shape[0]), k=npoints)
        oris = oris[rand_indeces, :]

        if i == 0:
            euler_n = oris
        else:
            euler_n = np.vstack([euler_n, oris])

    new_shape = (len(orientation_file_path_list), len(oris), 3)
    euler_n = np.reshape(euler_n, new_shape)
    return (euler_n)


def get_reciprocal_radius(detector_size, calibration):
    half_pattern_size = detector_size // 2
    reciprocal_radius = calibration * half_pattern_size
    return reciprocal_radius


def create_diffraction_library(phase_dict, euler_list_n, beam_energy, scattering_params, relrod_length, calibration,
                               detector_size, with_direct_beam):

    phase_names = list(phase_dict.keys())
    phases = list(phase_dict.values())

    sample_lib = StructureLibrary(phase_names, phases, euler_list_n)
    ediff = DiffractionGenerator(beam_energy, scattering_params, relrod_length)
    diff_gen = DiffractionLibraryGenerator(ediff)

    reciprocal_radius = get_reciprocal_radius(detector_size, calibration)

    library = diff_gen.get_diffraction_library(sample_lib,
                                               calibration=calibration,
                                               reciprocal_radius=reciprocal_radius,
                                               half_shape=(detector_size // 2, detector_size // 2),
                                               with_direct_beam=with_direct_beam)
    return library

def remove_peaks(lib_entry_n, n_intensity_peaks, n_peaks_to_remove):
    high_intensity_peaks_list = np.argpartition(lib_entry_n.intensities, -n_intensity_peaks)[-n_intensity_peaks:]
    lowest_intensity = np.min(lib_entry_n.intensities)

    if n_peaks_to_remove == 'random':
        n_peaks_to_remove = random.choice(range(0, n_intensity_peaks + 1))
    elif type(n_peaks_to_remove) != int:
        raise AttributeError("n_peaks must be an integer or 'random'.")
    elif n_peaks_to_remove >= n_intensity_peaks:
        raise AttributeError("n_peaks_to_remove must be equal or less than n_intensity_peaks")

    if n_peaks_to_remove == 0:
        return lib_entry_n
    else:
        # Create a new DiffractionGenerator object
        # Get params needed to initialise new object
        temp_arr_intensities = lib_entry_n.intensities
        temp_arr_coordinates = lib_entry_n.coordinates
        temp_indices = lib_entry_n.indices
        temp_calibration = lib_entry_n.calibration
        temp_offset = lib_entry_n.offset
        temp_with_direct_beam = lib_entry_n.with_direct_beam

        # Apply modifications
        high_intensity_peaks_list_int = [int(x) for x in high_intensity_peaks_list]
        peaks_to_remove = random.sample((high_intensity_peaks_list_int), n_peaks_to_remove)
        for peak in peaks_to_remove:
            temp_arr_intensities[peak] = 0.

        from simulator_diffsims.sims.diffraction_simulation import DiffractionSimulation
        lib_temp = DiffractionSimulation(
            coordinates=temp_arr_coordinates,
            indices=temp_indices,
            intensities=temp_arr_intensities,
            calibration=temp_calibration,
            offset=temp_offset,
            with_direct_beam=temp_with_direct_beam,
        )

    return lib_temp


def get_coordinates_from_library(diffraction_library, intensity_q_threshold=0):
    # Returns an array of shape (n_peaks, 2)
    lib = list(diffraction_library.values())[0]['simulations'][0]
    all_coords = lib.coordinates[:, :2]
    intensities = lib.intensities
    mask_intensities = intensities >= np.quantile(intensities, q=intensity_q_threshold)
    return all_coords[mask_intensities]


def get_coordinates_dict_from_library(diffraction_library):
    lib_entry = list(diffraction_library.values())[0]['simulations'][0]
    coords = lib_entry.coordinates.tolist()
    indices = lib_entry.indices.tolist()
    intensities = lib_entry.intensities.tolist()
    orientation = list(diffraction_library.values())[0]['orientations'][0].tolist()
    coords_lib = {
        'coords': coords,
        'indices': indices,
        'intensities': intensities,
        'orientation': orientation,
    }
    return coords_lib