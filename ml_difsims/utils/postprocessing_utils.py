import numpy as np

def find_nearest_idx(array, value):
    "Element in nd array  closest to the scalar value"
    idx = np.abs(array - value).argmin()
    return idx


def rebin_signal(s, desired_n, axis=-1):
    rebin_scale = s.data.shape[axis] / desired_n
    scale_array = np.ones(shape=len(s.data.shape))
    scale_array[axis] = rebin_scale
    return s.rebin(scale=scale_array)