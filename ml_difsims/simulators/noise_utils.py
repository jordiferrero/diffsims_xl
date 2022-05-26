import numpy as np

def add_noise_to_simulation(simulation_arr, snr, int_salt, ):
    import numpy as np
    # Salt and pepper
    def addsalt_pepper(dp_arr, snr, int_min=0, int_max=int_salt, ):
        p0 = snr
        # Add noise
        size = np.shape(dp_arr)
        mask = np.random.choice(a=(0, 1, 2),
                                size=size,
                                p=[p0, (1 - p0) / 2., (1 - p0) / 2.])

        im = dp_arr.copy()
        # im[mask == 1] = int_min # salt noise
        im[mask == 2] = int_max  # pepper noise
        return im

    # Add poisson noise on sp noise and normalise
    im = simulation_arr.copy()
    im += np.random.poisson(im)
    max = im.max()
    if max == 0:
        im = im
    else:
        im = im / im.max()
    # Add bright spots randomly accross detector
    im_sp = addsalt_pepper(im, snr, )
    return im_sp


def add_background_to_signal_array(normalised_sim_data_array, x_axis,
                                   a_val, tau_val, bkg_function='exp_decay', dimensions=1):
    """
    :param normalised_sim_data_array:
        The normalised 1d signal array (nav axis should be (points, phases, q))
    :param x_axis: array of the actual q values
        The A and tau values are optimised for 1/A-1 magnitude
    :return: extended signal with new sets of sim data without and with bakgrounds
    """

    def inv_q(x, A, tau):
        return A * x ** (-tau)

    def exp_decay(x, A, tau):
        return A * np.exp(- tau * x)

    if bkg_function == 'exp_decay':
        bkg = exp_decay(x_axis, a_val, tau_val)
    elif bkg_function == 'inv_q':
        bkg = inv_q(x_axis, a_val, tau_val)

    if dimensions == 1:
        return normalised_sim_data_array + bkg

    elif dimensions == 2:
        n = normalised_sim_data_array.shape[-1]
        bkg = np.tile(bkg, (n, 1)).T
        return normalised_sim_data_array + bkg