"""This file contains various coordinate_descriptor models.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""


import numpy as np

#-------------------------------------------------------------------------------
#   Quantity descriptros (coordinates to atomic contribution)
#-------------------------------------------------------------------------------

# a collection of guassian positioning models (cut_off, r_s, eta)
RADIAL_GAUSSIAN_MODELS = {
    "Origin-Centered_1": (
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.2, 0.5, 0.2, 0.1, 0.01]
    ),

    "Equidistant-Broadening_1": (
        [0.6, 1.1, 1.6, 2.1, 2.6],
        [0.9, 0.7, 0.6, 0.5, 0.4]
    ),

    "Man": (
        list(np.linspace(0.2, 0.7, 11))[:-1] + \
        list(np.linspace(0.7, 2.5, 30)) + \
        list(np.linspace(2.5, 4, 11))[1:],
        #
        [500]*10 + \
        [1000]*30 + \
        [200]*10,
    )


}

AZIMUTHAL_GAUSSIAN_MODELS = {
    # 8 gaussians equally distributed, litte overlap
    "Equisitant_1": (
        np.arange(1, 8 + 1) * 2 * np.pi / (8+1),
        [2 * np.pi / 8]*5,
        2 * np.pi
    )
}


POLAR_GAUSSIAN_MODELS = {
    # 2 gaussians equally distributed, little more overlap
    "Equisitant_1": (
        np.arange(1, 5 + 1) * np.pi / (5 + 1),
        [(5 / (np.pi))**2]*5,
        np.pi
    )
}

def make_uniform(n, R_c=5, eta_max=30, eta_min=10, R_min=0.1):
    return (
        np.linspace(R_min, R_c*0.8, n),
        np.linspace(eta_max, eta_min, n),
    )

def make_periodic_uniform(n, period):
    return (
        np.arange(n) * period / n,
        [1 / (2 * (0.5 * period / n)**2) ] * n,
        period
    )