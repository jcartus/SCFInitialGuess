"""This script contains all transormations and enhancements 
to be applied to nn results, to enhance the guesses.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np

def mc_wheeny_purification(p,s):
    """ The McWheeny Prurification for an idempotent matrix p in a basis with
    overlaps S
    """
    return (3 * np.dot(np.dot(p, s), p) - np.dot(np.dot(np.dot(np.dot(p, s), p), s), p)) / 2

def multi_mc_wheeny(p, s, n_max=4):
    """The McWheeny Purification applied n_max times."""
    for i in range(n_max):
        p = mc_wheeny_purification(p, s)
    return p
