"""This script contains all transormations and enhancements 
to be applied to nn results, to enhance the guesses.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np

#-------------------------------------------------------------------------------
# Idempotence
#-------------------------------------------------------------------------------

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


#-------------------------------------------------------------------------------
# Rescaling for Occupance
#-------------------------------------------------------------------------------

#--- rescale whole matrix ---
def rescale(p, s, n_electrons):
    return p / np.trace(p @ s) * n_electrons

def rescale_batch(p_batch, s_batch, n_electrons):
    result = []
    for (p, s) in zip(p_batch, s_batch):
        result.append(rescale(p, s, n_electrons))
    return np.array(result).astype("float64")
#---


# This does not make a lot of sense!
#--- Rescale diagonal ---
def rescale_diag(p, s, n_electrons):
    x = p.copy()
    x[np.diag_indices(len(p))] = \
        np.diag(x) / np.trace(p) * n_electrons
    return x

def rescale_diag_batch(p_batch, s_batch, n_electrons):
    result = []
    for (p, s) in zip(p_batch, s_batch):
        result.append(rescale_diag(p, s, n_electrons))
    return np.array(result).astype("float64")
#---