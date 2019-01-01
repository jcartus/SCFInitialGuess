""" Descriptor schemes used to build a fock matrix.

Author:
    - Johannes Cartus, TU Graz
"""

import numpy as np


def gwh_scheme(d, s, k=1.75):
    """Generalized Wolfsberg-Helmholz scheme
    
    f_ij = S_ij * k * (d_i + d_j) / 2

    Args:
       - D: diagonal to be used
       - S: overlap matrix
       - k: empirical factor
    """

    dim = s.shape[0]

    K = np.ones(s.shape) * k - \
            np.diag(np.ones(dim)) * (k - 1)  
    
    
    f = K * np.add.outer(d, d) * s / 2

    return np.array(f)
    

def gwh_scheme_batch(D, S, k=1.75):

    F  = []
    for (d, s) in zip(D, S):
        F.append(gwh_scheme(d, s, k))
    return np.array(F)
    