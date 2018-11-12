"""This file contains cut-off classes and functions used to 
weigh atomic contributions to the descriptor values according to 
the distance of the described model.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np



#-------------------------------------------------------------------------------
#   CutOffs classes
#-------------------------------------------------------------------------------


def behler_cutoff_1(r, R_c): 
    """Cut-off f_c,1 aus 
    J. Behler, Constructing High-Dimensional Neural Network Potentials:
    A Tutorial Review, International Journal of Quantum Chemistry, 2015, 
    Issue 15, 1032-1050

    Args:
        r . . the distance to be cut-off.
        R_c the cut-off radius.
    """
    L = r > R_c
    
    out = 0.5 * (np.cos(np.pi * r / R_c) + 1)
    
    try:
        # works only of out is non scalar
        out[r > R_c] = 0
    except:
        if r > R_c:
            out = 0
        
    return out

class AbstractCutoff(object):

    def __init__(self, threshold):

        self.threshold = threshold

    def apply(self, G, r, phi, theta):
        raise NotImplementedError("AbstractCutoff is an abstract class!")


class ConstantCutoff(object):

    def apply(self, G, r, phi, theta):
        return G

class BehlerCutoff1(AbstractCutoff):

    def __init__(self, threshold):

        self.threshold = threshold

    def apply(self, G, r, phi=None, theta=None):
        """Applies the cutoff to the symmetry vector G
        (Weights G according to r, phi , theta, the spherical coordinates
        of the distance vector from atom_i to atom_j)
        """
        return G * behler_cutoff_1(r, self.threshold)



class Damping(AbstractCutoff):
    """Applies a damping exp(-r/tau), where tau is the threshold"""

    def __init__(self, threshold):

        self.threshold = threshold

    def apply(self, G, r, phi, theta):
        """Applies the cutoff to the symmetry vector G
        (Weights G according to r, phi , theta, the spherical coordinates
        of the distance vector from atom_i to atom_j)
        """
        return G * np.exp(- r / self.threshold)