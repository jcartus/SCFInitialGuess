"""This file contains descriptor classes that use the direction of one 
atom to another in the molecule to descripte the environment

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np
from scipy.special import sph_harm

def carthesian_to_spherical_coordinates(x):
    """Returns the vector x in spherical coordinates."""
    r = np.sqrt(np.sum(x**2))
    phi = np.arctan2(x, x[0])[0]
    theta = np.arccos(x[2] / r)
    
    return r, phi, theta

#-------------------------------------------------------------------------------
#   Quantity descriptros (coordinates to atomic contribution)
#-------------------------------------------------------------------------------

# a collection of guassian positioning models (cut_off, r_s, eta)
RADIAL_GAUSSIAN_MODELS = {
    "Origin-Centered_1": (
        5.0,
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.2, 0.5, 0.2, 0.1, 0.01]
    ),

    "Equidistant-Broadening_1": (
        5.0,
        [0.6, 1.1, 1.6, 2.1, 2.6],
        [0.9, 0.7, 0.6, 0.5, 0.4]
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


class BehlerCutoff1(AbstractCutoff):

    def __init__(self, threshold):

        self.threshold = threshold

    def apply(self, G, r, phi, theta):
        """Applies the cutoff to the symmetry vector G
        (Weights G according to r, phi , theta, the spherical coordinates
        of the distance vector from atom_i to atom_j)
        """
        return G * behler_cutoff_1(r, self.threshold)



class Damping(AbstractCutoff):
    """Applies a damping exp(-r/tau), where tau is the thershold"""

    def __init__(self, threshold):

        self.threshold = threshold

    def apply(self, G, r, phi, theta):
        """Applies the cutoff to the symmetry vector G
        (Weights G according to r, phi , theta, the spherical coordinates
        of the distance vector from atom_i to atom_j)
        """
        return G * np.exp(- r / self.threshold)

#-------------------------------------------------------------------------------
#   Low level descriptor classes
#-------------------------------------------------------------------------------


class AbstractQuantityDescriptor(object):

    def __init__(self, r_cutoff):
        
        self.R_c = r_cutoff
        self.number_of_descriptors = None

    def calculate_descriptor(self, x):
        """Returns a descriptor value for an abstract quantity x (e.g. a 
        distance or an angle). The vetor will be list!!!!
        """
        raise NotImplementedError(
            "AbstractQuantityDescriptor is an abstract class."
        )



class Gaussians(object):

    def __init__(self, r_s, eta):
        
        # check if list are of equal length of if eta is scalar
        # check if list are of equal length of if eta is scalar
        if isinstance(eta, (list, tuple)):
            if len(r_s) != len(eta) and len(eta) != 1 :
                raise ValueError("Dimension of r_s and eta do not match")

        self.r_s = np.array(r_s)
        self.eta = np.array(eta)

        self.number_of_descriptors = len(r_s)

    def calculate_descriptor(self, x):
        """Returns a descriptor value for an abstract quantity x (e.g. a 
        distance or an angle). The vector will be list!!!!
        """
        return list(
            np.exp(-1 * self.eta*(x - self.r_s)**2) 
        )        

class PeriodicGaussians(object):

    def __init__(self, r_s, eta, period):
        """Constructor
        Args:
            - r_s list<float>: centers of gaussians.
            - eta list<float> or float: width of gaussians.
            - period float: the period after which the 
                gaussians are repeated.
        """
        
        # check if list are of equal length of if eta is scalar
        if isinstance(eta, (list, tuple)):
            if len(r_s) != len(eta) and len(eta) != 1 :
                raise ValueError("Dimension of r_s and eta do not match")

        self.r_s = np.array(r_s)
        self.eta = np.array(eta)
        self.period = period
        
        self.number_of_descriptors = len(r_s)

    def calculate_descriptor(self, x):
        """Returns a descriptor value for an abstract quantity x (e.g. a 
        distance or an angle). The vector will be list!!!!
        """
        return [
            np.exp(-1 * eta * ((x % self.period) - r_s)**2) + \
            np.exp(-1 * eta * ((x % self.period) - self.period - r_s)**2) \
            for (r_s, eta) in zip(self.r_s, self.eta)
        ]
    
    def calculate_inverse_descriptor(self, t, y):
        """Returns the y- weighted sum of the gaussians,
        evaluated at values t.
        """
        return np.dot(y, np.array(self.calculate_descriptor(t)))


class SphericalHarmonics(object):
    """
    Real and Immaginary part of the spherical harmonics 
    """

    def __init__(self, l_max):
        
        
        self.l_max = l_max

        self.number_of_descriptors = (l_max + 1)**2 * 2

    def calculate_descriptor(self, r, phi, theta):
        """Returns a vector with the spherical harmonics of 
        the direction specified by phi an theta. The result is a list
        of first the real and then the imaginary part.
        """


        real, imaginary = [], []
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                sph = sph_harm(m, l, theta, phi)
                real.append(sph.real)
                imaginary.append(sph.imag)
        
        return np.array(real + imaginary)


#-------------------------------------------------------------------------------
#   Top level descriptor classes
#-------------------------------------------------------------------------------


class AbstractCoordinateDescriptor(object):
    """This class takes a SCFInitialGuess.utilities.dataset.Molecule
    and calculates a descriptor vector that captures the atomic environment.
    """

    def __init__(self, 
            radial_descriptor, 
            azimuthal_descriptor,
            polar_descriptor, 
            cut_off
        ):      

        self.radial_descriptor = radial_descriptor
        self.azimuthal_descriptor = azimuthal_descriptor
        self.polar_descriptor = polar_descriptor
        self.cut_off = cut_off

    @property
    def number_of_descriptors(self):
        return self.radial_descriptor.number_of_descriptors + \
            self.azimuthal_descriptor.number_of_descriptors + \
            self.polar_descriptor.number_of_descriptors

    def calculate_descriptors_batch(self, molecules):
        """Calculate all descriptors for a batch of molecules"""
        
        descriptors = []
        for molecule in molecules:
            descriptors.append(self.calculate_all_descriptors(molecule))
        
        return np.array(descriptors).reshape(len(molecules), -1)

    def calculate_all_descriptors(self, molecule):
        """Calculates all descriptors for all atoms in the molecule"""
        
        geometry = molecule.geometry

        molecular_descriptor = []

        # calculate descriptor for each atom
        for i in range(molecule.number_of_atoms):
            
            molecular_descriptor.append(
                self.calculate_atom_descriptor(
                    i, 
                    molecule, 
                    self.number_of_descriptors
                )
            )
        return np.array(molecular_descriptor)

    def calculate_atom_descriptor(self, index_atom, molecule, number_of_descriptors):
        """Calculates the descriptor of the atom with index index_atom in the 
        molecule
        """

        geometry = molecule.geometry

        atomic_descriptor = np.zeros(number_of_descriptors)
            
        for j, geom_j in enumerate(geometry):
            if index_atom == j:
                continue
            
            atomic_descriptor += self.calculate_atomic_descriptor_contribution(
                    geometry[index_atom], 
                    geom_j
                )

        return atomic_descriptor




    def calculate_atomic_descriptor_contribution(self, geom_i, geom_j):
        """Calculates the contribution to the atomic descriptor of atom i
        for the atom j.
        """

        raise NotImplementedError(
            "AbstractCoordinateDescriptor is an abstract class!"
        )

    def apply_coordinate_descriptors(self, R):
        """This function is called by calculate_atomic_descriptor_contribution
        and avalues the repective coordinate descriptors of a vector R
        (probably the path from one atom to another in the molecule).
        """
        
        r, phi, theta = carthesian_to_spherical_coordinates(R)

        G = []

        G += self.radial_descriptor.calculate_descriptor(r)
        G += self.azimuthal_descriptor.calculate_descriptor(phi)
        G += self.polar_descriptor.calculate_descriptor(theta)

        return self.cut_off.apply(np.array(G), r, phi, theta)


class NonWeighted(AbstractCoordinateDescriptor):
    """This class calculates a desriptor of the vectors of the atoms
    to each other with gaussians in the radial direction.
    """

    def calculate_atomic_descriptor_contribution(self, geom_i, geom_j):
        """Calculates the contribution to the atomic descriptor of atom i
        for the atom j.
        """

        R = np.array(geom_i[1]) -  np.array(geom_j[1])

        return self.apply_coordinate_descriptors(R)


class AtomicNumberWeighted(AbstractCoordinateDescriptor):
    """This class calculates a desriptor of the vectors of the atoms
    to each other with gaussians in the radial direction.
    """

    def calculate_atomic_descriptor_contribution(self, geom_i, geom_j):
        """Calculates the contribution to the atomic descriptor of atom i
        for the atom j.
        """
        from SCFInitialGuess.utilities.constants import atomic_numbers as Z

        R = np.array(geom_i[1]) -  np.array(geom_j[1])

        return self.apply_coordinate_descriptors(R) * Z[geom_j[0]]

