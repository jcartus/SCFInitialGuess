"""In this module, all kinds of helper classes and functions 
for the descriptor classes are stored.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np
import matplotlib.pyplot as plt

from SCFInitialGuess.utilities.constants import number_of_basis_functions as N_BASIS
from SCFInitialGuess.utilities.dataset import extract_triu
from SCFInitialGuess.descriptors.coordinate_descriptors \
    import periodic_gaussian


def carthesian_to_spherical_coordinates(x):
    """Returns the vector x in spherical coordinates."""
    r = np.sqrt(np.sum(x**2))
    phi = np.arctan2(x[1], x[0])
    theta = np.arccos(x[2] / r)
    
    return r, phi % (2*np.pi), theta % np.pi

def real_spherical_harmonics(phi, theta, l, m):
    """Real spherical harmonics, also known as tesseral spherical harmonics
    with condon shortley phase.

    Only for scalar phi and theta!!!
    """    
    from scipy.special import lpmn, factorial

    
    if m == 0:
        y = np.sqrt(
            (2 * l + 1) / (4 * np.pi)
        ) * lpmn(m, l, np.cos(theta))[0][-1][-1]
    elif m < 0:
        y = (-1)**m * np.sqrt(2) * np.sqrt(
            (2 * l + 1) / (4 * np.pi) * \
            factorial(l - np.abs(m)) / factorial(l + np.abs(m))
        ) * lpmn(np.abs(m), l, np.cos(theta))[0][-1][-1] * np.sin(np.abs(m) * phi)
    elif m > 0:
        y = (-1)**m * np.sqrt(2) * np.sqrt(
            (2 * l + 1) / (4 * np.pi) * \
            factorial(l - np.abs(m)) / factorial(l + np.abs(m))
        ) * lpmn(np.abs(m), l, np.cos(theta))[0][-1][-1] * np.cos(np.abs(m) * phi)
    

    return y
    

def plot_normal_model(model, t):
    """Plot a model of Gaussians for values t"""
    for r_s, eta in zip(model[0], model[1]):
        plt.plot(t, np.exp(-1 * eta*(t - r_s)**2))
        
def plot_periodic_model(model, t):
    """Plot a model of Periodic Gaussians for values t"""
    period = model[2]
    for r_s, eta in zip(model[0], model[1]):
        plt.plot(t, periodic_gaussian(t, r_s, eta, period))


def plot_radial_activation(r, descriptor, mol, label, **kwargs):
    """Calculates and plots radial activation for descriptor model 
    for a molecule mol."""
    values = descriptor.calculate_atom_descriptor(
        0,
        mol,
        descriptor.number_of_descriptors
    )
    
    n_radial = descriptor.radial_descriptor.number_of_descriptors
    radial_values = values[:n_radial]
    
    inverse_values = descriptor.radial_descriptor.calculate_inverse_descriptor(r, radial_values)
    
    plt.plot(
        r, 
        inverse_values / np.max(inverse_values),
        label=label,
        **kwargs
    )
    


class BlockExtractor(object):
    """Extracts all blocks of relevance for atoms of a given species
    from a matrix of the right dimensions (e.g. all self overlap blocks
    from the S matrix)
    """

    def __init__(self, target_species, basis):
        self.target_species = target_species
        self.basis = basis


    def index_range(self, atoms, atom_index):
        """Calculate the range of matrix elements for atom specified by index in
        atoms list."""

        # summ up the number of basis functions of previous atoms
        start = 0
        for i in range(atom_index):
            start += N_BASIS[self.basis][atoms[i]]
        
        end = start + N_BASIS[self.basis][atoms[atom_index]]

        return start, end

    def extract_blocks(self, Matrix, atoms_in_molecule):
        """Returns a list with all blocks of a given matrix
        that correspond to atoms of the target species.

        Example: S-Matrix, self-overlap blocks are highlighted.
         ____ __________
        | C  |  :  :    | 
        |____|__:__:____|   
        |    |_H|__     |
        |       |_H|____|
        |          | O  |
        |__________|____|

        Args:
            TODO
        """
    
        extracted_blocks = []

        indices_of_atoms_of_interest = [
            i for i in range(len(atoms_in_molecule)) \
                if atoms_in_molecule[i] == self.target_species
        ]

        for index in indices_of_atoms_of_interest:
            start, end = self.index_range(atoms_in_molecule, index)
            block = Matrix[start:end, start:end]
            extracted_blocks.append(extract_triu(block, len(block)))
        ################
        # TODO: Extract only upper triu!
        ################

        return extracted_blocks
        