"""This module contains all relevent components and provides the 
function that generates a guess for the density matrix of a molecule for which
the diagonal of the density matrix was estimated with a neural network

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import normpath, join

import numpy as np
import tensorflow as tf

from SCFInitialGuess.nn.networks import EluFixedValue
from pyscf.scf.hf import init_guess_by_minao

class Descriptor(object):
    """This class contains all method required to calculate descriptor values.

    """

    @staticmethod
    def index_range(atoms, atom_index):
        """Calculate the range of matrix elements for atom specified by index in
        atoms list."""

        from utilities.constants import number_of_basis_functions as N_BASIS

        # summ up the numer of basis functions of previous atoms
        start = 0
        for i in range(atom_index):
            start += N_BASIS[atoms[i]]
        
        end = start + N_BASIS[atoms[atom_index]]

        return start, end

    @classmethod
    def values(cls, S, atoms, ind):
        """Returns the descriptors values for the atomic environment of atom i 
        specified by the atoms in the molecule.

        Args:
            - S <np.array>: the density matrix of the molecule.
            - atoms <list<str>>: list of names of the atoms in the molecule.
            - ind <int>: index of current atom in the molecule
        """

        from utilities.constants import electronegativities as chi


        atom_type = atoms[ind]       
            
        # start/end index of range of elements in e.g. S-Matrix
        # that correspond to current atom. Using the range object would 
        # trigger advanced indexing ...
        start, end = cls.index_range(atoms, ind)
        

        #--- get descriptros (network inputs) ---
        x = np.zeros(end - start)

        # add contribution to descriptor from every other atom in the 
        # molecule (weighted by electronegativity)
        for i, atom in enumerate(atoms):
            
            # an atom should not influence itself
            if i != ind:

                # add weighted summand
                x += np.sum(
                    S[start:end, range(*cls.index_range(atoms, i))], 
                    1
                ) * chi[atom]

        return x

def nn_guess(mol, S, P0=None):
    """This method returns a guess for the density matrix of a molecule
    that can be used as input for scf calculations. Until now only the diagonal 
    elements are estimated by the network, while the rest is 

    Args:
        - atoms <list<str>>: a list of chemical symbols for the atoms in the 
        molecule
        - S <np.array>: the density matrix of the molecule
        - P0 <np.array>: a (cheap) initial guess in which the network guess 
        (which calculates only the diagonal) shall be placed.

    Returns:
        - <np.array>: a matrix containing the initial guess for the density 
    matrix
    """
    
    atoms = mol.species

    sess = tf.Session()

    #--- acquire model and setup graph ---
    model_database = normpath("./models")
    
    networks = {}

    for species in list(set(atoms)):
        model = np.load(join(model_database, species + ".npy"))
        nn = EluFixedValue(*model)
        nn.setup()
        networks.update(species=nn)
    #---

    # if no init guess is given used pyscf default guess
    if P0 is None:
        P0 = init_guess_by_minao(mol)

    
    dm = []
    for atom_index, atom in enumerate(atoms):
        inputs = Descriptor.values(S, atoms, atom_index)
        dm.append(list(networks[atom].run(sess, inputs)))

    # overwrite the diag of P0

    return P0
