"""This module contains everything related to descriptors (i.e. mapping from
the environment to network inpurs)

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np

from SCFInitialGuess.utilities.constants \
    import number_of_basis_functions as N_BASIS

class AbstractDescriptor(object):
    """An abstract descriptor class (however 'desriptor' for output is already
    implemented)
    """

    @staticmethod
    def index_range(atoms, atom_index):
        """Calculate the range of matrix elements for atom specified by index in
        atoms list."""

        # summ up the number of basis functions of previous atoms
        start = 0
        for i in range(atom_index):
            start += N_BASIS[atoms[i]]
        
        end = start + N_BASIS[atoms[atom_index]]

        return start, end

    @classmethod
    def input_values(cls, S, atoms, index):
        raise NotImplementedError("AbstractDescriptor is an abstract class!")

    

    @classmethod
    def target_values(cls, result, index):

        start, end = cls.index_range(result.atoms, index)

        return result.P[start:end, start:end]


class ContractionDescriptor(AbstractDescriptor):
    """Descriptor that uses a contractor over S matrix"""

    @classmethod
    def inner_contraction(cls, S_section, species):
        """This is the inner contraction is a contraction over all 
        basis function contributions of an atom.
        """
        raise NotImplementedError("This is an Abstract class!")

    @classmethod
    def outer_contraction(cls, intermediate_result, next_operand):
        """This function connect the results of the inner contractions (atom 
        specific contraction) to be come the final input vector (i.e. a 
        contraction over atoms in the molecule). E.g. it could be a sum summing up 
        all the inner contraction results (Default is a sum).
        """
        return intermediate_result + next_operand

    @classmethod
    def input_values(cls, S, atoms, index):

        # start/end index of range of elements in e.g. S-Matrix
        # that correspond to current atom. Using the range object would 
        # trigger advanced indexing ...
        start, end = cls.index_range(atoms, index)
        

        x = np.zeros(N_BASIS[atoms[index]])

        # add contribution to descriptor from every other atom
        for i, atom in enumerate(atoms):
            
            # an atom should not influence itself
            if i != index:
                
                x = cls.outer_contraction(
                    x, 
                    cls.inner_contraction(
                        S[start:end, range(*cls.index_range(atoms, i))],
                        atom
                    )
                )

        return x

class SumWithElectronegativities(ContractionDescriptor):

    @classmethod
    def inner_contraction(cls, S_section, species):
        from SCFInitialGuess.utilities.constants \
            import electronegativities as chi

        # electronegativity weighted summand
        return np.sum(S_section, 1) * chi[species] / N_BASIS[species]

