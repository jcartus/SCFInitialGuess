"""This module contains everything related to descriptors (i.e. mapping from
the environment to network inpurs)

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np
import tensorflow as tf

from SCFInitialGuess.utilities.constants \
    import number_of_basis_functions as N_BASIS

class AbstractDescriptor(object):
    """An abstract descriptor class (however 'desriptor' for output is already
    implemented)
    """

    @classmethod
    def process_batch(cls, environments):
        """This function returns an input vector for neural networks. 
        This input vector is calculated from the "x"-value of the dataset, here
        refered to as environment.

        Args:
            environments <list<??>>: a list of objects returned by the 
            dataset as "x"- values. (E.g. S-matrix blocks).

        Returns:
            An iterator containing the descriptor vector for each environment
            in environmens.
        """
        
        for environment in environments:
            yield cls.calculate_description(environment)

    @classmethod    
    def calculate_description(environment):
        """Does the actual description of an environment (i.e. calculates
        the descriptor vector for an environment)
        """
        raise NotImplementedError("Abstract Descriptor is an abstract class.")



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
        return NotImplementedError("ContractionDescriptor is an Abstract class")

    @classmethod
    def calculate_description(cls, environment):
        """Calculates the environment vector, by first reducing all
        Blocks with the inner contraction and then combining the block
        results with the outer contraction.

        Args:
            envrionment <list<list<??>>: Musst be a list of lists to allow
            two contractions.
        """
        return             cls.outer_contraction, 
            tf.map_fun(
                cls.inner_contraction, 
                environment
            )

class WeightedSum(ContractionDescriptor):

    @classmethod
    def inner_contraction(cls, S_section, species):

        # electronegativity weighted summand
        return np.sum(S_section, 1) / N_BASIS[species] * cls.weights(species)
    
    @classmethod
    def weights(cls, species):
        return 1

class SumWithElectronegativities(WeightedSum):

    @classmethod
    def weights(cls, species):
        from SCFInitialGuess.utilities.constants \
            import electronegativities as chi

        # electronegativity weighted summand
        return chi[species]

class SumWithAtomicNumber(WeightedSum):

    @classmethod
    def weights(cls, species):
        from SCFInitialGuess.utilities.constants \
            import atomic_numbers as Z

        return Z[species]

class SumWithValenceElectrons(WeightedSum):

    @classmethod
    def weights(cls, species):
        from SCFInitialGuess.utilities.constants \
            import valence_electrons as z
        return z[species]


class SumWithInputCombinations(WeightedSum):

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

        #add xi * x_j

        combinations = [ cls.combination(xi, xj) for xi in x for xj in x]

        x = np.array(list(x) + combinations)

        return x

    @classmethod
    def combination(cls, xi, xj):
        return xi * xj