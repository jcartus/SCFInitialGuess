"""In this module, all kinds of helper classes and functions 
for the descriptor classes are stored.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from SCFInitialGuess.utilities.constants import number_of_basis_functions as N_BASIS
from SCFInitialGuess.utilities.dataset import extract_triu

# TODO a diagonal extractor, so i can train to just guess the diagonal

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
        