"""In this module all components needed to assemble and process input data will
be stored

Authors:
 - Johannes Cartus, QCIEP, TU Graz"""

from os.path import exists, isdir, isfile, join, splitext, normpath, basename
from os import listdir
import numpy as np
import re

import pyQChem as qc

from utilities.constants import number_of_basis_functions as N_BASIS

from utilities.usermessages import Messenger as msg


class Molecule(object):
    """Class that contains all relevant data about a molecule"""

    def __init__(self, species, positions, full_name=""):
        
        if len(species) != len(positions):
            raise ValueError("There have to be as many positions as atoms!")
        
        self.species = species
        self.positions = positions
        self.full_name = full_name


    @property
    def geometries(self):
        """The geometries as used by A. Fuchs in his NN Project """
        for x in zip(self.species, self.positions):
            yield x

    def get_sum_formula(self):
        raise NotImplementedError("Sum formula not available yet!")

    def get_QChem_molecule(self):
        """Get a pyqchem molecule object representation of the molecule"""
        xyz = qc.cartesian()

        for (s,p) in zip(self.species, self.positions):
            xyz.add_atom(s, *map(str,p))

        return qc.mol_array(xyz)

class PyQChemDBReader(object):
    """This will read all the molecules from the database files that are
    in a specified folder"""

    @classmethod
    def read_database(cls, folder):
        
        if not isdir(folder):
            raise OSError(
                "Could not read database. There is no folder {0}.".format(folder)
            )

        
        msg.info("Reading database: " + folder, 1)

        files = [x for x in listdir(folder) if isfile(join(folder, x))]

        molecules = []

        for file_name in files:
            try:
                molecules.append(
                    cls.read_molecule_from_file(join(folder, file_name))
                )
            except Exception as ex:
                msg.error(
                    "Could not parse from file {0}: ".format(file_name) + str(ex),
                    RuntimeWarning
                )

        msg.info("Done reading database.", 1)

        return molecules            

    @staticmethod                
    def read_molecule_from_file(file_name, name=""):

        if not isfile(file_name):
            raise OSError(
                "File could not be read. It does not exist at {0}!".format(file_name)
            )

        msg.info("Reading file: " + file_name)

        # use file name if nothing specified
        if not name:
            name = splitext(basename(file_name))[0]

        with open(file_name, 'r') as f:

            # read file and omitt first lines
            lines = f.readlines()[2:]

            # read geometries
            species, positions = [], []
            for line in lines:
                sep = line.split()

                # if not an empty line
                if len(sep) > 0:
                    species.append(sep[0])
                    positions.append(list(map(float, sep[1:])))

            return Molecule(species, positions, full_name=name)

def produce_randomized_geometries(molecules, amplification):
    """Will create a list of geometries similar to the ones given in molecules
    but with some random noise added. for each given geometry a 
    amplification times as much random ones are created. They will have the same
    name with a trailing underscore and index"""

    from scipy.spatial.distance import pdist
    
    msg.info(
        "Generating randomized geometries " + \
            "({0} for every source geometry).".format(amplification),
        1
    )

    random_molecules = []

    for mol in molecules:
        
        positions = np.array(mol.positions)
        # Todo better calculation for noise amplitude
        max_noise = min(pdist(positions)) * 0.1

        for i in range(amplification):
            random_molecules.append(
                Molecule(
                    mol.species,
                    positions * (1 + np.random.rand(*positions.shape) * max_noise),
                    full_name=mol.full_name + "_" + str(i)
                )
            )

    return random_molecules

class Result(object):
    """The result of a single point calculation done on a molecule by qchem
    with scf_matrix print on

    Attributes:
        S <np.array<double>>: overlap matrix
        H <np.array<double>>: core hamiltonian
        F <np.array<double>>: fock matrix from last SCF step.
        P <np.array<double>>: density matrix from last SCF step
    """

    def __init__(self, root_dir, job_name=None ):
        """Constructor.

        Args: 
            - root_dir <str>: full path to the directory in which the results
            can be found.
            - job name <str>: the name of the job (which as given to the
            qChem job for calculation). If none given it will be assumed to be
            the name of the root directory.
        
        TODO:
            - maybe turn this into a context
        """
        
        if not isdir(root_dir):
            raise ValueError("The folder " + str(root_dir) + " does not exist!")

        self._root_dir = root_dir

        # if no job name is given it is assumed to be the name of the directory
        if job_name is None:
            self._job_name = basename(normpath(root_dir))
        
        # read atoms from outfile
        self.atoms = self._discover_atoms()

        self._H = None
        self._S = None
        self._F = None
        self._P = None

    @staticmethod
    def _read_matrix_from_print_file(file_path):
        """reads matrix from scf matrix print file and returns it as numpy array """        
        with open(file, 'r') as f:
            lines = f.readlines()[2:]

        return np.array(map(float, map(lambda x: x.split(), lines)))

    # TODO vlt einfach im Konstruktor laden instead of lazy calls..
    @property
    def S(self):
        if self._S is None:
            self._S = \
                self._read_matrix_from_print_file(join(self._root_dir, "S.dat"))
        return self._S
        

    @property
    def H(self):
        if self._H is None:
            self._H = \
                self._read_matrix_from_print_file(join(self._root_dir, "H.dat"))
        return self._H

    @property
    def P(self):
        if self._P is None:
            self._P = \
                self._read_matrix_from_print_file(join(self._root_dir, "P.dat"))
        return self._P

    @property
    def F(self):
        if self._F is None:
            self._F = \
                self._read_matrix_from_print_file(join(self._root_dir, "F.dat"))
        return self._F

    def _discover_atoms(self):
        """Check out file to see which atoms there were in the molecule"""

        with open(join(self._root_dir, self._job_name + "out")) as f:
            
            molecule = re.search(r"\$molecule.*\$end", f.read(), re.DOTALL)
            if molecule is None:
                raise ValueError("No molecule found in " + f.name)
            else:
                molecule = molecule.group(0)

                # cut out geometries
                geometries = molecule.splitlines()[2:-1]

                # from geometries take the species
                atoms = [line.split()[0] for line in geometries]
        
        return atoms


    def _index_range(self, atom_id):
        """Calculate the range of matrix elements for atom specified by index in
        atoms list."""

        # summ up the numer of basis functions of previous atoms
        start = 0
        for i in range(atom_id):
            start += N_BASIS[self.atoms[i]]
        
        end = start + N_BASIS[self.atoms[atom_id]]

        return range(start, end)
        
    def create_batch(self, atom_type):
        """This will check for all atoms of the given type in the molecule and 
        create a set of inputs and expected outputs for each
        
        Args:
            atom_type <str>: element symbol for atom type for which input/output
            data shall be put together.

        Returns:
            A list of tuples that contains the descriptors x_i and and elements
            of the fock matrix that correspond to the atom in question. One 
            list element per instance of atom_type atoms found in the molecule.

        Example:
            let atom_type be C. Then the function will check for all C atoms in
            the molecule and return descriptors & sections of the fock matrix
            for each instance found.
        """

        from utilities.constants import electronegativites as chi

        try:
            atom_indices = self.atoms.find(atom_type)
        except ValueError as ex:
            msg.info(
                "No atoms of type " + atom_type + " found in " + self._job_name
            )

            # if nothing found just return an empty list
            return []


        result = []

        for ind in atom_indices:
            
            index_range = self._index_range(ind)

            #--- get descriptros (network inputs) ---
            x = np.zeros(N_BASIS[atom_type])

            # add contribution to descriptor from every other atom in the 
            # molecule (weighted by electronegativity)
            for i, atom in enumerate(self.atoms):
                
                # an atom should not influence itself
                if i != ind:

                    # add weighted summand
                    x += np.sum(
                        self.S[index_range, self._index_range(i)], 
                        1
                    ) * chi[atom]
            #---

            F = self.F[index_range, index_range]
            
            result.append((x, F))
        
        return F



    


