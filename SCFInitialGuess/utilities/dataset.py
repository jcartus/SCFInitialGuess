"""In this module all components needed to assemble and process input data will
be stored

Authors:
 - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import exists, isdir, isfile, join, splitext, normpath, basename
from os import listdir, walk

import sys
import numpy as np
import re

from utilities.constants import number_of_basis_functions as N_BASIS # todo: this is inconsitent. constantprovider class should be used.

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
    def geometry(self):
        """The geometries as used by A. Fuchs in his NN Project """
        for x in zip(self.species, self.positions):
            yield x

    def get_sum_formula(self):
        raise NotImplementedError("Sum formula not available yet!")

    def get_QChem_molecule(self):
        """Get a pyqchem molecule object representation of the molecule"""
        
        if sys.version_info[0] >= 3:
            raise ImportError("PyQChem cannot be used with python 3 or higher!")
        
        import pyQChem as qc

        xyz = qc.cartesian()

        for (s,p) in zip(self.species, self.positions):
            xyz.add_atom(s, *map(str,p))

        return qc.mol_array(xyz)

    def get_pyscf_molecule(self):
        """Get a pyscf mole representation of the  molecule"""
        from pyscf.gto import Mole

        mol = Mole()
        mol.atom = self.geometry
        mol.basis = "6-311++g**"
        mol.build()

        return mol


class XYZFileReader(object):
    """This will read all the molecules from the database files (which were 
    downloaded from the pyqChem repository) that are in a specified folder"""

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

        self._S = None
        self._H = None
        self._P = None
        self._F = None

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

    @staticmethod
    def _read_matrix_from_print_file(file_path):
        """reads matrix from scf matrix print file and returns it as numpy array """        
        with open(file_path, 'r') as f:
            lines = f.readlines()[2:]

        return np.array(list(map(lambda x: list(map(float, x.split())), lines)))

    def _discover_atoms(self):
        """Check out file to see which atoms there were in the molecule"""

        with open(join(self._root_dir, self._job_name + ".out")) as f:
            
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

        return start, end
        
    def create_batch(self, atom_type):
        """This will check for all atoms of the given type in the molecule and 
        create a set of inputs (x) and expected outputs (y) for each instance.
        
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

        from utilities.constants import electronegativities as chi

        try:
            atom_instance_indices = \
                [i for i, atom in enumerate(self.atoms) if atom == atom_type]

            if len(atom_instance_indices) == 0:
                raise ValueError("No atoms of this type found")
        except ValueError as ex:
            msg.info(
                "No atoms of type " + atom_type + " found in " + self._job_name
            )

            # if nothing found just return an empty list
            return ([], [])

        # dataset that will be returned
        x_list = []
        y_list = []

        for ind in atom_instance_indices:
            
            # start/end index of range of elements in e.g. S-Matrix
            # that correspond to current atom. Using the range object would 
            # trigger advanced indexing ...
            start, end = self._index_range(ind)
            

            #--- get descriptros (network inputs) ---
            x = np.zeros(N_BASIS[atom_type])

            # add contribution to descriptor from every other atom in the 
            # molecule (weighted by electronegativity)
            for i, atom in enumerate(self.atoms):
                
                # an atom should not influence itself
                if i != ind:

                    # add weighted summand
                    x += np.sum(
                        self.S[start:end, range(*self._index_range(i))], 
                        1
                    ) * chi[atom]

            x_list.append(x)
            #---

            #--- get target (network output)---
            y_list.append(self.P[start:end, start:end])
            #---
        
        return x_list, y_list

class Dataset(object):
    """This class will govern the whole dataset and has methods to process and 
    split it.
    """
    def __init__(self, x, y, split_test=0.1, split_validation=0.2):
        """Ctor

        Args:
            - x <np.array>: input data that result in target values/labels
            - y <np.array>: the target values/labels
            - split_test <float>: the fraction how much of the dataset shall
                be used for testing
            - split_validation <float>: the fraction how much of the dataset -test
                set shall be used for validation.
        """

        # normalize the dataset
        x, self.x_mean, self.x_std = self.normalize(x)

        # shuffle dataset
        dataset = self.shuffle_batch(x, y)

        # extract test data
        dataset, self.testing = self.split_dataset(
            dataset[0], 
            dataset[1], 
            split_test
        )


        # split rest in test and validation
        self.training, self.validation = self.split_dataset(
            dataset[0], 
            dataset[1], 
            split_validation
        )

    def sample_minibatch(self, size):
        """returns a sub set of the training data 
        
        Args:
            size <int/float>: if size is int it will be assumed as the number
            of points required, if float it will be assumed as the
            fraction of the training data to be used.
        """

        if isinstance(size, int):
            return self.random_subset(*self.training, size=size)
        elif isinstance(size, float):
            return self.random_subset_by_fraction(*self.training, fraction=size)
        else:
            raise TypeError(
                "Size parameter must be either int or float, but was " + \
                str(type(size)) + "."
            )
        
    @staticmethod
    def shuffle_batch(x, y):
        """randomly shuffles the elements of x, y, so the the elements still
        correspond to each other"""
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        return x[indices], y[indices]

    @staticmethod
    def random_subset(x, y, size=1):
        """Cut out size random values from the batch (x,y)"""
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        indices = indices[:int(size)] 
        return x[indices], y[indices]    

    @classmethod
    def random_subset_by_fraction(cls, x, y, fraction):
        """Get a subset of the batch (x,y) with a fraction of the values"""
        return cls.random_subset(x, y, int(np.ceil(fraction * len(x))))

    @staticmethod
    def split_dataset(x_raw, y_raw, fraction=0.2):
        """Splits a data set into two subsets (e.g. test and training data)
        
        Args:
            - fraction <float>: the fraction of the ds that should be split off
        """

        ind_train = int(np.floor(fraction * len(x_raw)))
        x_train, y_train = x_raw[ind_train:], y_raw[ind_train:]
        x_test, y_test = x_raw[:ind_train], y_raw[:ind_train]

        return (x_train, y_train), (x_test, y_test)
    
    @staticmethod
    def normalize(x):
        """Will trans form a dataset with elements x_ij, where j is the index
        that labels the example and i the index that labels to which input
        the value corresponds, in the following way:

            x_ij = x_ij - mean(x_ij, j) / var(x_ij, j)

        where mean(..., j) and var(..., j) denote operation w.r.t j (i fixed.)
        """

        mean = np.average(x, 0)
        std = np.std(x, 0)

        return (x - mean) / std, mean, std

    @staticmethod
    def denormalize(x, mean, std):
        """The inverse trans formation to normalize"""

        return x * std + mean

def assemble_batch(folder_list, species="C"):
    """Looks in all folders for results to create a large batch to test the 
    network on.

    Args:
        - folder_list <list<str>>: a list of full paths to data base folders 
        w/ molecule results in them as subfolders.
        - species <str>: atomic species for which data shall be assembled
        - protion_stest <float>: the fraction how much of data shall be reserved 
        for testing.

    Returns (as tuple):
        - the normalized batch inputs
        - the batch ouput
        - the mean of the unnormalized batch
        - the stanard deviation of the unnormalized batch
    """

    msg.info("Assembling batch for: " + species, 2)

    x, y = [], []
    for database in folder_list:
        
        msg.info("Fetching data from " + database, 1)

        tree = walk(database)
            
        # logg how many points were found
        points_found = 0

        # go through all molecules in this data base
        for directory, _, _ in list(tree)[1:]:
            try:
                result = Result(directory)
                data = result.create_batch(species)
                x += data[0] 
                y += list(map(np.diag, data[1])) #todo: maybe the cast to list is not necessary
                points_found += len(data[0])
            except Exception as ex:
                msg.warn("There was a problem: " + str(ex))
        

        msg.info("Found " + str(points_found) + " points.", 1)

        
    msg.info("Done assembling. Found " + str(len(x)) + " points.", 2)

    return np.array(x), np.array(y)




