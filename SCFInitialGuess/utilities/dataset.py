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

from .constants import number_of_basis_functions as N_BASIS
from .usermessages import Messenger as msg


class Molecule(object):
    """Class that contains all relevant data about a molecule"""

    def __init__(self, species, positions, full_name=""):
        
        if len(species) != len(positions):
            raise ValueError("There have to be as many positions as atoms!")
        
        self.species = species
        self.positions = positions
        self.full_name = full_name

        self.basis = "6-311++g**"

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
        mol.basis = self.basis
        mol.build()

        return mol

class QChemResultsReader(object):

    def __init__(self):
        
        self.geometries = []

    @staticmethod
    def read_file(file_name):

        if not isfile(file_name):
            raise OSError(
                "File could not be read. It does not exist at {0}!".format(
                    file_name
                )
            )

        with open(file_name, "r") as f:
            content = f.read()
            
        start_mark = r"I     Atom"
        end_mark = r"Nuclear Repulsion Energy"

        match_start = re.finditer(start_mark, content, re.DOTALL)
        match_end = re.finditer(end_mark, content, re.DOTALL)
        
        start_pos = [m.start() for m in match_start]
        end_pos = [m.start() for m in match_end]

        
        if len(start_pos) == 0 or len(end_pos) == 0:
            raise ValueError("No molecule found in " + f.name)

        if len(start_pos) != len(end_pos):
            raise ValueError("Matches are mixed up. Uneven number of start/end tokes.")

        for (s,p) in zip(start_pos, end_pos):
            
            molecule = content[s:p]

            # cut out geometries
            geometries = molecule.splitlines()[2:-2]

            # from geometries take the species and positions
            species, positions = [], []
            for line in geometries:
                splits = line.split()
                species.append(splits[1])
                positions.append(list(map(float, splits[2:])))

            yield species, positions

    @classmethod
    def read_folder(cls, folder):
        
        files = [file for file in listdir(folder) if ".out" in file]
        
        files.sort()
        results = []
        for i, file in enumerate(files):
            
            msg.info("Fetching: " + str(i + 1) + "/" + str(len(files)))

            results.append(cls.read_file(join(folder, file)))
        return results

class XYZFileReader(object):
    """This will read all the molecules from the database files (which were 
    downloaded from the pyqChem repository) that are in a specified folder"""
    
    @classmethod
    def read_tree(cls, database_root):
        """This method locates all xyz in the root folder and all its
        sub folders and creates a molecule for each of them.
        """

        if not isdir(database_root):
            raise OSError("Could not find database root. There is no " + \
                " folder {0}.".format(database_root))

        # read files in root dir
        molecules = cls.read_folder(database_root)

        # walk down the tree
        for root, dirs, _ in walk(database_root):
            for directory in dirs:
                molecules += cls.read_folder(join(root, directory))

        return molecules

    @classmethod
    def read_folder(cls, folder):
        """This method locates all xyz files in a folder and creates a molecule
        for each of it (returned as a list). All other files or subfolders are
        ignored.
        """
        
        if not isdir(folder):
            raise OSError(
                "Could not read database. There is no folder {0}.".format(folder)
            )

        files = sorted(list(filter(
            lambda x: isfile(join(folder, x)) and x.endswith(".xyz"),
            listdir(folder)
        )))
        
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

        return molecules            

    @staticmethod                
    def read_molecule_from_file(file_name, name=""):

        if not isfile(file_name):
            raise OSError(
                "File could not be read. It does not exist at {0}!".format(file_name)
            )

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
        atoms list<str>: the species available in the molecule of the result.
    """

    def __init__(self, root_dir, job_name=None ):
        """Constructor.

        Args: 
            - root_dir <str>: full path to the directory in which the results
            can be found.
            - job name <str>: the name of the job (which as given to the
            qChem job for calculation). If none given it will be assumed to be
            the name of the root directory.
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

    def _index_range(self, atom_index):
        """Calculate the range of matrix elements for atom specified by index in
        atoms list."""

        # summ up the numer of basis functions of previous atoms
        start = 0
        for i in range(atom_index):
            start += N_BASIS[self.atoms[i]]
        
        end = start + N_BASIS[self.atoms[atom_index]]

        return start, end
        
    def create_batch(self, atom_type, extractor):
        """This will check for all atoms of the given type in the molecule and 
        create a set of inputs (x) and expected outputs (y) for each instance.
        The values x and y are extracted by an extractor object from the 
        Result. 
        [Previously that instead of an extractor the descriptor 
        was directly applied at this stage.]
        
        Args:
            atom_type <str>: element symbol for atom type for which input/output
            data shall be put together.

        Returns:
            A tuples of two list that contain the extractors x_i and and elements
            of the fock matrix that correspond to the atom in question. One 
            list element per instance of atom_type atoms found in the molecule.
        """

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
            x_list.append(extractor.input_values(self.S, self.atoms, ind))
            y_list.append(extractor.target_values(self, ind))
           
        return x_list, y_list


def extract_triu(A, dim):
    """Extracts the upper triangular part of the matrix.
    Input can be matrix, will be reshaped if it is not.
    """
    return A.reshape(dim, dim)[np.triu_indices(dim)]

def reconstruct_from_triu(A_flat, dim):
    """Reconstructus the full symmetric matrix (dim x dim, not
    flattened out) from the flattend elements of the upper 
    triag of a symmetric matrix!"""
    result = np.zeros((dim, dim))
    result[np.triu_indices(dim)] = A_flat
    return result + result.T - np.diag(np.diag(result))

def make_matrix_batch(vector_batch, dim, is_triu=False):
        """Turns a batch of flatted out matrices into a batch of actual matrices
        i.e. reshapes the vectors into dim x dim matrices again.
        TODO describe inputs
        """

        if is_triu:
            vector_batch = np.array(list(map(
                lambda x: reconstruct_from_triu(x, dim), 
                vector_batch
            )))

        return vector_batch.reshape([-1, dim, dim])

def make_butadien_dataset(molecules, S, P, test_samples=50, index=None):
    """This function creates a dataset where S, P Matrix and 
    the molecules match.

    Args:
        S <np.array<float>>: the overlap matrix,
        P <np.array<float>>: the density matrix,
        molecules list<Molecule>: the molecules in the dataset, in the
        same order as S and P.
        #TODO rest of the inputs

    Returns:
        The dataset, and the molecules (split in train and test)
    """

    ind_cut = len(S) - test_samples

    if index is None:
        index = np.arange(len(S))

    S_test = np.array(S)[index[ind_cut:]]
    P_test = np.array(P)[index[ind_cut:]]
    molecules_test = [molecules[index[i]] for i in range(ind_cut, len(S))]

    S_train = np.array(S)[index[:ind_cut]]
    P_train = np.array(P)[index[:ind_cut]]
    molecules_train = [molecules[index[i]] for i in range(ind_cut)]

    dataset = Dataset(np.array(S_train), np.array(P_train), split_test=0.0, split_validation=0.1)

    dataset.testing = (Dataset.normalize(S_test, mean=dataset.x_mean, std=dataset.x_std)[0], P_test)

    return dataset, (molecules_train, molecules_test)


class AbstractExtractor(object):
    """An abstract descriptor class (previously used as 'desriptor').
    Used to extract parts of S or P matrix to put into a dataset object for
    training.
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
        raise NotImplementedError("AbstractExtractor is an abstract class!")

    @classmethod
    def target_values(cls, result, index):
        raise NotImplementedError("AbstractExtractor is an abstract class!")

class BlockExtractor(object):
    """An abstract extractor class (however 'desriptor' previously used for
    descriptors). Extracts some parts of S and P Matrix to use for network
    training.
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
        """Returns a list with for each atom in the molecule of the result 
        a list of overlap blocks.

         ____ __________
        | C  |  :  :    | <-- For the first C, the three blocks right from the
        |____|__:__:____|     self overlap area (i.e. the C-Block) are retrned
        |    |_H|__     |
        |       |_H|____|
        |          | O  |
        |__________|____|

        Args:
            - S np.array (2D): with a quadratic Matrix of the shape as, say the
            overlap matrix.
            - atoms list<str>: a list of atoms in the molecule
            - index int: index of the molecule for which the descriptor schall 
            be calculated.
        """
        
        # find the range of the block-band for the atom in question
        start_rows, end_rows = cls.index_range(atoms, index)

        atom_blocks = []

        for i, atom in enumerate(atoms):
            
            #ignore the self-overlap part
            if i == index:
                continue
            
            start_cols, end_cols = cls.index_range(atoms, i)

            # find block 
            atom_blocks.append((
                atoms[i], 
                S[start_rows:end_rows, start_cols:start_cols]
            ))

        return atom_blocks


    

    @classmethod
    def target_values(cls, result, index):
        """Extract a part of the P or the F Matrix"""
        start, end = cls.index_range(result.atoms, index)

        return result.P[start:end, start:end]

class Dataset(object):
    """This class will govern the whole dataset and has methods to process and 
    split it.
    """
    def __init__(self, 
        x, 
        y, 
        split_test=0.1, 
        split_validation=0.2,
        normalize_input=True,
        static_mode=False
        ):
        """Ctor

        Args:
            - x <np.array>: input data that result in target values/labels
            - y <np.array>: the target values/labels
            - split_test <float>: the fraction how much of the dataset shall
                be used for testing
            - split_validation <float>: the fraction how much of the dataset -test
                set shall be used for validation.
            - static_mode <bool>: the randomisation if turned off!
        """

        if not isinstance(x, np.ndarray):
            raise TypeError(
                "x-dataset is not a numpy array but: " + str(type(x))
            )
        
        if not isinstance(y, np.ndarray):
            raise TypeError(
                "y-dataset is not a numpy array but: " + str(type(y))
            )

        # normalize the dataset
        if normalize_input:
            x, self.x_mean, self.x_std = self.normalize(x)
            msg.info("Data set normalized. Mean value std: {0}".format(
                np.mean(self.x_std)
            ), 1)
        else:
            self.x_mean = None
            self.x_std = None

        

        # shuffle dataset
        if not static_mode:
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


    @classmethod
    def create_from_splits(cls, 
        testing, 
        validation, 
        training, 
        normalize_input=True
        ):

        x = training[0] + validation[0] + testing[0]
        y = training[1] + validation[1] + testing[1]
        
        split_test = len(testing) / len(x)
        split_validation = len(validation) / len(x)

        return cls(
            x, y, 
            split_test, split_validation, normalize_input, 
            static_mode=True
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
    
    def input_transformation(self, x, std_tolerance=1e-20):
        """Will normalize a set of given input vectors with the mean and std
        of the dataset.
        """

        return self.normalize(x, mean=self.x_mean, std=self.x_std)[0]

    def inverse_input_transform(self, x):

        return self.denormalize(x, self.x_mean, self.x_std)



    @staticmethod
    def normalize(x, std_tolerance=1e-20, mean=None, std=None):
        """Will trans form a dataset with elements x_ij, where j is the index
        that labels the example and i the index that labels to which input
        the value corresponds, in the following way:

            x_ij = x_ij - mean(x_ij, j) / var(x_ij, j)

        where mean(..., j) and var(..., j) denote operation w.r.t j (i fixed.)
        """

        if mean is None or std is None:
            mean = np.average(x, 0)
            std = np.std(x, 0)

        # handle dvision by zero if std == 0
        return (
            (x - mean) / np.where(np.abs(std) < std_tolerance, 1, std),
            mean,
            std
        )

    @staticmethod
    def denormalize(x, mean, std):
        """The inverse trans formation to normalize"""

        return x * std + mean

class SCFResultsDataset(object):
    """This class will serve as a gathering of input data needed to train and
    evaluate one or many neural networks with SCF calculation results.

    The difficulty here is to create batches for training of atomic nns, while
    still being able to provide S and P matrices of the whole molecule.

    Attribute:
        - available_species dict<str, int>: a dictionary storing the number of
            environments available for atom types that appear in the dataset.
    """

    def __init__(self, 
        results, 
        split_test=0.1, 
        split_validation=0.2
        ):



        #--- update avail able species ---
        self.available_species = {}
        for result in results:

            for atom in result.atoms:
                if atom in self.available_species:
                    self.available_species[atom] += 1
                else:
                    self.available_species[atom] = 1
        #---

        #--- randomize and split dataset in test & train & validation ---
        indices = np.arange(len(results))
        np.random.shuffle(indices)

        # TODO stimmen die so?
        index_test = self.process_fractions(split_test, len(results))
        index_validation = index_test + self.process_fractions(
            split_validation, 
            len(results) - index_test
        )

        self.testing = results[:index_test]
        self.validation = results[index_test:index_validation]
        self.training = results[index_validation:]

        if len(self.testing) + len(self.validation) + len(self.training) \
                != len(results):
            raise ValueError("Split values for testing, validation and " + \
                "training are inconsistent!")
        #---

    @staticmethod
    def process_fractions(fraction, total_count):
        """Converts a fraction to an int (number of elements)"""
        if isinstance(fraction, int):
            return fraction
        elif isinstance(fraction, float):
            return int(np.ceil(fraction * total_count))

    @staticmethod
    def assemble_batch(results, species, extractor):

        if extractor is None:
            extractor = BlockExtractor()
        
        x, y = [], []
        for result in results:
            
            # logg how many points were found
            points_found = 0

            # go through all molecules in this data base
            data = result.create_batch(species, extractor) 
            x += data[0] 
            y += list(map(np.diag, data[1])) #todo: maybe the cast to list is not necessary
            points_found += len(data[0])
            
        return x, y

    def create_trainable_dataset(self, species, descriptor):
        """Returns a dataset that can be used in training
        routines SCFInitialguess.nn.training. The values in this dataset
        are inputs for Descriptor function which in turn produce the network 
        inputs!!!
        """

        testing = self.assemble_batch(self.testing, species, descriptor)
        validation = self.assemble_batch(self.validation, species, descriptor)
        training = self.assemble_batch(self.training, species, descriptor)

        return Dataset.create_from_splits(
            testing, 
            validation, 
            training, 
            normalize_input=False,
        )
    
   

def assemble_batch(folder_list, species="C", descriptor=None):
    """Looks in all folders for results to create a large batch to test the 
    network on.

    Args:
        - folder_list <list<str>>: a list of full paths to data base folders 
        w/ molecule results in them as subfolders.
        - species <str>: atomic species for which data shall be assembled

    Returns (as tuple):
        - the normalized batch inputs
        - the batch output
    """

    msg.info("Assembling batch for: " + species, 2)

    if descriptor is None:
        from SCFInitialGuess.nn.descriptors \
            import SumWithElectronegativities as descriptor
        
    
    if not isinstance(folder_list, list):
        folder_list = [folder_list]

    x, y = [], []
    for database in folder_list:
        
        msg.info("Fetching data from " + database, 1)
            
        # logg how many points were found
        points_found = 0

        # go through all molecules in this data base
        for directory, _, files in list(walk(database)):
            try:
                # search for results if there is an .out file
                if ".out" in "".join(files):
                    result = Result(directory)
                    data = result.create_batch(species, descriptor)
                    x += data[0] 
                    y += list(map(np.diag, data[1])) #todo: maybe the cast to list is not necessary
                    points_found += len(data[0])
            except Exception as ex:
                msg.warn("There was a problem: " + str(ex))
        

        msg.info("Found " + str(points_found) + " points.", 1)

        
    msg.info("Done assembling. Found " + str(len(x)) + " points.", 2)

    return np.array(x), np.array(y)




