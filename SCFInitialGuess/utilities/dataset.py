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
from pyscf.scf import hf

from .constants import number_of_basis_functions as N_BASIS
from .usermessages import Messenger as msg


class Molecule(object):
    """Class that contains all relevant data about a molecule"""

    def __init__(self, species, positions, full_name="", basis="6-311++g**"):
        
        if len(species) != len(positions):
            raise ValueError("There have to be as many positions as atoms!")
        
        self.species = species
        self.positions = positions
        self.full_name = full_name

        self.basis = basis

        self._dim_cache = None 
    
    @property
    def number_of_atoms(self):
        return len(self.species)

    @property
    def geometry(self):
        """The geometries as used by A. Fuchs in his NN Project """
        return [x for x in zip(self.species, self.positions)]

    @property
    def number_of_electrons(self):
        from SCFInitialGuess.utilities.constants import atomic_numbers as Z
        return np.sum([Z[atom] for atom in self.species])

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

    @property
    def dim(self):
        """Calculates the dimension of e.g. the overlap matrix"""
        try:
            if self._dim_cache is None:
                dim = 0
                for atom in self.species:
                    dim += N_BASIS[self.basis][atom] 
            else:
                dim = self._dim_cache

        except AttributeError as ex:
            # For compatibilites with older datasets.
            dim = 0
            for atom in self.species:
                dim += N_BASIS[self.basis][atom] 

        return dim

    def make_atom_mask(self, atom_index):
        """Creates a mask for the atom with the index atom_index in the molecule
        """

        #--- calculate the ranges ---
        current_dim = 0
        for i in range(atom_index + 1):
            # calculate block range
            index_start = current_dim
            current_dim += N_BASIS[self.basis][self.species[i]] 
            index_end = current_dim
        #---

        # calculate logical vector
        L = np.arange(self.dim)
        L = np.logical_and(index_start <= L, L < index_end)

        mask = np.logical_and.outer(L, L)
                
        
        return mask


    def make_masks_for_species(self, species):
        """Creates a list of masks for all atoms of species species in the 
        molecule.
        """
        masks = []
        current_dim = 0
        for atom in self.species:
            # calculate block range
            index_start = current_dim
            current_dim += N_BASIS[self.basis][atom] 
            index_end = current_dim

            if atom == species:

                # calculate logical vector
                L = np.arange(self.dim)
                L = np.logical_and(index_start <= L, L < index_end)

                masks.append(np.logical_and.outer(L, L))
                
        return masks



def do_scf_runs(molecules):
    """Do scf calculation for molecules in molecules and extract all relevant 
    matrices
    """

    S, P, F = [], [], []
    for i, molecule in enumerate(molecules):
        
        msg.info(str(i + 1) + "/" + str(len(molecules)))
        try:
            mol = molecule.get_pyscf_molecule()
            mf = hf.RHF(mol)
            mf.verbose = 1
            mf.run()
            
            if mf.iterations == mf.max_cycle:
                raise AssertionError("Sample could not be converged!")

            h = mf.get_hcore(mol)
            s = mf.get_ovlp()
            p = mf.make_rdm1()
            f = fock_from_density(p, s, h, mol)

            S.append(s)
            P.append(p)
            F.append(f)
        except Exception as ex:
            msg.warn("There was a problem: " + str(ex))

    return S, P, F

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

    @classmethod
    def fetch_from_folder(cls, folder, basis, ecp=None):
    
        files = [file for file in listdir(folder) if ".out" in file]
    
        files.sort()

        mols = []
        for i, file in enumerate(files):
            
            msg.info("Fetching: " + str(i + 1) + "/" + str(len(files)))

            molecules = QChemResultsReader.read_file(folder + file)

            for molecule_values in molecules:
                mol = Molecule(*molecule_values)
                mol.basis = basis
                try:
                    mol.ecp = ecp
                except:
                    pass

                mols.append(mol)
                
        return mols


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
    def read_folder(cls, folder, basis="sto-3g"):
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
                    cls.read_molecule_from_file(
                        join(folder, file_name), 
                        basis=basis
                    )
                )
            except Exception as ex:
                msg.error(
                    "Could not parse from file {0}: ".format(file_name) + str(ex),
                    RuntimeWarning
                )

        return molecules            

    @staticmethod                
    def read_molecule_from_file(file_name, name="", basis="sto-3g"):

        if not isfile(file_name):
            raise OSError(
                "File could not be read. " + \
                "It does not exist at {0}!".format(file_name)
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
                    if re.match("^[A-Z][a-z]?$", sep[0]):
                        species.append(sep[0])
                        positions.append(list(map(float, sep[1:4])))

            return Molecule(species, positions, full_name=name)



def fock_from_density(p, s, h, mol):
    try:
        f = hf.get_fock(
            None, 
            h1e=h, 
            s1e=s, 
            vhf=hf.get_veff(mol=mol.get_pyscf_molecule(), dm=p), 
            dm=p
        )

    except AttributeError:
        f = hf.get_fock(
            None, 
            h1e=h, 
            s1e=s, 
            vhf=hf.get_veff(mol, dm=p), 
            dm=p
        )

    return f

def fock_from_density_batch(p_batch, s_batch, h_batch, molecules):
    f = []
    for p, s, h, mol in zip(p_batch, s_batch, h_batch, molecules):
        f.append(
            fock_from_density(p, s, h, mol)
        )

    return np.array(f)

def density_from_fock(f, s, mol):
    
    mo_energy, mo_coeff = hf.eig(f, s)
    mo_occ = hf.get_occ(mf=hf.SCF(mol), mo_energy=mo_energy, mo_coeff=mo_coeff)
    
    return hf.make_rdm1(mo_coeff, mo_occ)

def density_from_fock_batch(f_batch, s_batch, molecules):
    p = []
    for (s, f, mol) in zip(s_batch, f_batch, molecules):
        p.append(density_from_fock(f, s, mol.get_pyscf_molecule()))
    return np.array(p)



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

def extract_triu_batch(A_batch, dim):
    return np.array(
        [extract_triu(A, dim) for A in A_batch]
    )

def extract_triu(A, dim):
    """Extracts the upper triangular part of the matrix.
    Input can be matrix, will be reshaped if it is not.
    """
    return A.reshape(dim, dim)[np.triu_indices(dim)]

def reconstruct_from_triu_batch(A_batch, dim):
    return np.array(
        [reconstruct_from_triu(A, dim) for A in A_batch]
    )

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




class AbstractDataset(object):
    """This class will govern the whole dataset and has methods to process and 
    split it.
    """
    def __init__(self):
        raise NotImplementedError("AbstractDataset is an abstract class!")


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


class StaticDataset(AbstractDataset):

    def __init__(self, train, validation, test, mu, std, mu_y=None, std_y=None):

        self.training = train
        self.validation = validation
        self.testing = test

        self.x_mean = mu
        self.x_std = std
        self.y_mean = mu_y
        self.y_std = std_y


class Dataset(AbstractDataset):
    """Should actally be called SimpleDataset or so (current name is historical)
    . You feed it all the 
    input-target pairs you have and it will randomize them and split them up for
     you.
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


class DescribedMoleculesDataset(AbstractDataset):
    """This class holds a dataset of molecules. With external descriptors
    objects, a input/target values for training can be created.
    """

    def __init__(self, 
        molecules, 
        input_descriptor, 
        output_descriptor,
        split_test=0.2, 
        split_validation=0.1,
        static_mode=True
    ):

        # input can be normalized later on!
        self.x_mean = None
        self.y_mean = None
        self.input_normalized = False

        self.input_descriptor = input_descriptor
        self.output_descriptor = output_descriptor

        # with the descriptors the input-target pairs are created.
        # after the first calculation the values are cached
        self._testing_pairs_cache = None
        self._training_pairs_cache = None
        self._validation_pairs_cache = None


        # shuffle dataset
        self.index = np.arange(len(molecules))
        if not static_mode:
            np.shuffle(self.index)
            molecules = molecules[self.index]


        #--- split in test and validation and traing ---

        # assume fraction
        if split_test < 1:
            split_test = int(split_test * len(molecules))

        if split_validation < 1:
            split_validation = int(split_validation * len(molecules))


        self._testing = molecules[-split_test:]
        self._validation = \
            molecules[-(split_test + split_validation):-split_validation]
        self._training = molecules[:-(split_test + split_validation)]
        #---

    def make_dataset_pairs(self, molecules):
        """For a given set of molecules fetch all input-target pairs"""

        x, y = [], []
        for mol in molecules:
            x += self.input_descriptor.calculate_descriptors(mol)
            y += self.output_descriptor.calculate_descriptors(mol)
        
        return np.array(x), np.array(y)

    @property
    def training(self):
        """The input-target pairs for all molecules in self._training.
        If a mu and std was set (e.g. by calling the make_normalization 
        function), the inputs will be normalized.
        """
        if self._training_pairs_cache is None:
            self._training_pairs_cache = \
                self.make_dataset_pairs(self._training)
        return self._training_pairs_cache

    @property
    def validation(self):
        """see method training"""
        if self._validation_pairs_cache is None:
            self._validation_pairs_cache = \
                self.make_dataset_pairs(self._validation)
        return self.make_dataset_pairs(self._validation)

    @property
    def testing(self):
        """see method testing"""
        if self._training_pairs_cache is None:
            self._training_pairs_cache = \
                self.make_dataset_pairs(self._training)
        return self.make_dataset_pairs(self._training)

    def normalize_pairs(self, x, y):
        if not self.x_mean is None:
            x = self.input_transformation(x) 
        return x, y

    def make_normalization(self):
        
        _, self.x_mean, self.x_std = \
            self.normalize(
                list(self.training[0]) + \
                list(self.validation[0]) + \
                list(self.testing[0])
            )
        
        self._training_pairs_cache = \
            (self.input_transformation(self.training[0]), self.training[1])
        self._validation_pairs_cache = \
            (self.input_transformation(self.validation[0]), self.validation[1])
        self._testing_pairs_cache = \
            (self.input_transformation(self.testing[0]), self.testing[1])

    def export(self, save_path, comment=None):
        """Export the dataset to a numpy binary"""
        data = [
            (self._training, self._validation, self._training),
            #(self.input_descriptor, self.output_descriptor),
            (self.x_mean, self.x_std),
            comment
        ]
        np.save(save_path, data)




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
        split_test=0.2, 
        split_validation=0.1
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




#-------------------------------------------------------------------------------
# Block Descriptors
#-------------------------------------------------------------------------------

def extract_center_block_dataset_pairs(descriptor, molecules, p_batch, species):  
    """Creates pairs of inputs and outputs for all all atoms of the element 
    species in the molecules to be used to set up a dataset for an NN 
    for a given descriptor and a target matrix p_batch. 
    """
    
    descriptor_values, blocks = [], []
    for p, mol in zip(p_batch, molecules):

        dim = mol.dim
        # make mask to extract central blocks
        masks = mol.make_masks_for_species(species)
    
        
        #--- calculate symmetry vectors ---
        for i, atom in enumerate(mol.species):
            if atom == species:
                descriptor_values.append(
                    descriptor.calculate_atom_descriptor(
                        i, 
                        mol,
                        descriptor.number_of_descriptors
                    )
                )
        #---

        #--- extract blocks from target matrices ---
        for mask in masks:
            blocks.append(extract_triu(
                np.asarray(p).reshape(dim, dim).copy()[mask], 
                N_BASIS[mol.basis][species]
            ))
        #---

    return descriptor_values, blocks

def extract_HOMO_block_dataset_pairs(descriptor, molecules, p_batch, species):  
    """Creates pairs of inputs and outputs for all all atoms of the element 
    species in the molecules to be used to set up a dataset for an NN 
    for a given descriptor and a target matrix p_batch. 
    The output are the (off-diagonal) homo-nuclear overlap blocks.
    """
    from SCFInitialGuess.construction.utilities import \
        make_atom_pair_mask  
    
    descriptor_values, blocks = [], []
    for p, mol in zip(p_batch, molecules):

        dim = mol.dim
        
        for i, atom_i in enumerate(mol.species):
            for j, atom_j in enumerate(mol.species):
                if i <= j:
                    continue
                    
                if atom_j == species and atom_i == species:
                    
                    #--- calculate symmetry vectors ---
                    descriptor_values.append(
                        list(
                            descriptor.calculate_atom_descriptor(
                                i, 
                                mol,
                                descriptor.number_of_descriptors
                            )
                        ) + list(
                            descriptor.calculate_atom_descriptor(
                                j, 
                                mol,
                                descriptor.number_of_descriptors
                            )
                        )
                    )
                    #---
                    
                    #--- extract blocks from target matrices ---
                    mask = make_atom_pair_mask(mol, i, j)
                    blocks.append(np.asarray(p).reshape(dim, dim).copy()[mask])
                    #---
                

    return descriptor_values, blocks


def extract_HETERO_block_dataset_pairs(descriptors, molecules, p_batch, species):  
    """Creates pairs of inputs and outputs for all all atoms of the element 
    species in the molecules to be used to set up a dataset for an NN 
    for a given descriptor and a target matrix p_batch. 
    The output are the (off-diagonal) hetero-nuclear overlap blocks.
    
    Args:
        descriptors <list> list of descriptors
        species <list<str>> list of species
    """
    from SCFInitialGuess.construction.utilities import \
        make_atom_pair_mask
    
    descriptor_values, blocks = [], []
    for p, mol in zip(p_batch, molecules):

        dim = mol.dim
    
        
        for i, atom_i in enumerate(mol.species):
            for j, atom_j in enumerate(mol.species):
                
                # only check lower triu
                if i <= j:
                    continue
                    
                if (atom_j in species and atom_i in species and atom_i != atom_j):
                    
                    
                    #--- calculate symmetry vectors ---
                    # make sure descriptor[0] describes atom [1], regardless 
                    # of storage order of atoms in molecule
                    # by using aliases ii and jj
                    if atom_i == species[1]:
                        ii = i
                        jj = j
                    else:
                        ii = j
                        jj = i
                        
                    
                    descriptor_values.append(
                        list(
                            descriptors[0].calculate_atom_descriptor(
                                ii, 
                                mol,
                                descriptors[0].number_of_descriptors
                            )
                        ) + list(
                            descriptors[1].calculate_atom_descriptor(
                                jj, 
                                mol,
                                descriptors[1].number_of_descriptors
                            )
                        )
                    )
                    #---
                    
                    #--- extract blocks from target matrices ---
                    mask = make_atom_pair_mask(mol, i, j)
                    blocks.append(np.asarray(p).reshape(dim, dim).copy()[mask])
                    #---
                
    return descriptor_values, blocks

def make_center_block_dataset(
    descriptor, 
    molecules, 
    T, 
    species, 
    normalize_output=False
):
    """Makes a dataset with blocks and symmetry vectors from all molecules in 
    molecules. 

    descriptor <SCFInitialGuess.descriptors.high_level.*>: 
        a high level descriptor object.
    molecules <list<list<SCFInitialGuess.utilities.dataset.Molecule>>>:
        List with 3 elements (training data, validation and test). 
        Each are a list of molecules. 
    T <list<np.array>> or <list<list<list>>>: 
        List with training, validation and test data. 
        each is a numpy array. 
    species <string>: the element name of the desired species.
    normalize_output <bool>: flag whether the output should be normalized.
    """

    inputs_test, outputs_test = extract_center_block_dataset_pairs(
        descriptor,
        molecules[2], 
        T[2],
        species
    )
    
    inputs_validation, outputs_validation = extract_center_block_dataset_pairs(
        descriptor,
        molecules[1], 
        T[1],
        species
    )

    inputs_train, outputs_train = extract_center_block_dataset_pairs(
        descriptor,
        molecules[0], 
        T[0],
        species
    )
    
    
    _, mu_x, std_x = StaticDataset.normalize(inputs_train + inputs_validation + inputs_test)
    _, mu_y, std_y = StaticDataset.normalize(outputs_train + outputs_validation + outputs_test)


    if normalize_output:
        dataset = StaticDataset(
            train=(
                StaticDataset.normalize(inputs_train, mean=mu_x, std=std_x)[0], 
                StaticDataset.normalize(outputs_train, mean=mu_y, std=std_y)[0]
            ),
            validation=(
                StaticDataset.normalize(inputs_validation, mean=mu_x, std=std_x)[0], 
                StaticDataset.normalize(outputs_validation, mean=mu_y, std=std_y)[0]
            ),
            test=(
                StaticDataset.normalize(inputs_test, mean=mu_x, std=std_x)[0], 
                StaticDataset.normalize(outputs_test, mean=mu_y, std=std_y)[0]
            ),
            mu=mu_x,
            std=std_x,
            mu_y=mu_y,
            std_y=std_y
        )

    else:
        dataset = StaticDataset(
            train=(
                StaticDataset.normalize(inputs_train, mean=mu_x, std=std_x)[0], 
                np.asarray(outputs_train)
            ),
            validation=(
                StaticDataset.normalize(inputs_validation, mean=mu_x, std=std_x)[0], 
                np.asarray(outputs_validation)
            ),
            test=(
                StaticDataset.normalize(inputs_test, mean=mu_x, std=std_x)[0], 
                np.asarray(outputs_test)
            ),
            mu=mu_x,
            std=std_x,
            mu_y=mu_y,
            std_y=std_y
        )
    
    return dataset


def make_block_dataset(
    descriptor, 
    molecules, 
    T, 
    species, 
    extractor_callback,
    normalize_output=False
):
    """Makes a dataset with blocks and symmetry vectors from all molecules in 
    molecules. 

    descriptor <SCFInitialGuess.descriptors.high_level.*>: 
        a high level descriptor object.
    molecules <list<list<SCFInitialGuess.utilities.dataset.Molecule>>>:
        List with 3 elements (training data, validation and test). 
        Each are a list of molecules. 
    T <list<np.array>> or <list<list<list>>>: 
        List with training, validation and test data. 
        each is a numpy array. 
    species <string>: the element name of the desired species.
    normalize_output <bool>: flag whether the output should be normalized.
    """

    inputs_test, outputs_test = extractor_callback(
        descriptor,
        molecules[2], 
        T[2],
        species
    )
    
    inputs_validation, outputs_validation = extractor_callback(
        descriptor,
        molecules[1], 
        T[1],
        species
    )

    inputs_train, outputs_train = extractor_callback(
        descriptor,
        molecules[0], 
        T[0],
        species
    )
    
    _, mu_x, std_x = StaticDataset.normalize(inputs_train + inputs_validation + inputs_test)
    _, mu_y, std_y = StaticDataset.normalize(outputs_train + outputs_validation + outputs_test)


    if normalize_output:
        dataset = StaticDataset(
            train=(
                StaticDataset.normalize(inputs_train, mean=mu_x, std=std_x)[0], 
                StaticDataset.normalize(outputs_train, mean=mu_y, std=std_y)[0]
            ),
            validation=(
                StaticDataset.normalize(inputs_validation, mean=mu_x, std=std_x)[0], 
                StaticDataset.normalize(outputs_validation, mean=mu_y, std=std_y)[0]
            ),
            test=(
                StaticDataset.normalize(inputs_test, mean=mu_x, std=std_x)[0], 
                StaticDataset.normalize(outputs_test, mean=mu_y, std=std_y)[0]
            ),
            mu=mu_x,
            std=std_x,
            mu_y=mu_y,
            std_y=std_y
        )

    else:
        dataset = StaticDataset(
            train=(
                StaticDataset.normalize(inputs_train, mean=mu_x, std=std_x)[0], 
                np.asarray(outputs_train)
            ),
            validation=(
                StaticDataset.normalize(inputs_validation, mean=mu_x, std=std_x)[0], 
                np.asarray(outputs_validation)
            ),
            test=(
                StaticDataset.normalize(inputs_test, mean=mu_x, std=std_x)[0], 
                np.asarray(outputs_test)
            ),
            mu=mu_x,
            std=std_x,
            mu_y=mu_y,
            std_y=std_y
        )
        

    return dataset

class Data(object):
    """This class is used to automatically fetch data (i.e. collection 
    of molecules, overlap and density matrices etc.)
    """

    def __init__(self):
        
        self._T = [
            [], # train
            [], # val
            []  # test
        ]

        self._S = [
            [], 
            [], 
            []
        ]
        self._molecules =[
            [], 
            [], 
            []
        ]
    
    @classmethod
    def fetch_data(cls, data_path, postfix, target="P"):
        """Fetches MD run results from a folder"""

        S = np.load(join(data_path, "S" + postfix + ".npy"))
        P = np.load(join(data_path, target + postfix + ".npy"))

        molecules = np.load(join(data_path, "molecules" + postfix + ".npy"))
        
        return S, P, molecules
    
    def _package_and_append(
            self, 
            S, 
            T, 
            molecules, 
            split_test, 
            split_validation
        ):
        """S ... overlap matrix,
            T... target matrix,
            molecules ... molecules list,
            TODO
        """
        
        ind_test = int(split_test * len(molecules))
        ind_val = int(split_validation * ind_test)
        
        self._S[0] += list(S[:ind_val])
        self._S[1] += list(S[ind_val:ind_test])
        self._S[2] += list(S[ind_test:])
        
        self._T[0] += list(T[:ind_val])
        self._T[1] += list(T[ind_val:ind_test])
        self._T[2] += list(T[ind_test:])
        
        self._molecules[0] += list(molecules[:ind_val])
        self._molecules[1] += list(molecules[ind_val:ind_test])
        self._molecules[2] += list(molecules[ind_test:])
        
    
    def include(
        self, 
        data_path, 
        postfix, 
        target="P",
        split_test=0.8, 
        split_validation=0.8
    ):
        """Fetches data and packages it in train, validation and test.
        Target matrix is specified by target.
        """
        
        self._package_and_append(
            *self.fetch_data(data_path, postfix), 
            split_test=split_test,
            split_validation=split_validation
        )
        
    @property
    def molecules(self):
        return self._molecules
    
    @property
    def S(self):
        return self._S
    
    @property
    def T(self):
        return self._T
    
    @property
    def t_test(self):
        return self._T[2]
    
    @property
    def t_val(self):
        return self._T[1]
    
    @property
    def t_train(self):
        return self._T[0]
    
    @property
    def s_test(self):
        return self._S[2]
    
    @property
    def s_val(self):
        return self._S[1]
    
    @property
    def s_train(self):
        return self._S[0]
    
    @property
    def numer_of_samples(self):
        return (
            len(self._molecules[0]),
            len(self._molecules[1]),
            len(self._molecules[2]),
        )
    @property
    def number_of_samples_total(self):
        return np.sum(self.number_of_samples)


class ScreenedData(Data):
    """Dataset, that is screened for broken molecules, by checking 
    to largest C-H distance"""


    def __init__(self, r_max):
        
        self.r_max = r_max
        
        super(ScreenedData, self).__init__()
    
    def calculate_max_CH_distance(self, mol):
        """Retunes the greatest distance between any H 
        and any C atom in the Molecule"""
        
        r = []
    
        for i, geom_i in enumerate(mol.geometry):
            for j, geom_j in enumerate(mol.geometry):

                # avoid duplicates
                if i < j:
                    continue

                # only count C-H distances
                if set([geom_i[0], geom_j[0]]) == set(["H", "C"]):
                    r.append(
                        np.sqrt(np.sum((np.array(geom_i[1]) - np.array(geom_j[1]))**2))
                    )
        return r
    
    def select(self, S, P, molecules):
        
        is_selected = np.array([
            np.max(self.calculate_max_CH_distance(mol)) < self.r_max \
                for mol in molecules
        ])
        
        return S[is_selected], P[is_selected], molecules[is_selected]
    
    def fetch_data(self, data_path, postfix, target="P"):
        """Fetches MD run results from a folder"""

        S = np.load(join(data_path, "S" + postfix + ".npy"))
        P = np.load(join(data_path, target + postfix + ".npy"))

        molecules = np.load(join(data_path, "molecules" + postfix + ".npy"))
        
        
        
        return self.select(S, P, molecules)
    