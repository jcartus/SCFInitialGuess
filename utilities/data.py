"""In this module all components needed to assemble and process input data will
be stored

Authors:
 - Johannes Cartus, QCIEP, TU Graz"""

from os.path import exists, isdir, isfile, join, splitext, normpath, basename
from os import listdir
import numpy as np

import pyQChem as qc
from utilities.usermessages import Messenger as msg

class Molecule(object):
    """Class that contains all relevant data about a molecule"""

    def __init__(self, species, positions, full_name=""):
        
        if len(species) != len(positions):
            raise ValueError("")
        
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

class QChemJob(object):

    def __init__(self, job_name, molecule):
        self._job_name = job_name
        self._molecule = molecule
        self._rem_array = None
        self._job_type = None

        raise NotImplementedError("QChemJob is an abstract class!")

    def run(self):
        job = qc.inputfile()
        job.add(self._rem_array)
        job.add(self._molecule.get_QChem_molecule())

        if exists(self._job_name + ".out"):
            raise IOError("Output file already exists for job: {0}.".format(
                    self._job_name
                ))
        else:
            msg.info("Starting " + self._job_type + ": " + self._job_name, 1)
            job.run(name=self._job_name)

    @property
    def job_name(self):        
        return self._job_name
    
    @property
    def molecule(self):        
        return self._molecule

class QChemSCFJob(QChemJob):
    """Abstract super class for all jobs that have scf cycles (sp, aimd, etc.)"""

    def __init__(self, 
        job_name,
        molecule,
        basis_set,
        method,
        scf_print=3, 
        scf_final_print=3,
        scf_convergence=8
        ):
        """QChemSCFJob constructor.

        Args:
            job_name (str): name of the job
            molecue (Molecule object): molecule job should be done on (geometry etc.)
            basis_set (str): name of the basis to be used
            method (str): evaluation mathod (e.g. B3LYP)
            scf_convergence (int): convergence criterium (diff of energies must be 
                lower 10^scf_convergence)
            scf_print (int): print level for every scf step
            scf_final_print (int): print level for final scf step
        """

        
        self._job_name = job_name
        self._molecule = molecule

        self._rem_array = qc.rem_array()
        
        self._rem_array.method(method)
        self._rem_array.basis(basis_set)
        self._rem_array.scf_convergence(str(scf_convergence))

        self._rem_array.scf_print(str(scf_print))
        self._rem_array.scf_final_print(str(scf_final_print))

class QChemMDRun(QChemSCFJob):
    """Warpper for pyQChem for ab initio md runs. 
    See https://www.q-chem.com/qchem-website/manual/qchem43_manual/sect-aimd.html
    """

    def __init__(self, 
        job_name, 
        molecule, 
        aimd_method="BOMD",
        aimd_temperature=300,
        time_step=100,
        aimd_steps=100,
        aimd_init_veloc="THERMAL",
        basis_set="6-311++G**",
        method="b3lyp",
        **kwargs
        ):
        """Constructor:

        Arguments:
         - aimd_method: name of the md run method ('BOMD' or 'CURVY').
         - aimd_temperature: temperature in K the md run is performed at
         - aimd_steps: number of md steps.
         - time_step: time step in hartree atomic units. 1 a.u. = 0.0242 fs.
         - aimd_init_veloc: how the volcities should be initialized 
           ('THERMAL', 'ZPE', 'QUASICLASSICAL')
        """

        super(QChemMDRun, self).__init__(
            job_name=job_name,
            molecule=molecule,
            basis_set=basis_set,
            method=method,
            **kwargs
        )

        self._job_type = "AIMD"

        self._rem_array.jobtype("aimd")

        self._rem_array.aimd_method(aimd_method)
        self._rem_array.aimd_steps(str(aimd_steps))
        self._rem_array.aimd_temperature(str(aimd_temperature))
        self._rem_array.aimd_time_step(str(time_step))
        self._rem_array.aimd_initial_velocities(aimd_init_veloc)

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

