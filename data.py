"""In this module all components needed to assemble and process input data will
be stored

Authors:
 - Johannes Cartus, QCIEP, TU Graz"""

from os.path import exists, isdir, isfile, join, splitext, normpath, basename
from os import listdir
import multiprocessing as mp
from shutil import move
import argparse
import numpy as np
from warnings import warn

import pyQChem as qc


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

        raise NotImplementedError("QChemJob is an abstract class!")

    def run(self):
        job = qc.inputfile()
        job.add(self._rem_array)
        job.add(self._molecule.get_QChem_molecule())

        if exists(self._job_name + ".out"):
            raise IOError("Output file already exists for job: {0}".format(
                    self._job_name
                ))
        else:
            job.run(name=self._job_name)

    @property
    def job_name(self):        
        return self._job_name
    
    @property
    def molecule(self):        
        return self._molecule

class QChemMDRun(QChemJob):
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
        method="b3lyp"
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

        self._job_name = job_name
        self._molecule = molecule

        self._rem_array = qc.rem_array()
        self._rem_array.jobtype("aimd")
        self._rem_array.method(method)
        self._rem_array.basis(basis_set)

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
                "Could not read database. There is no folder {0}".format(folder)
            )

        files = [x for x in listdir(folder) if isfile(join(folder, x))]

        molecules = []

        for file_name in files:
            try:
                molecules.append(
                    cls.read_molecule_from_file(join(folder, file_name))
                )
            except Exception as ex:
                warn(
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
                species.append(sep[0])
                positions.append(list(map(float, sep[1:])))

            return Molecule(species, positions, full_name=name)

def produce_randomized_geometries(molecules, amplification):
    """Will create a list of geometries similar to the ones given in molecules
    but with some random noise added. for each given geometry a 
    amplification times as much random ones are created. They will have the same
    name with a trailing underscore and index"""

    from scipy.spatial.distance import pdist
    
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

def main():

    parser = argparse.ArgumentParser(
        prog='PROG',
        description="This program will read molecule geometries from a data" + 
            "base folder, generate a few geometries and do md runs with qChem on them" 
        
    )

    parser.add_argument(
        "-s", "--source", 
        required=False,
        help="The (path to the) data base folder, which contains the original molecules",
        metavar="source directory",
        dest="source",
        default=normpath("data_base/s22/")
    )

    parser.add_argument(
        "-d", "--destination", 
        required=False,
        help="The (path to the) results folder, where the calculationresults can be stored in",
        metavar="destination directory",
        dest="destination",
        default=normpath("result/")
    )

    parser.add_argument(
        "-f", "--multiplication-factor", 
        default=1,
        required=False,
        help="The number of randomized geometries generated for every molecule in the data base",
        type=int,
        dest="amplification"
    )

    args = parser.parse_args()

    # todo args richtig umsetzen
    molecules = PyQChemDBReader.read_database(args.source)
    random_molecules = produce_randomized_geometries(molecules, args.amplification)



    # todo get num of trherads dynamically. evtl as argument?
    pool = mp.Pool(processes=4)
    for mol in random_molecules:
        pool.apply_async(qchem_execution_section(mol, args))
    pool.close()
    pool.join()




# define paralell section    
def qchem_execution_section(mol, args):
    run = QChemMDRun(mol.full_name, mol)

    # add path for result as opton to qChemmdrun!!
    run.run()
    for ext in ["in", "out", "sh"]:
        try:
            fname = run.job_name + "." + ext
            move(fname, join(args.destination,fname))
        except Exception as ex:
            warn("Could not move {0}: ".format(fname) + str(ex))    

if __name__ == '__main__':
    main()
