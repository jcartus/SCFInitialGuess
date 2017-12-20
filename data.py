"""Todo docstring"""

from os.path import exists

import pyQChem as qc


class Molecule(object):
    """Class that contains all relavant data about a molecule"""

    def __init__(self, species, positions):
        
        if len(species) != len(positions):
            raise ValueError("")
        
        self._species = species
        self._positions = positions

    @property
    def geometries(self):
        """The geometries as used by A. Fuchs in his NN Project """
        for x in zip(self._species, self._positions):
            yield x

    def get_QChem_molecule(self):
        """Get a pyqchem molecule object representation of the molecule"""
        xyz = qc.cartesian()

        for (s,p) in zip(self._species, self._positions):
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
        job.add(self._molecule)

        if exists(self._job_name + ".out"):
            raise FileExistsError(
                "Output file already exists for job: {0}".format(self._job_name)
                )
        else:
            try:
                job.run(name=self._job_name)
            except Exception as ex:
                raise ex            

class QChemMDRun(QChemJob):
    """Warpper for pyQChem for ab initio md runs. 
    See https://www.q-chem.com/qchem-website/manual/qchem43_manual/sect-aimd.html
    """

    def __init__(self, 
        job_name, 
        molecule, 
        aimd_method="BOMD",
        aimd_temperature=300,
        time_step=10,
        aimd_steps=1000,
        aimd_init_veloc="THERMAL",
        basis_set="6-311++G**",
        method="b3lyp"
        ):
        """Constructor:

        Arguments:
         - aimd_method: name of the md run method ('BOMD' or 'CURVY').
         - aimd_temperature: temperature in K the md run is performed at
         - aimd_steps: number of md steps.
         - time_step: time step in atomic units. 1 a.u. = 0.00242 fs.
         - aimd_init_veloc: how the volcities should be initialized 
           ('THERMAL', 'ZPE', 'QUASICLASSICAL')
        """

        self._job_name = job_name
        self._molecule = molecule

        self._rem_array = qc.rem_array()
        self._rem_array.jobtype("aimd")
        self._rem_array.method(method)
        self._rem_array.basis_set(basis_set)

        self._rem_array.aimd_method(aimd_method)
        self._rem_array.aimd_steps(aimd_steps)
        self._rem_array.aimd_temperature(aimd_temperature)
        self._rem_array.aimd_time_step(time_step)
        self._rem_array.aimd_initial_velocities(aimd_init_veloc)


        super.__init__(job_name, molecule)



        
    