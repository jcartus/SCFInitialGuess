"""This file will contains all QChem utility functions and classes,
used to controll qChem (mostly via PyQChem)

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from os.path import exists

import pyQChem as qc

from utilities.usermessages import Messenger as msg

class QChemJob(object):
    """Base model for QCHem Jobs. Implements functions to run a job, 
    so derived classes only have to implement job properties (in rem_array).
    """
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
        job_type,
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
        self._job_type = job_type

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
        method="HF",
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
            job_type="AIMD",
            basis_set=basis_set,
            method=method,
            **kwargs
        )

        self._rem_array.jobtype("aimd")

        self._rem_array.aimd_method(aimd_method)
        self._rem_array.aimd_steps(str(aimd_steps))
        self._rem_array.aimd_temperature(str(aimd_temperature))
        self._rem_array.aimd_time_step(str(time_step))
        self._rem_array.aimd_initial_velocities(aimd_init_veloc)

class QChemSinglePointCalculation(QChemSCFJob):
    """This class controlls a qchem job of the type sp.
    See also here: https://www.q-chem.com/qchem-website/manual/qchem44_manual/sec-SCF_job-control.html
    """

    def __init__(self,
        job_name,
        molecule,
        basis_set="6-311++G**",
        method="HF",
        **kwargs
    ):

        super(QChemSinglePointCalculation, self).__init__(
            job_name=job_name,
            molecule=molecule,
            job_type="SP",
            basis_set=basis_set,
            method=method,
            **kwargs
        )

        
        self._rem_array.jobtype("SP")

