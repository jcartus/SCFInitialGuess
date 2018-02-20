"""This module will contain everything needed for communation to PySCF

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os import linesep
from os.path import join


import numpy as np

from pyscf import gto, scf

from utilities.usermessages import Messenger as msg



class RHFJob(object):
    """Class for a resitrcted hf calculation with pyscf."""

    def __init__(self, 
        job_name, 
        molecule, 
        basis_set="6-311++g**",
        initial_guess=None
        ):

        self._job_type = "RHF single point"
        self.job_name = job_name
        self.molecule = molecule

        self.basis = basis_set
        self.initial_guess = initial_guess
        
        self._job = self._setup()

        self._S = None
        self._H = None
        self._P = None
        self._F = None

    @property
    def S(self):
        if self._S is None
            self._S = self._job.get_ovlp()
        return self._S

    @property
    def H(self):
        if self._H is None
            self._H = self._job.get_hcore()
        return self._H

    @property
    def P(self):
        if self._P is None:
            self._P = self._job.make_rdm1()
        return self._P

    @property
    def F(self):
        if self._F is None:
            self._F = self._job.get_fock()
        return self._F

    def _setup(self):
        """set up the pyscf job"""

        # create pyscf molecule
        pyscf_molecule = gto.Mole()
        pyscf_molecule.build(
            atom=self.molecule.geometry,
            basis=self.basis
        )

        #--- create job and set detail settings if desired ---
        job = scf.RHF(pyscf_molecule)

        if not self.initial_guess is None:
            job.init_guess = self.initial_guess
        #---

        return job

    def run(self):
        """run the job created in setup"""
        return self._job.kernel()

    def export_results(self, directory):
        """Save calculated results (so far just a logfile and the matrices) to directory"""

        # save matrices
        np.save(join(directory, self.job_name + ".S.npy"), self.S)
        np.save(join(directory, self.job_name + ".H.npy"), self.H)
        np.save(join(directory, self.job_name + ".F.npy"), self.F)

        # save log
        with open(join(directory, self.job_name + ".log"), "w") as f:
            f.write(self._create_log_str)

    @property
    def _create_log_str(self):

        def separator():
            return "-------------------------------------------"

        # general data
        log  = self.job_name + linesep() + linesep()
        log += self._job_type + linesep()
        log += self.basis + linesep()
        log += self._job.init_guess + linesep()
        log += linesep() + separator + linesep()

        # molecule information
        log += "$molecule" + linesep()
        log += "0 1" + linesep()
        log += linesep().join("   ".join(self.molecule.geometry)) + linesep()
        log += "$end" + linesep()

        return log

    def export_results_qChemFormat(self, directory):
        """Save the results like it would habe been for qchem"""
        raise ValueError("This was not implemented yet.")




    
