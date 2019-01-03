""" Test the schemes to generate fock or density matrices."""

import numpy as np
import unittest
from pyscf.scf import hf
from pyscf.gto import Mole

from SCFInitialGuess.utilities.dataset import \
    fock_from_density, density_from_fock


# HACK/TODO: gwh scheme from jycartus/pyscf fork.
def init_guess_by_wolfsberg_helmholtz(mol):
    """Diagonal will be taken from core hamiltonian, the off diagonal elements
    are interpolated by wolfsberg helmholtz scheme. 
    
        H_ji = k_ji (H_ii + H_ij) S_ij / 2, with k_ij =1.75
    
    (Generalized Wolfsberg Helmholtz GWH). See here:
    http://www.q-chem.com/qchem-website/manual/qchem50_manual/sect-initialguess.html

    M. Wolfsberg and L. Helmholz, J. Chem. Phys. 20, 837 (1952). 
    """
    from pyscf.scf.hf import *

    H = numpy.diag(get_hcore(mol))

    k = numpy.ones((len(H), len(H))) * 1.75 - \
        numpy.diag(numpy.ones(H.shape)) * 0.75  
    S = get_ovlp(mol)

    H = k * numpy.add.outer(H, H) * S / 2

    mo_energy, mo_coeff = eig(H, S)
    mo_occ = get_occ(SCF(mol), mo_energy, mo_coeff)
    
    return make_rdm1(mo_coeff, mo_occ)


class TestGWH(unittest.TestCase):

    def setUp(self):

        self.molecule = self.make_molecule()
        
    def make_molecule(self):
        mol = Mole()
        mol.atom = """
        O 0 0 0
        H 1 0 0
        H 0 1 0
        """
        mol.basis = "sto-3g"

        mol.build()

        return mol

    def test_gwh_by_fock(self):
        from SCFInitialGuess.construction.fock import gwh_scheme

        mol = self.molecule

        h = hf.get_hcore(mol)
        s = hf.get_ovlp(mol)

        f_gwh = gwh_scheme(np.diag(h), s)
        p_gwh = density_from_fock(f_gwh, s, mol)
        f_gwh = fock_from_density(p_gwh, s, h, mol)

        #TODO: use pyscf method
        p_ref = init_guess_by_wolfsberg_helmholtz(mol)
        f_ref = fock_from_density(p_ref, s, h, mol)

        np.testing.assert_allclose(f_gwh, f_ref, atol=1e-5)


    def test_gwh_by_density(self):
        from SCFInitialGuess.construction.fock import gwh_scheme

        mol = self.molecule

        h = hf.get_hcore(mol)
        s = hf.get_ovlp(mol)

        f_gwh = gwh_scheme(np.diag(h), s)
        p_gwh = density_from_fock(f_gwh, s, mol)

        #TODO: use pyscf method
        p_ref = init_guess_by_wolfsberg_helmholtz(mol)
        
        np.testing.assert_allclose(p_gwh, p_ref)



if __name__ == '__main__':
    unittest.main()
