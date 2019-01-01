""" Test the schemes to generate fock or density matrices."""

import numpy as np
import unittest
from pyscf.scf import hf
from pyscf.gto import Mole

from SCFInitialGuess.utilities.dataset import \
    fock_from_density, density_from_fock

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

        p_ref = hf.init_guess_by_wolfsberg_helmholtz(mol)
        f_ref = fock_from_density(p_ref, s, h, mol)

        np.testing.assert_allclose(f_gwh, f_ref, atol=1e-5)


    def test_gwh_by_density(self):
        from SCFInitialGuess.construction.fock import gwh_scheme

        mol = self.molecule

        h = hf.get_hcore(mol)
        s = hf.get_ovlp(mol)

        f_gwh = gwh_scheme(np.diag(h), s)
        p_gwh = density_from_fock(f_gwh, s, mol)

        p_ref = hf.init_guess_by_wolfsberg_helmholtz(mol)
        
        np.testing.assert_allclose(p_gwh, p_ref)



if __name__ == '__main__':
    unittest.main()
