"""This is a first comparison of initialguess implemented in pyscf and the 
network model.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from pyscf import scf

from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities import Molecule
from SCFInitialGuess.diagonal import nn_guess

def main():
    msg.info("Welcome to the method benchmark.", 2)

    #--- set up a H2 molecule ---
    msg.info("Setting up molecule: H_2")
    positions = [
        [0, 0, 0.0],
        [0, 0, 1.1],
        [0, 0, 2.2],
        [0, 0, 3.3]
    ]
    mol = Molecule(['H' for i in range(4)], positions, 'H4')
    pyscf_mol = mol.get_pyscf_molecule()
    #---

    #--- test network model ---
    msg.info("Starting SCF with method: nn guess")
    mf = scf.RHF(pyscf_mol)
    mf.init_guess = "nn"
    S = mf.get_ovlp()
    dm = nn_guess(mol, S)
    mf.verbose = 4 # todo instead extract number of cycles and plot it with msg
    mf.kernel(dm)
    #---

    #--- test pyscf methods ---
    for method in ['minao', 'atom', '1e']:
        msg.info("Starting SCF with method: " + method, 1)
        mf = scf.RHF(pyscf_mol)
        mf.verbose = 4 # todo instead extract number of cycles and plot it with msg
        mf.init_guess = method
        mf.run()
    #---

    

if __name__ == '__main__':
    main()