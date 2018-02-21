"""This is a qick and dirty solution to calculate all inputs 
for 'molecules' that consists of only 1 atom.

Author:
    Johannes Cartus. QCIEP, TU Graz
"""
import numpy as np

from utilities.dataset import Molecule
from utilities.qChem import QChemSinglePointCalculation
from utilities.usermessages import Messenger as msg

from os.path import join, normpath


def main():
    atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Si", "Br"]

    out_dir = normpath("/home/jcartus/Documents/SCFInitialGuess/atoms")

    for atom in atoms:
        msg.info("Doing " + atom)
        mol = Molecule(
            [atom], 
            [np.array([0.0, 0.0, 0.0])],
            atom
        )

        job = QChemSinglePointCalculation(
            atom,
            mol,
            scf_print=1,
            scf_final_print=1
        )

        try:
            job.run_in_directory(join(out_dir, atom))
            msg.info("Finshed " + atom)
        except Exception as ex:
            msg.info("Smth went wrong: " + str(ex))


if __name__ == '__main__':
    main()