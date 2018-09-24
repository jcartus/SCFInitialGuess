"""This script will fetch the molecules and store them together with 
the dataset of S/P-Matrices in numpy binaries.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from os import listdir
from os.path import join

from pyscf.scf import hf
import numpy as np


from SCFInitialGuess.utilities.dataset import Molecule, QChemResultsReader
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.dataset import do_scf_runs


#--- inputs ---
dim = 46
BASIS = "3-21++g*"
FOLDER = "thesis/dataset/TSmall/"
INDEX = None
PREFIX = "TSmall"
#---


def main():        

    data_folder = FOLDER
    index_file=INDEX

    msg.info("Fetching molecules", 2)
    molecules = QChemResultsReader.fetch_from_folder(data_folder, BASIS)
    

    if index_file is None:
        index = np.arange(len(molecules))
        np.random.shuffle(index)
    else:
        index = np.load(index_file)

    molecules =[molecules[i] for i in index]

    msg.info("Starting SCF Calculation", 2)
    S, P, F = do_scf_runs(molecules)

    msg.info("Exporting Results", 2)
    msg.info("Index ...", 1)
    np.save(data_folder + "index" + PREFIX + ".npy", index)

    msg.info("S & P ...", 1)
    np.save(data_folder + "S" + PREFIX + ".npy", np.array(S).reshape(-1, dim**2))
    np.save(data_folder + "P" + PREFIX + ".npy", np.array(P).reshape(-1, dim**2))
    np.save(data_folder + "F" + PREFIX + ".npy", np.array(F).reshape(-1, dim**2))

    msg.info("Molecules ...", 1)
    np.save(data_folder + "molecules" + PREFIX + ".npy", molecules)

    msg.info("All Done. ", 2)

if __name__ == '__main__':
    main()


