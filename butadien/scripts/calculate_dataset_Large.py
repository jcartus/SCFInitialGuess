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

dim = 26
BASIS = "sto-3g"

def fetch_molecules(folder):
    
    files = [file for file in listdir(folder) if ".out" in file]
    
    files.sort()

    for i, file in enumerate(files):
        
        msg.info("Fetching: " + str(i + 1) + "/" + str(len(files)))

        molecules = QChemResultsReader.read_file(folder + file)

        for molecule_values in molecules:
            mol = Molecule(*molecule_values)
            mol.basis = BASIS
            

            yield mol



def main(data_folder="butadien/data/", index_file=None):        

    msg.info("Fetching molecules", 2)
    molecules = list(fetch_molecules(data_folder + "MDRuns/results"))

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
    np.save(data_folder + "index_Large.npy", index)

    msg.info("S & P ...", 1)
    np.save(data_folder + "S_Large.npy", np.array(S).reshape(-1, dim**2))
    np.save(data_folder + "P_Large.npy", np.array(P).reshape(-1, dim**2))
    np.save(data_folder + "F_Large.npy", np.array(F).reshape(-1, dim**2))

    msg.info("Molecules ...", 1)
    np.save(data_folder + "molecules_Large.npy", molecules)

    msg.info("All Done. ", 2)

if __name__ == '__main__':
    main()


