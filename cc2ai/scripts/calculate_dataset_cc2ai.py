"""This script will fetch the molecules and store them together with 
the dataset of S/P-Matrices in numpy binaries.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from os import listdir
from os.path import join

from pyscf.scf import hf
import numpy as np

from SCFInitialGuess.utilities import Molecule
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.dataset import QChemResultsReader


BASIS = "lanl2dz" #"cc-pvdz-pp"
ECP = "lanl2dz" #"cc-pvdz-pp"

MOLECULE = "platin"

# pt cc-pvdz-pp
#dim = 152
# pt lanl2dz
dim = 88


# ethan
#dim = 58

# ethen
#dim = 48

# ethin
#dim = 38

def fetch_molecules(folder):
    
    files = [file for file in listdir(folder) if ".out" in file]
    
    files.sort()

    for i, file in enumerate(files):
        
        msg.info("Fetching: " + str(i + 1) + "/" + str(len(files)))

        molecules = QChemResultsReader.read_file(folder + file)

        for molecule_values in molecules:
            mol = Molecule(*molecule_values)
            mol.basis = BASIS
            try:
                mol.ecp = ECP
            except:
                pass

            yield mol

def scf_runs(molecules):

    S, P = [], []
    for i, molecule in enumerate(molecules):
        
        msg.info(str(i + 1) + "/" + str(len(molecules)))
        
        mol = molecule.get_pyscf_molecule()
        mf = hf.RHF(mol)
        mf.verbose = 1
        mf.run()
        
        S.append(mf.get_ovlp().reshape((dim**2, )))
        P.append(mf.make_rdm1().reshape((dim**2, )))

    return S, P

def main(data_folder="cc2ai/"):        

    data_folder += MOLECULE + "/"

    msg.info("Fetching molecules", 2)
    molecules = list(fetch_molecules(data_folder))

    msg.info("Starting SCF Calculation", 2)
    S, P = scf_runs(molecules)

    msg.info("Exporting Results", 2)
    np.save(data_folder + "dataset_" + MOLECULE + "_" + BASIS + ".npy", (S,P))
    msg.info("S & P ...", 1)
    np.save(data_folder + "molecules_" + MOLECULE + "_" + BASIS + ".npy", molecules)
    msg.info("Molecules ...", 1)

    msg.info("All Done. ", 2)

if __name__ == '__main__':
    main()


