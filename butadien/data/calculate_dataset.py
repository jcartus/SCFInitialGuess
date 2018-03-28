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

dim = 26

def grep_molecule(input_file):
    import re
    
    with open(input_file) as f:
            
        molecule = re.search(r"\$molecule.*\$end", f.read(), re.DOTALL)
        if molecule is None:
            raise ValueError("No molecule found in " + f.name)
        else:
            molecule = molecule.group(0)

            # cut out geometries
            geometries = molecule.splitlines()[2:-1]

    # from geometries take the species and positions
    species, positions = [], []
    for line in geometries:
        splits = line.split()
        species.append(splits[0])
        positions.append(splits[1:])

    return species, positions

def fetch_molecules(folder):
    
    files = [file for file in listdir(folder) if ".inp" in file]
    
    for i, file in enumerate(files):
        
        msg.info("Fetching: " + str(i + 1) + "/" + str(len(files)))

        mol = Molecule(*grep_molecule(join(folder, file)))
        
        mol.basis = "sto-3g"
        
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

def main(data_folder="butadien/data/"):        

    msg.info("Fetching molecules", 2)
    molecules = list(fetch_molecules(data_folder + "data"))

    msg.info("Starting SCF Calculation", 2)
    S, P = scf_runs(molecules)

    msg.info("Exporting Results", 2)
    msg.info("Molecules ...", 1)
    np.save(data_folder + "dataset.npy", (S,P))
    msg.info("S & P ...", 1)
    np.save(data_folder + "molecules.npy", molecules)

    msg.info("All Done. ", 2)

if __name__ == '__main__':
    main()


