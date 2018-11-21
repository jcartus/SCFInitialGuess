"""This script will fetch the molecules and store them together with 
the dataset of S/P-Matrices in numpy binaries.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import os 
from os.path import join

from pyscf.scf import hf
import numpy as np


from SCFInitialGuess.utilities.dataset import \
    Molecule, QChemResultsReader, XYZFileReader
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.dataset import do_scf_runs


#--- inputs ---
BASIS = "6-311++g**"
FOLDER = "thesis/dataset/QM9/dsgdb9nsd.xyz"
INDEX = None
PREFIX = "QM9-300"
FRACTION = 10000
#---




def main():        

    data_folder = FOLDER
    index_file=INDEX

    msg.info("Fetching molecules", 2)
    molecules = QChemResultsReader.fetch_from_folder(data_folder, BASIS)
    #molecules = XYZFileReader.read_folder(data_folder, basis=BASIS)[:300] # todo calculate full list!!


    msg.info("Starting SCF Calculation", 2)

    if len(molecules) > FRACTION:
        msg.info("Too many samples. Doing SCF calculations batchwise ...", 2)

        number_of_batches = int(np.ceil(len(molecules) / FRACTION))
    
        # if dataset is too large do calculation in subsets
        for i in range(number_of_batches):
            
            msg.info("Starting Subset {0}/{1}".format(
                i + 1, 
                number_of_batches
            ), 2)

            folder = data_folder + "/SubSet_" +  str(i + 1) + "/"

            try:
                subset = molecules[i*FRACTION : (i+1) * FRACTION]

                if not os.path.exists(folder):
                    os.mkdir(folder)

                S, P, F = do_scf_runs(subset)

                msg.info("Exporting Results", 1)

                msg.info("S & P ...", 1)
                np.save(folder + "S" + PREFIX + ".npy", S)
                np.save(folder + "P" + PREFIX + ".npy", P)
                np.save(folder + "F" + PREFIX + ".npy", F)

                msg.info("Molecules ...", 1)
                np.save(folder + "molecules" + PREFIX + ".npy", subset)
            
            except Exception as ex:
                msg.error("Something went wrong: " + str(ex))

    else:

        S, P, F = do_scf_runs(molecules)

        msg.info("Exporting Results", 2)

        msg.info("S & P ...", 1)
        np.save(data_folder + "S" + PREFIX + ".npy", S)
        np.save(data_folder + "P" + PREFIX + ".npy", P)
        np.save(data_folder + "F" + PREFIX + ".npy", F)

        msg.info("Molecules ...", 1)
        np.save(data_folder + "molecules" + PREFIX + ".npy", molecules)

    msg.info("All Done. ", 2)

if __name__ == '__main__':
    main()


