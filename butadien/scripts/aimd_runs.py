"""This script will run md runs for all mol files in a folder.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from os import listdir
from os.path import join

from pyQChem import inputfile
from pyQChem.utilities import _readinput
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.misc import cd

import multiprocessing as mp

def worker(rem, molecule, index):

    msg.info("Starting Job: " + str(index))

    try:
        job = inputfile()
        job.add(rem)
        job.add(molecule)

        job.run(name="Butadien_{0}.mol".format(index))

        msg.info("Finished JOb: " + str(index))
    except:
        msg.warn("Job failed: " + str(index))

def fetch_rem(inp_file):
    """Read rem from a settings file or another input file"""
    return _readinput(inp_file).rem

def fetch_molecules(folder):
    """Read all molecules from .mol files in a folder 
    (in pyqchem molecule format)
    """
    # TODO put this funciton into the SCFInitialGuess.utilities.dataset.Molcule 
    # class!

    files = [file for file in listdir(folder) if ".mol" in file]

    return [_readinput(join(folder, f)).molecule for f in files]
    


def main(folder="/home/jcartus/Repos/SCFInitialGuess/butadien/data/MDRuns/", number_of_processes = 6):

    msg.info("Fetching aimd run settings ...", 1)
    rem = fetch_rem(join(folder, "MDRunSettings.inp"))
    
    msg.info("Fetching qchem molecules ...", 1)
    molecules = fetch_molecules(folder)

    msg.info("Setting up parallelisation ...", 1)
    pool = mp.Pool(processes=number_of_processes)
    
    with cd(folder):
        msg.info("Starting the calculations ...", 1)
        for i, mol in enumerate(molecules):
            pool.apply_async(worker, (rem, mol, i + 1))

        pool.close()
        pool.join()
        msg.info("Closed worker pool.")

    msg.info("Calculations Done", 1)

if __name__ == '__main__':
    main()