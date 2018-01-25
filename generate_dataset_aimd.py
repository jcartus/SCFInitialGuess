"""
This module will read sample geometries from a database folder
and do some md runs with it to generate an input data set.

Authors:
    - Johannes Cartus, QCIEP, TU Graz
TODO:
    - print some info about progress of program, i.e. via  a Printer class.s
"""

from os.path import exists, isdir, isfile, join, splitext, normpath, basename
from os import listdir, makedirs, remove
from shutil import move
from warnings import warn

import multiprocessing as mp
import argparse

from utilities.usermessages import Messenger as msg

from utilities.dataset import PyQChemDBReader, QChemMDRun, produce_randomized_geometries


def main():

    parser = argparse.ArgumentParser(
        prog='PROG',
        description="This program will read molecule geometries from a data" + 
            "base folder, generate a few geometries and do md runs with qChem on them" 
        
    )

    parser.add_argument(
        "-s", "--source", 
        required=False,
        help="The (path to the) data base folder, which contains the original molecules",
        metavar="source directory",
        dest="source",
        default=normpath("data_base/s22/")
    )

    parser.add_argument(
        "-d", "--destination", 
        required=False,
        help="The (path to the) results folder, where the calculationresults can be stored in",
        metavar="destination directory",
        dest="destination",
        default=normpath("result/")
    )

    parser.add_argument(
        "-f", "--multiplication-factor", 
        default=0,
        required=False,
        help="The number of randomized geometries generated for every " + \
            "molecule in the data base",
        type=int,
        dest="amplification"
    )

    parser.add_argument(
        "-p", "--processes", 
        default=4,
        required=False,
        help="The number of worker processes used " + \
            "to set up worker pool for parallelisation",
        type=int,
        dest="number_of_processes"
    )

    parser.add_argument(
        "--aimd-steps", 
        default=1,
        required=False,
        help="The number md steps to be done in each aimd run",
        type=int,
        dest="aimd_steps"
    )

    
    args = parser.parse_args()

    #--- the actual program ---
    msg.info("Welcome! Let's do some AIMD runs.", 2)

    
    # fetch data from data base
    molecules = PyQChemDBReader.read_database(args.source)
    if args.amplification:
        molecules = produce_randomized_geometries(molecules, args.amplification)

    # prepare result dir if not exists
    if not isdir(args.destination):
        makedirs(args.destination)

    # todo get num of trherads dynamically. evtl as argument?
    pool = mp.Pool(processes=args.number_of_processes)
    msg.info(
        "Create worker pool of " + str(args.number_of_processes) + " processes."
    )
    for mol in molecules:
        pool.apply_async(qchem_execution_section, (mol, args))
    pool.close()
    pool.join()
    msg.info("Closed worker pool.")

    msg.info("All done. See you later ...", 2)
    #---



# define paralell section    
def qchem_execution_section(mol, args):
    run = QChemMDRun(
        mol.full_name, 
        mol, 
        aimd_steps=args.aimd_steps,
        scf_convergence=5,
        scf_print=1,
        scf_final_print=1
        )

    # add path for result as opton to qChemmdrun!!
    run.run()

    msg.info("Finished run.")
    msg.info("Cleaning up and Moving results to: " + args.destination)
    for ext in ["in", "out", "sh"]:
        
        fname = run.job_name + "." + ext
        try:
            if ext == "out":
                move(fname, join(args.destination,fname))
            else:
                remove(fname)
        except Exception as ex:
            if ext == "out":
                msg.warn("Could not move {0}: ".format(fname) + str(ex))    
            else:
                msg.warn("Could not delete {0}: ".format(fname) + str(ex))    
        finally:
            if isfile(fname):
                remove(fname)

if __name__ == '__main__':
    main()
