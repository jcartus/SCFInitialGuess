"""
This module will read sample geometries from a database folder
and do some md runs with it to generate an input data set.

Authors:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import exists, isdir, isfile, join, splitext, normpath, basename
from os import listdir, makedirs, remove
from shutil import move
from warnings import warn

import multiprocessing as mp
import argparse


from utilities.data import PyQChemDBReader, QChemMDRun, produce_randomized_geometries


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
        default=1,
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

    # todo args richtig umsetzen
    molecules = PyQChemDBReader.read_database(args.source)
    random_molecules = produce_randomized_geometries(molecules, args.amplification)

    # prepare result dir if not exists
    if not isdir(args.destination):
        makedirs(args.destination)



    # todo get num of trherads dynamically. evtl as argument?
    pool = mp.Pool(processes=args.number_of_processes)
    for mol in random_molecules:
        pool.apply_async(qchem_execution_section(mol, args))
    pool.close()
    pool.join()




# define paralell section    
def qchem_execution_section(mol, args):
    run = QChemMDRun(mol.full_name, mol, aimd_steps=args.aimd_steps)

    # add path for result as opton to qChemmdrun!!
    run.run()
    for ext in ["in", "out", "sh"]:
        
        fname = run.job_name + "." + ext
        try:
            move(fname, join(args.destination,fname))
        except Exception as ex:
            warn("Could not move {0}: ".format(fname) + str(ex))    
        finally:
            if isfile(fname):
                remove(fname)

if __name__ == '__main__':
    main()
