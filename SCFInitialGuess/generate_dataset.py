"""
This module will read sample geometries from a database folder
and do some single point calculations with it to generate an input data set.

Authors:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import exists, isdir, isfile, join, splitext, normpath, basename
from os import listdir, makedirs, remove
from shutil import move
from warnings import warn

import multiprocessing as mp
import argparse

from SCFInitialGuess.utilities.usermessages import Messenger as msg

from SCFInitialGuess.utilities.dataset import XYZFileReader, produce_randomized_geometries
from SCFInitialGuess.utilities.qChem import QChemSinglePointCalculation


def main(
    source, 
    destination,
    amplification=0,
    number_of_processes=4,
    ):

    #--- the actual program ---
    msg.info("Welcome! Let's do some SP calculations.", 2)

    
    # fetch data from data base
    msg.info("Fetching molecules ...", 2)
    molecules = XYZFileReader.read_tree(source)
    if amplification:
        molecules = produce_randomized_geometries(molecules, amplification)

    # assemble jobs
    msg.info("Assembling jobs ...", 2)

    jobs = []
    for mol in molecules:
        jobs.append(
            QChemSinglePointCalculation(
                mol.full_name,
                mol,
                scf_convergence=5,
                scf_print=1,
                scf_final_print=1
            )
        )
    

    # prepare result dir if not exists
    if not isdir(destination):
        msg.info("Create destination folder at : " + destination)
        makedirs(destination)

    msg.info("Starting calculations ...", 2)
    # todo get num of trherads dynamically. evtl as argument?
    pool = mp.Pool(processes=number_of_processes)
    msg.info(
        "Create worker pool of " + str(number_of_processes) + " processes."
    )

    # run calculations in parallel
    for job in jobs:
        pool.apply_async(parallel_job, (job, destination, ))
    
    # clean up pool
    pool.close()
    pool.join()
    msg.info("Closed worker pool.")

    msg.info("All done. See you later :*", 2)
    #---

def parallel_job(job, destination_folder):
    """This warpper is reqire as the instance method cannot be pickled """
    
    try:
        job.run_in_directory(join(destination_folder, job.job_name))
        msg.info("Finished calculation of " + job.job_name, 1)
    except Exception as ex:
        msg.warn("There was a problem: " + str(ex))

def clean_up(job, destination_folder):
    """Delete temporary files and move results to results folder"""

    msg.info("Cleaning up and Moving results to: " + destination_folder)

    for ext in ["in", "out", "sh", "dat"]:
        
        fname = job.job_name + "." + ext
        try:
            if ext in ["out", "dat"]:
                move(fname, join(destination_folder,fname))
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

    parser = argparse.ArgumentParser(
        prog='PROG',
        description="This program will read molecule geometries from a data" + 
            "base folder, generate a few geometries and do md runs with qChem" +
                " on them" 
    )

    parser.add_argument(
        "-s", "--source", 
        required=False,
        help="The (path to the) data base folder, which contains the original" +
            " molecules",
        metavar="source directory",
        dest="source",
        default=normpath("data_base/s22/")
    )

    parser.add_argument(
        "-d", "--destination", 
        required=False,
        help="The (path to the) results folder, where the calculationresults " +
            "can be stored in",
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
    
    args = parser.parse_args()
    
    main(
        source=args.source,
        destination=args.destination,
        amplification=args.amplification,
        number_of_processes=args.number_of_processes,
    )
