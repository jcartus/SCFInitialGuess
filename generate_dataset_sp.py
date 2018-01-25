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

from utilities.usermessages import Messenger as msg

from utilities.dataset import PyQChemDBReader, produce_randomized_geometries
from utilities.qChem import QChemSinglePointCalculation


def main():

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

    #--- the actual program ---
    msg.info("Welcome! Let's do some SP calculations.", 2)

    
    # fetch data from data base
    molecules = PyQChemDBReader.read_database(args.source)
    if args.amplification:
        molecules = produce_randomized_geometries(molecules, args.amplification)

    # assemble jobs
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
    if not isdir(args.destination):
        makedirs(args.destination)

    # todo get num of trherads dynamically. evtl as argument?
    pool = mp.Pool(processes=args.number_of_processes)
    msg.info(
        "Create worker pool of " + str(args.number_of_processes) + " processes."
    )

    # run calculations in parallel
    for job in jobs:
        pool.apply_async(parallel_job, (job, args.destination, ))
    
    # clean up pool
    pool.close()
    pool.join()
    msg.info("Closed worker pool.")

    msg.info("All done. See you later ...", 2)
    #---

def parallel_job(job, destination_folder):
    """This warpper is reqire as the instance method cannot be pickled """

    try:
        job.run()
        msg.info("Finished calculation of " + job.job_name)
    except Exception as ex:
        msg.warn("There was a problem: " + str(ex))
    finally:
        clean_up(job, destination_folder)
    

def clean_up(job, destination_folder):
    """Delete temporary files and move results to restults folder"""

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
    main()
