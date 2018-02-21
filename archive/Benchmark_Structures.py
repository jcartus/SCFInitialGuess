"""This script will execute a Benchmark and train networks with different 
structures
"""


from os.path import normpath, join, isdir
from os import listdir, mkdir
from shutil import rmtree

import tensorflow as tf

from utilities.constants import number_of_basis_functions as N_BASIS
from utilities.dataset import assemble_batch, Dataset
from utilities.usermessages import Messenger as msg
from nn.training import Model, network_benchmark, CustomAdam
from nn.networks import EluTrNNN

def main():
    project_dir = normpath("/home/jcartus/Documents/SCFInitialGuess/")
    database = join(project_dir, "dataset/")

    species = "O"

    log_dir = join(project_dir, "log/" + species)
    db_names = listdir(database)
    #db_names = ["a24", "ala27", "s22", "l7", "p76", "shbc"]
    #db_names = ["a24"]
    

    dataset = Dataset(
        *assemble_batch([join(database, db) for db in db_names], species=species),
        split_test=0.1,
        split_validation=0.2
    )

    dim = N_BASIS[species]

    # define list of models to be tested
    msg.info("Welcome to the network structure benchmark.",2)

    structures = [
        [dim, dim],
        [dim, dim, dim],
        [dim, 50, dim],
        [dim, 100, dim],
        #[dim, 200, dim],
        [dim, 50, 50, dim]
        #[dim, 100, 100, dim],
        #[dim, 50, 50, 50, dim],
        #[dim, 100, 50, 100, dim],
        #[dim, 200, 100, 50, dim]
    ]

    msg.info("Investigating " + str(len(structures)) + " network structures.", 2) 

    models = [
        Model(
            "x".join(map(str, structure)), 
            EluTrNNN(structure, log_histograms=False), 
            CustomAdam(learning_rate=0.001)
        ) for structure in structures
    ]

    msg.info("Clearing old results ...", 1)

    if isdir(log_dir):
        try:
            rmtree(log_dir)
            mkdir(log_dir)
        except Exception as ex:
            msg.warn("Problem deleting old results: " + str(ex))
                
    network_benchmark(
        models, dataset, log_dir, 
        convergence_eps=1e-7,
        steps_report=500
    )


if __name__ == '__main__':
    main()