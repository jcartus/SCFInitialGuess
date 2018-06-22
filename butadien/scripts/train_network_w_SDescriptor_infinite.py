"""This script will train a neural network with the S matrix a input descriptor
on an with the continuous trainer on the big md run dataset.
Only Upper triu of course.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from os.path import join

import numpy as np
import tensorflow as tf
import argparse

from SCFInitialGuess.nn.networks import EluTrNNN, EluFixedValue
from SCFInitialGuess.nn.training import ContinuousTrainer
from SCFInitialGuess.nn.cost_functions import RegularizedMSE
from SCFInitialGuess.utilities.dataset import make_butadien_dataset, extract_triu
from SCFInitialGuess.utilities.usermessages import Messenger as msg

DIM = 26

def prep_dataset():
    """Fetching the dataset """
    
    def load_triu(S, P, dim):
        
        return [extract_triu(s, dim) for s in S], [extract_triu(p, dim) for p in P]


    folder = join("butadien", "data", "400")
    

    dataset, molecules = make_butadien_dataset(
        np.load(join(folder, "molecules400.npy")),
        *load_triu(
            np.load(join(folder, "S400.npy")),
            np.load(join(folder, "P400.npy")),
            DIM
        ),
        test_samples=100
    )

    return dataset#, molecules

def train_network(dataset, network, save_path, test_error):
    """Training the network"""

    trainer = ContinuousTrainer(
        network,
        cost_function=RegularizedMSE(alpha=1e-7),
    )

    trainer.setup()

    network, sess = trainer.train(
        dataset,
        save_path,
        evaluation_period=1000,
        mini_batch_size=100,
        old_error=test_error
    )
    
    return trainer, network, sess



def main():
    #todo this funciton and taining should become part of the library!!
    # sodass man nur mehr savepath und dataset angeben muss!

    msg.info("Traing a network for butadien", 2)

    msg.info("Fetching dataset ... ", 2)
    dataset = prep_dataset()

    save_path = "butadien/data/networks/networkSMatrixBigData.npy"


    user_input =  msg.input(
        "This will overwrite the model at " + save_path + \
        "Are you sure you want that? (y for yes)"
    )

    if user_input.upper() != "Y":
        msg.info("Aborting", 2)
        return

    msg.info("Try to fetch current model")
    try:
        
        model = np.load(save_path, encoding="latin1")
        structure, weights, biases = model[0], model[1], model[2]
        network = EluFixedValue(structure, weights, biases)
        test_error = model[3]

        user_input =  msg.input(
            "Model found with test error :  " + str(test_error) + \
            ". Do you want to continue to train it? (y for yes)"
        )

        if user_input.upper() != "Y":
            msg.info("Creating new network", 2)
            model = None

    except:
        model = None
        
    if model is None:
        dim_triu = int(DIM * (DIM + 1) / 2)
        structure = [dim_triu, int(dim_triu * 0.75), int(dim_triu * 0.5), dim_triu, dim_triu]
        test_error = 1e10


    msg.info("Train ... ", 2)
    
    network = EluTrNNN(structure)

    train_network(dataset, network, save_path, test_error)

    
    msg.info("All done. Bye bye..", 2)

if __name__ == '__main__':
    main()
