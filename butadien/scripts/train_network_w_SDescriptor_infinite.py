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

from SCFInitialGuess.nn.networks import EluTrNNN
from SCFInitialGuess.nn.training import ContinuousTrainer
from SCFInitialGuess.nn.cost_functions import RegularizedMSE
from SCFInitialGuess.utilities.dataset import make_butadien_dataset, extract_triu
from SCFInitialGuess.utilities.usermessages import Messenger as msg

DIM = 26

def prep_dataset():
    """Fetching the dataset """
    
    def load_triu(S, P, dim):
        
        return [extract_triu(s, dim) for s in S], [extract_triu(p, dim) for p in P]


    folder = join("butadien", "data", "dataBigData")
    

    dataset, molecules = make_butadien_dataset(
        np.load(join(folder, "molecules_BigDataset.npy")),
        *load_triu(
            np.load(join(folder, "S_BigDataset.npy")),
            np.load(join(folder, "P_BigDataset.npy")),
            DIM
        ),
        test_samples=1000
    )

    return dataset#, molecules

def train_network(dataset, structure, save_path):
    """Training the network"""

    trainer = ContinuousTrainer(
        EluTrNNN(structure),
        cost_function=RegularizedMSE(alpha=1e-7),
    )

    trainer.setup()

    network, sess = trainer.train(
        dataset,
        save_path,
        evaluation_period=5000,
        mini_batch_size=100
    )
    
    return trainer, network, sess



def main():

    msg.info("Traing a network for butadien", 2)

    msg.info("Fetching dataset ... ", 2)
    dataset = prep_dataset()

    save_path = "butadien/data/networks/networkSMatrixBigData.npy"

    user_input =  msg.input(
        "This will overwrite the model at " + save_path + \
        "\nAre you sure you want that? (y for yes)"
    )

    if user_input.upper() != "Y":
        msg.info("Aborting", 2)
        return
        

    msg.info("Train ... ", 2)
    dim_triu = int(DIM * (DIM + 1) / 2)
    structure = [dim_triu, dim_triu, dim_triu, dim_triu, dim_triu, dim_triu, dim_triu]


    train_network(dataset, structure, save_path)

    
    msg.info("All done. Bye bye..", 2)

if __name__ == '__main__':
    main()
