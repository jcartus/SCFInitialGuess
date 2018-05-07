"""This script will train a neural network for any of the 
given molecules and store the result


Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from os.path import join

import numpy as np
import tensorflow as tf
import argparse

from SCFInitialGuess.nn.networks import EluTrNNN
from SCFInitialGuess.nn.training import Trainer
from SCFInitialGuess.nn.cost_functions import RegularizedMSE
from SCFInitialGuess.utilities.dataset import make_butadien_dataset, extract_triu
from SCFInitialGuess.utilities.usermessages import Messenger as msg

DIM = 26

def prep_dataset():
    """Fetching the dataset """
    
    def load_triu(S, P, dim):
        
        return [extract_triu(s, dim) for s in S], [extract_triu(p, dim) for p in P]


    folder = join("butadien", "data")

    dataset, molecules = make_butadien_dataset(
        np.load(join(folder, "molecules.npy")),
        *np.load(join(folder, "dataset.npy")) 
    )

    return dataset#, molecules

def train_network(dataset):
    """Training the network"""


    trainer = Trainer(
        EluTrNNN([DIM**2, DIM**2, DIM**2, DIM**2]),
        cost_function=RegularizedMSE(alpha=1e-7),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-3)
    )

    trainer.setup()

    network, sess = trainer.train(
        dataset,
        convergence_threshold=5e-7
    )
    
    return trainer, network, sess



def main():

    msg.info("Traing a network for butadien", 2)

    msg.info("Fetching dataset ... ", 2)
    dataset = prep_dataset()

    msg.info("Train ... ", 2)
    trainer, network, sess = train_network(dataset)

    user_input =  msg.input(
        "Keep this network for butadien " + " (y for yes)?"
    )

    if user_input.upper() == "Y":
        save_path = join("butadien/data", "network.npy")
        network.export(sess, save_path)
        msg.info("Exported network to: " + save_path, 2)
    else:
        msg.info("Network discarded ...", 2)

    msg.info("All done. Bye bye..", 2)

if __name__ == '__main__':
    main()