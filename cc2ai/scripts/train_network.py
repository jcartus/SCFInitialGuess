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

#dim ethin
DIM = {
    "ethan": 58,
    "ethen": 48,
    "ethin": 38,
    "platin": 88
}

BASIS = {
    "ethan": "6-31g**",
    "ethen": "6-31g**",
    "ethin": "6-31g**",
    "butadien": "sto-3g",
    "platin": "lanl2dz",
}

def prep_dataset(molecule_type):
    """Fetching the dataset """
    
    def load_triu(S, P, dim):
        
        return [extract_triu(s, dim) for s in S], [extract_triu(p, dim) for p in P]


    folder = join("cc2ai", molecule_type)

    dataset, molecules = make_butadien_dataset(
        np.load(join(
            folder, 
            "molecules_" + molecule_type + "_" + BASIS[molecule_type] + ".npy"
        )),
        *load_triu(*np.load(
            join(
                folder, 
                "dataset_" + molecule_type + "_" + BASIS[molecule_type] + ".npy"
            )), 
            DIM[molecule_type]
        ), 
    )

    return dataset#, molecules

def train_network(molecule_type, dataset):
    """Training the network"""


    dim_triu = int(DIM[molecule_type] * (DIM[molecule_type] + 1) / 2)

    trainer = Trainer(
        EluTrNNN([dim_triu, dim_triu, dim_triu]),
        cost_function=RegularizedMSE(alpha=1e-7),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-3)
    )

    trainer.setup()

    network, sess = trainer.train(
        dataset,
        convergence_threshold=5e-7
    )
    
    return trainer, network, sess



def main(molecule_type):

    msg.info("Traing a network for " + molecule_type, 2)

    msg.info("Fetching dataset ... ", 2)
    dataset = prep_dataset(molecule_type)

    msg.info("Train ... ", 2)
    trainer, network, sess = train_network(molecule_type, dataset)

    user_input =  msg.input(
        "Keep this network for " + molecule_type + " (y for yes)?"
    )

    if user_input.upper() == "Y":
        save_path = \
            join("cc2ai", molecule_type, "network_" +  molecule_type + ".npy")
        network.export(sess, save_path)
        msg.info("Exported network to: " + save_path, 2)
    else:
        msg.info("Network discarded ...", 2)

    msg.info("All done. Bye bye..", 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG',
        description="This program will read molecule geometries from a data" + 
            "base folder, generate a few geometries and do md runs with qChem on them" 
        
    )

    parser.add_argument(
        "-m", "--molecule", 
        required=False,
        help="name of the molecule for which to train a network",
        dest="molecule",
        choices=["ethan", "ethen", "ethin", "platin"]
    )

    args = parser.parse_args()

    main(args.molecule)