"""This is a demo script to train a neural network and to store the result

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import normpath, join
from os import listdir

import tensorflow as tf
import numpy as np

from utilities.dataset import assemble_batch, Dataset
from utilities.constants import number_of_basis_functions as N_BASIS
from utilities.usermessages import Messenger as msg

from nn.networks import EluTrNNN
from nn.training import train_network


def main(species="C"):

    #--- assemble the dataset ---
    dataset_source_folder = normpath(
        "/home/jo/Documents/SCFInitialGuess/dataset/"
    )
    sources = [
        join(dataset_source_folder, directory) \
            for directory in listdir(dataset_source_folder)
    ]

    dataset = Dataset(*assemble_batch(sources, species))
    #---

    #--- setup and train the network ---
    dim = N_BASIS[species]

    structure = [dim, dim, dim]

    network = EluTrNNN(structure)

    network, sess = train_network(network, dataset)
    #---

    #--- save trained model ---
    save_path = join(
        normpath(
            "/home/jo/Documents/SCFInitialGuess/models/"
        ),
        species + ".npy"
    )

    save_object = [
        network.structure,
        network.weights_values(sess),
        network.biases_values(sess)
    ]

    np.save(
        save_path,
        save_object
    )
    #---

if __name__ == '__main__':
    main()