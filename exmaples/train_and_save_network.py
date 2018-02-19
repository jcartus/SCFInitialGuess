"""This is a demo script to train a neural network and to store the result

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import normpath, join, realpath, dirname
from os import listdir

import tensorflow as tf
import numpy as np

from utilities.dataset import assemble_batch, Dataset
from utilities.constants import number_of_basis_functions as N_BASIS
from utilities.usermessages import Messenger as msg

from nn.networks import EluTrNNN, EluFixedValue
from nn.training import train_network


def main(species="C"):

    #--- assemble the dataset ---
    root_directory = normpath(join(dirname(realpath(__file__)), "../"))
    dataset_source_folder = join(root_directory, "dataset/")
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
    save_path = join(root_directory, "models", species + ".npy")

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

    #--- load and reinitialize model ---
    model = np.load(save_path)

    new_network = EluFixedValue(*model)
    print(new_network)
    #---

if __name__ == '__main__':
    main()