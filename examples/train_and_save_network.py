"""This is a demo script to train a neural network and to store the result
and load it again.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import normpath, join, realpath, dirname, isfile
from os import listdir, remove

import tensorflow as tf
import numpy as np

from SCFInitialGuess.utilities.dataset import assemble_batch, Dataset
from SCFInitialGuess.utilities.constants import number_of_basis_functions as N_BASIS
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.nn.networks import EluTrNNN, EluFixedValue
from SCFInitialGuess.nn.training import train_network


def main(species="H"):

    #--- assemble the dataset ---
    root_directory = normpath(join(dirname(realpath(__file__)), "../"))
    dataset_source_folder = join(root_directory, "dataset/")
    sources = [
        join(dataset_source_folder, directory) \
            for directory in ["GMTKN55"]
    ]

    dataset = Dataset(*assemble_batch(sources, species))
    #---

    #--- setup and train the network ---
    dim = N_BASIS[species]

    structure = [dim, 25, dim]

    network = EluTrNNN(structure)

    network, sess = train_network(network, dataset)
    #---

    save_path = join(root_directory, "tmp" + species + ".npy")
    #try:
    #--- save trained model ---      
    save_object = [
        network.structure,
        network.weights_values(sess),
        network.biases_values(sess)
    ]

    np.save(
        save_path,
        save_object
    )
    sess.close()
    msg.info("Session closed", 1)
    #---


    #--- load and reinitialize model ---
    msg.info("Starting new session and loading the model ...", 1)
    sess = tf.Session()
    model = np.load(save_path)

    new_network = EluFixedValue(*model)
    new_network.setup()
    sess.run(tf.global_variables_initializer())

    #finally:
    if isfile(save_path):
        remove(save_path)
    #---

if __name__ == '__main__':
    main()