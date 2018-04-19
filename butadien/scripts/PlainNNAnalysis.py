"""This script will investigate as to how far my neural networks fullfill
the P matrix requirements.^

"""

import numpy as np
import tensorflow as tf

from SCFInitialGuess.nn.networks import EluTrNNN, SeluTrNNN
from SCFInitialGuess.nn.training import Trainer
from SCFInitialGuess.nn.cost_functions import RegularizedMSE, MSE
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.dataset import Dataset

from functools import reduce

dim = 26
msg.print_level = 1

def fetch_dataset():
    #--- the dataset ---
    S, P = np.load("butadien/data/dataset.npy")

    ind_cut = 150
    index = np.arange(200)
    np.random.shuffle(index)

    S_test = np.array(S)[index[ind_cut:]]
    P_test = np.array(P)[index[ind_cut:]]

    S_train = np.array(S)[index[:ind_cut]]
    P_train = np.array(P)[index[:ind_cut]]

    dataset = Dataset(np.array(S_train), np.array(P_train), split_test=0.0)

    dataset.testing = (Dataset.normalize(S_test, mean=dataset.x_mean, std=dataset.x_std)[0], P_test)
    #---
    return dataset

def analysis(dataset, network_instance, costs):

    

    #--- the nn stuff ---
    trainer = Trainer(
        network=network_instance,
        cost_function=costs
    )

    trainer.setup()

    graph = trainer.graph
    #---


    #--- error functions ---
    with graph.as_default():

        x = trainer.network.input_tensor
        f = trainer.network.output_tensor
        y = trainer.target_placeholder

        p_batch = tf.reshape(f, [-1, dim, dim])

        transposed = tf.matrix_transpose(p_batch)
        symmetry_error = tf.reduce_mean(tf.abs(p_batch - transposed), axis=0)

        s_raw = dataset.inverse_input_transform(x)
        s_batch = tf.reshape(s_raw, [-1, dim, dim])
        
        idempotency_error = \
            tf.reduce_mean(
                tf.abs(
                    reduce(tf.matmul, (p_batch, s_batch, p_batch)) - 2 * p_batch
                ), 
            axis=[1,2])
        
        occupancy = tf.trace(tf.matmul(p_batch, s_batch)) 

        absolute_error = tf.reduce_mean(tf.abs(y - f), axis=1)
        #absolute_error = tf.losses.absolute_difference(y, f)

    #---

    total_err_abs = []
    total_err_sym = []
    total_err_idem = []
    total_err_occ = []
    for i in range(10):
        
        network, sess = trainer.train(
            dataset,
            convergence_threshold=1e-6
        )


        with graph.as_default():
        
            err_abs = \
                sess.run(absolute_error, {x: dataset.testing[0], y: dataset.testing[1]})
            err_sym = sess.run(symmetry_error, {x: dataset.testing[0]})
            err_idem = sess.run(idempotency_error, {x: dataset.testing[0]})
            err_occ = sess.run(occupancy, {x: dataset.testing[0]}) - 30
            
            total_err_abs.append(np.mean(err_abs))
            total_err_sym.append(np.mean(err_sym))
            total_err_idem.append(np.mean(err_idem))
            total_err_occ.append(np.mean(err_occ))

            def stats(x):
                return np.mean(x), np.std(x)

            print("----------------------------------------")
            print("Network " + str(i+1))

            print("Abs: {:0.5E} +- {:0.5E}".format(*stats(err_abs)))
            print("Sym: {:0.5E} +- {:0.5E}".format(*stats(err_sym)))
            print("Ide: {:0.5E} +- {:0.5E}".format(*stats(err_idem)))
            print("Occ: {:0.5E} +- {:0.5E}".format(*stats(err_occ)))

    print("=========================================")
    print("Abs: {:0.5E} +- {:0.5E}".format(*stats(total_err_abs)))
    print("Sym: {:0.5E} +- {:0.5E}".format(*stats(total_err_sym)))
    print("Ide: {:0.5E} +- {:0.5E}".format(*stats(total_err_idem)))
    print("Occ: {:0.5E} +- {:0.5E}".format(*stats(total_err_occ)))


def main():
    dataset = fetch_dataset()
    analysis(dataset, EluTrNNN([dim**2, 700, 700, dim**2]), MSE())

if __name__ == '__main__':
    main()