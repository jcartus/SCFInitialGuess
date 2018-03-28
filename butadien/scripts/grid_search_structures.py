from SCFInitialGuess.nn.networks import SeluTrNNN
from SCFInitialGuess.nn.training import Trainer
from SCFInitialGuess.nn.cost_functions import AbsoluteError, RegularizedMSE
from SCFInitialGuess.utilities.dataset import Dataset
from SCFInitialGuess.utilities.usermessages import Messenger as msg

from selu_network import IdempotencyPenalty

import numpy as np
import tensorflow as tf

from shutil import rmtree

dim = 26

log_file = "butadien/gridsearch/logfile.txt"

def main():   

    S, P = np.load("butadien/data/dataset.npy")

    dataset = Dataset(S, P, split_test=0.25)

    structures = sample_structures()
    for structure in uniquifiy(structures):
        try:
            investigate_structure(dataset, structure)
        except Exception as ex:
            msg.error("Something went wrong during investigation: " + str(ex))

def sample_structures(
    max_hidden_nodes=1000, 
    min_hidden_nodes=700, 
    max_hidden_layers=11,
    min_hidden_layers=1
    ):

    possible_number_of_nodes = np.arange(
        min_hidden_nodes,
        max_hidden_nodes + 50,
        50
    )

    structures = []
    for num_layers in range(min_hidden_layers + 1, max_hidden_layers + 1):
        
        # pyramid
        structures.append(
            [dim**2] + \
            list(np.fliplr(possible_number_of_nodes[-num_layers:])) + \
            [dim**2]
        )

        # constant 
        for i in range(3):
            ind = np.random.randint(len(possible_number_of_nodes))
            structures.append(
                [dim**2] + [
                    possible_number_of_nodes[ind] for j in range(num_layers)
                ] + [dim**2]
            )

        # random
        for i in range(5):
            ind = np.random.randint(
                len(possible_number_of_nodes), 
                size=num_layers
            )
            structures.append(
                [dim**2] + [
                    possible_number_of_nodes[j] for j in ind
                ] + [dim**2]
            )

    return structures

def uniquifiy(full_list):

    y = []
    for x in full_list:
        if not x in y:
            y.append(x)
    return y

def investigate_structure(dataset, structure, nsamples=10):
    
    msg.info("Investigate " + str(structure), 2)

    error_val, error_idem, error_sym = [], [], []
    for run in range(nsamples):
        
        msg.info("Starting run {0}/{1}".format(run + 1, nsamples), 2)

        trainer = Trainer(
            SeluTrNNN(
                structure, 
                log_histograms=True
            ),
            #error_function=AbsoluteError(),
            #cost_function=RegularizedMSE(alpha=1e-7),
            cost_function=IdempotencyPenalty(
                dataset.inverse_input_transform,
                coupling=1e-5
            ),
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-3)
        )

        trainer.setup()
        network, sess = trainer.train(
            dataset,
            convergence_threshold=1e-6,
            #summary_save_path="butadien/log/idem",
            mini_batch_size=15
        )


        with trainer.graph.as_default():
            y = tf.placeholder(
                    dtype="float32", 
                    shape=[None, network.structure[-1]],
                    name="y"
                )
            error_val.append(sess.run(
                AbsoluteError().function(network, y), 
                {
                    network.input_tensor: dataset.testing[0],
                    y: dataset.testing[1]
                }
            ))
            
            error_idem.append(sess.run(
                trainer.cost_function.idempotency_error(network), 
                {network.input_tensor: dataset.testing[0]}
            ))

            error_sym.append(sess.run(
                symmetry_error(network.output_tensor),
                {network.input_tensor: dataset.testing[0]}
            ))

        msg.info("Achieved absolute error: {:0.3E}".format(error_val[-1]), 2)
        msg.info("Achieved idemp. error:   {:0.3E}".format(error_idem[-1]), 2)
        msg.info("Achieved sym. error:     {:0.3E}".format(error_sym[-1]), 2)

    log(
        structure, 
        np.array(error_val), np.array(error_idem), np.array(error_sym)
    )

def symmetry_error(batch):

    transposed_batch = tf.reshape(
        tf.transpose(
            tf.reshape(
                batch, 
                [-1, dim, dim]
            ), 
            perm=[0, 2, 1]
        ), 
        [-1, dim**2]
    )

    return tf.losses.absolute_difference(batch, transposed_batch)

def log(
    structure, 
    error_val, 
    error_idem, 
    error_sym):

    msg  = str(structure) + "\n"
    msg += "Error in values: {:0.3E} +- {:0.3E}".format(
        error_val.mean(), 
        error_val.std()
    ) + "\n"
    msg += "Error in idemp:  {:0.3E} +- {:0.3E}".format(
        error_idem.mean(), 
        error_idem.std()
    ) + "\n"
    msg += "Error in sym:    {:0.3E} +- {:0.3E}".format(
        error_sym.mean(), 
        error_sym.std()
    ) + "\n"

    msg += "--------------------------------------\n\n" 

    with open(log_file, 'a') as f:
        f.write(msg)

if __name__ == '__main__':
    main()