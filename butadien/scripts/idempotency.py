from SCFInitialGuess.nn.networks import EluTrNNN
from SCFInitialGuess.nn.training import Trainer, MSE
from SCFInitialGuess.utilities.dataset import Dataset
from SCFInitialGuess.utilities.usermessages import Messenger as msg

import numpy as np
import tensorflow as tf

from shutil import rmtree

dim = 26


class IdempotencyPenalty(MSE):
    def __init__(self, coupling=1e-4):
        
        self.coupling = coupling
        
    def function(self, network, y_placeholder):

        error = \
            super(IdempotencyPenalty, self).function(network, y_placeholder)

        penalty = self.idempotency_error(network) * self.coupling

        cost = error + penalty

        tf.summary.scalar("idempotency_error", penalty)
        tf.summary.scalar("total_loss", cost)

        return cost
    
    def idempotency_error(self, network):
        p = tf.reshape(network.output_tensor, [-1, dim, dim])
        s = tf.reshape(network.input_tensor, [-1, dim, dim])

        lhs = tf.matmul(tf.matmul(p, s), p)
        rhs = 2 * p
        return tf.reduce_mean(tf.norm(tf.abs(lhs - rhs), axis=(1,2)) **2 )

def main():   

    S, P = np.load("notebooks/butadien/dataset.npy")

    dataset = Dataset(S, P, split_test=0.25)


    trainer = Trainer(
        EluTrNNN([dim**2, 200, 100, dim**2], log_histograms=True),
        cost_function=IdempotencyPenalty(coupling=1e-6),
        optimizer=tf.train.AdamOptimizer(learning_rate=5e-3)
    )

    trainer.setup()
    network_idem, sess_idem = trainer.train(
        dataset,
        convergence_threshold=1e-5,
        summary_save_path="log2/idem"
    )
    graph_idem = trainer.graph

    with trainer.graph.as_default():
        error = trainer.cost_function.idempotency_error(network_idem)
        error_val = sess_idem.run(error, {network_idem.input_tensor: dataset.testing[0]})

    msg.info("Achieved idempotency error: " + str(error_val), 2)

if __name__ == '__main__':
    main()