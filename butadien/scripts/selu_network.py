from SCFInitialGuess.nn.networks import SeluTrNNN
from SCFInitialGuess.nn.training import Trainer, MSE, RegularizedMSE
from SCFInitialGuess.utilities.dataset import Dataset
from SCFInitialGuess.utilities.usermessages import Messenger as msg

import numpy as np
import tensorflow as tf

from shutil import rmtree

dim = 26


class IdempotencyPenalty(MSE):
    def __init__(self, input_transformation, coupling=1e-4):
        
        self.coupling = coupling
        self.input_transformation = input_transformation
        
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
        s = tf.reshape(
            self.input_transformation(network.input_tensor), 
            [-1, dim, dim]
        )
        

        lhs = tf.matmul(tf.matmul(p, s), p)
        rhs = 2 * p
        return tf.reduce_mean(tf.norm(tf.abs(lhs - rhs), axis=(1,2)) **2 )


def main():   

    S, P = np.load("butadien/data/dataset.npy")

    dataset = Dataset(S, P, split_test=0.25)


    trainer = Trainer(
        SeluTrNNN(
            [dim**2, 400, 400, 400, 400, 400, 400, dim**2], 
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
        convergence_threshold=1e-7,
        #summary_save_path="butadien/log/idem",
        mini_batch_size=15
    )
    graph_idem = trainer.graph

    with trainer.graph.as_default():
        y = tf.placeholder(
                dtype="float32", 
                shape=[None, network.structure[-1]],
                name="y"
            )
        error_val = sess.run(
            AbsoluteError().function(network, y), 
            {
                network.input_tensor: dataset.testing[0],
                y: dataset.testing[1]
            }
        )
        
        error_idem = sess.run(
            trainer.cost_function.idempotency_error(network), 
            {network.input_tensor: dataset.testing[0]}
        )

    msg.info("Achieved absolute error:    {:0.3E}".format(error_val), 2)
    msg.info("Achieved idempotency error: {:0.3E}".format(error_idem), 2)

if __name__ == '__main__':
    main()