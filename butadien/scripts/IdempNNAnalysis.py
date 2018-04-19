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

from PlainNNAnalysis import analysis, fetch_dataset

from functools import reduce

dim = 26
msg.print_level = 1

dataset = fetch_dataset()

class IdempotencyPenalty(MSE):
    def __init__(self, coupling=1e-4):
        
        self.coupling = coupling
        
    def function(self, network, y_placeholder):

        error = \
            super(IdempotencyPenalty, self).function(network, y_placeholder)

        penalty = self.idempotency_error(network) * self.coupling

        cost = error + penalty

        tf.summary.scalar("symmetry_penalty", penalty)
        tf.summary.scalar("total_loss", cost)

        return cost
    
    def idempotency_error(self, network):
        p = tf.reshape(network.output_tensor, [-1, dim, dim])
        s_raw = dataset.inverse_input_transform(network.input_tensor)
        s = tf.reshape(s_raw, [-1, dim, dim])
        

        lhs = tf.matmul(tf.matmul(p, s), p)
        rhs = 2 * p
        return tf.reduce_mean(tf.norm(tf.abs(lhs - rhs), axis=(1,2)) **2 )


def main():
    
    analysis(dataset, EluTrNNN([dim**2, 700, 700, dim**2]), IdempotencyPenalty(1e-2))

if __name__ == '__main__':
    main()