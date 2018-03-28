"""This module contains all cost function models.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np
import tensorflow as tf



class MSE(object):

    def function(self, network, y_placeholder):

        error = tf.losses.mean_squared_error(
            network.output_tensor,
            y_placeholder
        )

        cost = error
        
        tf.summary.scalar("error", error)
        
        return error

class AbsoluteError(object):

    def function(self, network, y_placeholder):

        error = tf.losses.absolute_difference(
            network.output_tensor,
            y_placeholder
        )

        cost = error
        
        tf.summary.scalar("error", error)
        
        return error


class RegularizedMSE(MSE):

    def __init__(self, alpha=1e-5):

        self.alpha = alpha

    def function(self, network, y_placeholder):

        error = super(RegularizedMSE, self).function(network, y_placeholder)

        regularisation = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(self.alpha),
            network.weights
        )

        cost = error + regularisation 

        tf.summary.scalar("weight_decay", regularisation)
        tf.summary.scalar("total_loss", cost)

        return cost
