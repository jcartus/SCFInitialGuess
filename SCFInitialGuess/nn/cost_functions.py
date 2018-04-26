"""This module contains all cost function models.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from functools import reduce

import numpy as np
import tensorflow as tf

from SCFInitialGuess.utilities.dataset import extract_triu, reconstruct_from_triu

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

def make_matrix_batch_tensor(vector_batch, dim):
    """Turns a batch of flatted out matrices into a batch of actual matrices
    i.e. reshapes the vectors into dim x dim matrices again.
    Difference to make_matrix_batch in SCFInitialguess.utilities.dataset:
    this function works on tensorflow tensors
    """
    return tf.reshape(vector_batch, [-1, dim, dim])

def absolute_error(f, y):
    """Absolute error of two tensors of matching dimension (batchwise)"""

    #TODO fix this!.
    #if f.get_shape() != y.get_shape():
    #    raise ValueError("Dimensions missmatch in input tesors!")
    
    if len(f.get_shape()) == 2:
        return tf.reduce_mean(tf.abs(f - y), axis=[1])
    elif len(f.get_shape()) == 3:
        return tf.reduce_mean(tf.abs(f - y), axis=[1,2])
    else:
        raise ValueError(
            "Unknown shape encountered. Can only be of 2 or 3 dimensions!"
        )

def symmetry_error(f):
    """Symmetry error of square matrix shaped tensor (batchwise)"""
    return absolute_error(f, tf.matrix_transpose(f))

def idempotence_error(p, s):
    """Idempotence error between densty matrix p and the raw (i.e. NOT NORMED) 
    overlap matrix s! Both must be in QUADRATIC MATRIX shape.

    error = 2*p - p s p 
    """
    return absolute_error(2 * p, reduce(tf.matmul, (p, s, p)))

def predicted_occupance(p, s):
    """Number of electrons in system as predicted by 
        N = tr[p s]
    p and s must be in QUADRATIC MATRIX SHAPE. s must be raw 
    (i.e. not NORMALIZED!).
    """
    return tf.trace(tf.matmul(p, s))

