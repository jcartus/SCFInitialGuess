""" This module contains all cost functions.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from os import remove, mkdir
from os.path import isdir
from shutil import rmtree

import tensorflow as tf
import numpy as np
import unittest

from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.nn.cost_functions import MSE
from helper import NeuralNetworkMock


class TestHelperFunctions(unittest.TestCase):

    def setUp(self):

        msg.print_level = 1
        tf.reset_default_graph()

    def test_absolute_error(self):
        from SCFInitialGuess.nn.cost_functions import absolute_error

        tf.reset_default_graph()

        #--- prep matrix with asymmetric matix ---
        dataset_lhs = np.array(
            [
                np.arange(2),
                np.ones(2)
            ]
        )

        dataset_rhs = np.array(
            [
                np.arange(2),
                np.zeros(2)
            ]
        )
        #---

        f = tf.placeholder(tf.float64, shape=(None, 2))
        g = tf.placeholder(tf.float64, shape=(None, 2))
        error = absolute_error(f, g)

        sess = tf.Session()
        result = sess.run(error, {f: dataset_lhs, g: dataset_rhs})

        np.testing.assert_array_equal(np.array([0.0, 1.0]), result)


    def test_symmetry_error(self):    
        from SCFInitialGuess.nn.cost_functions import symmetry_error

        tf.reset_default_graph()

        #--- prep matrix with asymmetric matix ---
        dataset = []
        b = np.ones((2, 2))
        dataset.append(b) # error in first batch should be 0

        a = np.ones((2, 2))
        a[1][0] = 3 # average errror in first batch should be 1
        dataset.append(a) 
        dataset = np.array(dataset)
        #---
        
        f = tf.placeholder(tf.float64, shape=(None, 2, 2))
        error = symmetry_error(f)

        sess = tf.Session()
        result = sess.run(error, {f: dataset})

        np.testing.assert_array_equal(np.array([0.0, 1.0]), result)
        

class TestErrorFunctions(unittest.TestCase):

    def setUp(self):
        self.structure = [1, 1]
        msg.print_level = 1

        tf.reset_default_graph()

class TestMSE(TestErrorFunctions):

    def test_MSE(self):
        

        expected = 4
        x_values = np.random.rand(100, self.structure[0])
        y_values = x_values + np.sqrt(expected)

        mse = MSE()

        with tf.Session() as sess:
            network = NeuralNetworkMock(self.structure)
            network.setup()
            x = network.input_tensor
            f = network.output_tensor
            y = tf.placeholder(
                dtype="float32", 
                shape=[None, network.structure[-1]],
                name="y"
            )

            error = mse.function(network, y)

            actual = sess.run(
                error,
                feed_dict={x: x_values, y: y_values}
            )
            
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()