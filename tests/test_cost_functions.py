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
from SCFInitialGuess.nn.training import MSE
from helper import NeuralNetworkMock


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