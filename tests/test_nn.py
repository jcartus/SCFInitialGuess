"""This module contains tests for nn subpackage.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import tensorflow as tf
import numpy as np
import unittest

from SCFInitialGuess.nn.networks import EluTrNNN

class TestNetworks(unittest.TestCase):

    def test_EluTrNNN_setup(self):

        sess = tf.Session()

        structures = [
            [5, 5],
            [5, 15, 5],
            [5, 20, 15, 5]
        ]        

        for structure in structures:
            self._test_network_setup(sess, structure, EluTrNNN(structure))

    def _test_network_setup(self, sess, structure, network):

        network.setup()

        sess.run(tf.global_variables_initializer())

        number_of_layers = len(structure) - 1
        weights = network.weights_values(sess)
        biases = network.biases_values(sess)

        #--- check if number of layers fits ---
        self.assertEqual(number_of_layers, len(network.weights))
        self.assertEqual(number_of_layers, len(network.biases))
        self.assertEqual(number_of_layers, len(weights))
        self.assertEqual(number_of_layers, len(biases))
        #---

        for layer in range(0, number_of_layers):
            dim_in = structure[layer]
            dim_out = structure[layer + 1]

            self.assertListEqual([dim_in, dim_out], list(weights[layer].shape))
            self.assertEqual(dim_out, biases[layer].shape[0])




    def test_linear_EluTrNNN(self):

        #--- setup tf and network
        sess = tf.Session()

        structure = [2, 2]
        network = EluTrNNN(structure)
        y = network.setup()
        x = network.input_tensor

        sess.run(tf.global_variables_initializer())
        #---    


        #--- extract ----
        w_list = network.weights_values(sess)
        b_list = network.biases_values(sess)

        self.assertEqual(1, len(w_list))
        self.assertEqual(1, len(b_list))

        w = w_list[0]
        b = b_list[0]
        #---

        x_test = np.random.rand(1,2)
        y_test = x_test.dot(w) + b

        np.testing.assert_almost_equal(
            y_test, 
            sess.run(y, feed_dict={x: x_test})
        )



        
