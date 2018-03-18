"""This module contains tests for nn subpackage.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from os import remove

import tensorflow as tf
import numpy as np
import unittest

from SCFInitialGuess.nn.networks import EluTrNNN, EluFixedValue

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

    def test_save_model(self):

        structures = [
            [5, 5],
            [5, 10, 5]
        ]

        for structure in structures:
            with tf.Session() as sess:
                network = EluTrNNN(structure)
                network.setup()
                sess.run(tf.global_variables_initializer())

                save_path = "tmp.npy"

                try:
                    network.export(sess, save_path)
                except Exception as ex:
                    self.fail("Export failed: " + str(ex))
                finally:
                    remove(save_path)

    def test_load_model(self):

        model_path = "tests/data/C_model.npy"


        try:
            sess = tf.Session()            

            #load model
            structure, weights, biases = np.load(model_path, encoding="latin1")

            network = EluFixedValue(structure, weights, biases)
            y = network.setup()
            x = network.input_tensor

            # try to run the network
            x_result = sess.run(
                y, 
                feed_dict={y: np.random.rand(1, structure[0])}
            )

        except Exception as ex:
            self.fail("Network could not be loaded: " + str(ex))

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

    def test_two_networks(self):    

        #--- setup tf and network
        sess = tf.Session()

        structure = [2, 2]
        network_1 = EluTrNNN(structure)
        y1 = network_1.setup()
        x1 = network_1.input_tensor
        sess.run(tf.global_variables_initializer())

        network_2 = EluTrNNN(structure)
        y2 = network_2.setup()
        x2 = network_2.input_tensor
        sess.run(tf.global_variables_initializer())        
        #---    
        
        #--- extract ----
        w_1_list = network_1.weights_values(sess)
        b_1_list = network_1.biases_values(sess)
        w_2_list = network_2.weights_values(sess)
        b_2_list = network_2.biases_values(sess)

        self.assertEqual(1, len(w_1_list))
        self.assertEqual(1, len(b_1_list))
        self.assertEqual(1, len(w_2_list))
        self.assertEqual(1, len(b_2_list))

        w_1 = w_1_list[0]
        b_1 = b_1_list[0]
        w_2 = w_2_list[0]
        b_2 = b_2_list[0]
        #---

        #--- assert if the calculation output the right values ---
        x_test = np.random.rand(1,2)
        y_test_1 = x_test.dot(w_1) + b_1
        y_test_2 = x_test.dot(w_2) + b_2

        np.testing.assert_almost_equal(
            y_test_1, 
            sess.run(y1, feed_dict={x1: x_test})
        )

        np.testing.assert_almost_equal(
            y_test_2, 
            sess.run(y2, feed_dict={x2: x_test})
        )
        #---