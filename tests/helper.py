"""This module contains helper funcitons and utilties for the
testing.
"""

import sys

import tensorflow as tf
import numpy as np

import unittest

from SCFInitialGuess.utilities.constants import \
    number_of_basis_functions as N_BASIS


def assert_python3(test):
    """Renders tests inconclusive if python2.x is current python version"""
    if sys.version_info[0] < 3:
        test.skipTest("Test is can not be run with python2.")

class AbstractTest(unittest.TestCase):

    def assert_geometries_match(self, expected, actual):

        for e, a in zip(expected, actual):

            # compare species
            self.assertEqual(e[0], a[0], msg="Species do not match.")

            # compare positions
            self.assertListEqual(
                e[1], 
                a[1], 
                msg="Positions do not exactly match."
            )

    def assert_geometries_almost_match(self, expected, actual, delta=1e-5):

        for e, a in zip(expected, actual):

            # compare species
            self.assertEqual(e[0], a[0], msg="Species do not match.")

            # compare positions
            self.assertListAlmostEqual(e[1], a[1], delta)

    def assertListAlmostEqual(self, expected, actual, delta=1e-5):
        
        self.assertEqual(
            len(expected), 
            len(actual), 
            msg="Length of lists did not match."
        )
        
        for i, (e, a) in enumerate(zip(expected, actual)):

            self.assertAlmostEqual(
                e, a, 
                delta=delta,
                msg="Element " + str(i) + " did not match to req. precision."
            )



class DatasetMock(object):
    
    def __init__(self, training=None, validation=None, testing=None):

        self.training = training
        self.validation = validation
        self.testing = testing

class NeuralNetworkMock(object):

    def __init__(self, structure, function=None):
        
        # not that it is good for anythin
        self.structure = structure

        # if no special mapping is stated just return output again
        if function is None:
            self.function = self.function_in_out
        else:
            self.function = function

        self.input_tensor = None

    @staticmethod
    def function_in_out(x):
        return x

    @property
    def output_tensor(self):
        if self._graph is None:
            self.setup()
        return self._graph

    def setup(self):

        # set up input placeholder    
        self.input_tensor = tf.placeholder(
                dtype="float32", 
                shape=[None, self.structure[0]],
                name="x"
            )        

        # put in simulated mapping
        self._graph = self.function(self.input_tensor)

        return self._graph


    def run(self, session, inputs):
        """Evaluate the neural network"""
        return session.run(self._graph, feed_dict={self.input_tensor: inputs})


def make_target_matrix_mock(mol):
    """This function creates a matrix of strings that can be used to 
    debug matrix block extractions. Its values e.g. for element (i,j) are 
    <spies_i><i>-<species_j><j>.
    """
    from SCFInitialGuess.construction.utilities import make_atom_pair_mask

    dim = mol.dim
    
    T = np.zeros((dim, dim), dtype="object")
    
    for i, atom_i in enumerate(mol.species):
        for j, atom_j in enumerate(mol.species):
            mask = make_atom_pair_mask(mol, i, j)
            
            m = atom_i + str(i) + "-" + atom_j + str(j)
            
            if N_BASIS[mol.basis][atom_i] * N_BASIS[mol.basis][atom_j] :
                T[mask] = m
            else:
                T[mask] = np.array(
                    [
                        [
                            m
                        ] * N_BASIS[mol.basis][atom_j]
                    ] * N_BASIS[mol.basis][atom_i],
                    dtype="object"
                )
    return T

class DescriptorMock(object):
    
    def __init__(self):
        self.number_of_descriptors = 1
        
    def calculate_atom_descriptor(self, index, mol, number_of_descriptors):
        return [index, mol.species[index]]

class DescriptorMockSum(DescriptorMock):
    
    def calculate_atom_descriptor(self, index, mol, number_of_descriptors):
        return [index]


class DataMock(object):
    """Mock for SCFInitialGuess.utilities.Data"""

    def __init__(self, molecules_test, T_test):
        
        self.molecules = [
            [],
            [],
            molecules_test
        ]

        self.T = [
            [],
            [],
            T_test
        ]
        