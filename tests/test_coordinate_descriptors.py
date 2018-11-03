"""This file contains tests for the backend-descriptor classes 
that use the direction of one 
atom to another in the molecule to describe the environment.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import unittest
import numpy as np

class TestGaussians(unittest.TestCase):

    def setUp(self):

        # two gaussians 
        self.model_1 = ( 
            [1.0, 2.5],
            [0.5, 5]
        )

    def test_execution(self):
        
        descriptor = self._test_initialisation()

        x, values = self._test_forward(descriptor)

        self._test_backward(descriptor, values, x)

    
    def _test_initialisation(self):
        from SCFInitialGuess.descriptors.coordinate_descriptors import Gaussians

        try:
            descriptor = Gaussians(*self.model_1)
        except Exception as ex:
            self.fail("Initialisation failed: " + str(ex))

        return descriptor

    def _test_forward(self, descriptor):

        try:
            x = np.random.rand(1)
            values = descriptor.calculate_descriptor(x)
        except Exception as ex:
            self.fail("Calculation failed: " + str(ex))

        return x, values

    def _test_backward(self, descriptor, values, x):

        try:
            t = np.linspace(0, 2 * x, 200)
            activation = descriptor.calculate_inverse_descriptor(t, values)
        except Exception as ex:
            self.fail("Inverse calculation failed: " + str(ex))


class TestPeriodicGaussians(TestGaussians):

    def setUp(self):
         self.model_1 = ( 
            [1.0, 2.5],
            [0.5, 5], 
            4
        )

    def _test_initialisation(self):
        from SCFInitialGuess.descriptors.coordinate_descriptors import \
            PeriodicGaussians

        try:
            descriptor = PeriodicGaussians(*self.model_1)
        except Exception as ex:
            self.fail("Initialisation failed: " + str(ex))

        return descriptor

if __name__ == '__main__':
    unittest.main()