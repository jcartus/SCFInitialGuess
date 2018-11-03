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

class Wrapper(object):
    class TestAngularDescriptor(unittest.TestCase):


        def test_execution(self):
            
            descriptor = self._test_initialisation()

            r, phi, theta, values = self._test_forward(descriptor)

            self._test_backward(descriptor, values)


        def _test_initialisation(self):
            raise NotImplementedError("Abstract Test class")

        def _test_forward(self, descriptor):

            try:
                r, phi, theta = \
                    np.random.rand(1), np.random.rand(1), np.random.rand(1)
                values = descriptor.calculate_descriptor(r, phi, theta)
            except Exception as ex:
                self.fail("Calculation failed: " + str(ex))

            return r, phi, theta, values

        def _test_backward(self, descriptor, values):

            try:
                r = np.random.rand(30) * 4
                phi = np.random.rand(30) * 2 
                theta = np.random.rand(30) * 2
                activation = \
                    descriptor.calculate_inverse_descriptor(r, phi, theta, values)
            except Exception as ex:
                self.fail("Inverse calculation failed: " + str(ex))

            return r, phi, theta


class TestIndependentAngularDescriptor(Wrapper.TestAngularDescriptor):

    def setUp(self):

        # two gaussians 
        self.azimuthal_model_1 = ( 
            [1.0, 2.5],
            [0.5, 5],
            4
        )

        self.polar_model_1 = (
            [0.5, 1.0, 1.5],
            [1.0, 0.5, 5.0],
            2 
        )
    
    def _test_initialisation(self):
        from SCFInitialGuess.descriptors.coordinate_descriptors import \
            PeriodicGaussians, IndependentAngularDescriptor

        try:
            descriptor = IndependentAngularDescriptor(
                PeriodicGaussians(*self.azimuthal_model_1),
                PeriodicGaussians(*self.polar_model_1)
            )
            
        except Exception as ex:
            self.fail("Initialisation failed: " + str(ex))

        return descriptor

class TestSPHAngularDescriptor(Wrapper.TestAngularDescriptor):

    def setUp(self):

        self.l_max = 3
    
    def _test_initialisation(self):
        from SCFInitialGuess.descriptors.coordinate_descriptors import \
            SPHAngularDescriptor

        try:
            descriptor = SPHAngularDescriptor(self.l_max)
            
        except Exception as ex:
            self.fail("Initialisation failed: " + str(ex))

        return descriptor

    def _test_backward(self, descriptor, values):
        # Not available
        pass

    def test_values_are_real_and_not_nan(self):

        descriptor = self._test_initialisation()

        G = descriptor.calculate_descriptor(
            np.random.rand(1),
            np.random.rand(1),
            np.random.rand(1)
        )

        # chekc if result is real
        self.assertFalse(np.iscomplex(G).any())

        # check if result is not nan
        self.assertFalse(np.isnan(G).any())


if __name__ == '__main__':
    unittest.main()