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

    
    def test_normalisation(self):
        """Check if all values are < 1 (tests the underlying function)"""
        from SCFInitialGuess.descriptors.coordinate_descriptors \
            import periodic_gaussian
        

        period = np.random.rand() * 10
        eta = np.random.rand() * 5
        rs = np.random.rand() * period
        
        t = np.linspace(-period, 3* period, 1000)
        np.testing.assert_array_less(
            periodic_gaussian(t, rs, eta, period),
            1.0
        )

    def test_periodicity(self):
        """Check if gaussians are actually periodic 
        (tests the underlying function)"""
        from SCFInitialGuess.descriptors.coordinate_descriptors \
            import periodic_gaussian
        

        period = np.random.rand() * 10
        eta = np.random.rand() * 5
        rs = np.random.rand() * period
        
        t = np.linspace(-period, 3* period, 1000)
        np.testing.assert_allclose(
            periodic_gaussian(t, rs, eta, period),
            periodic_gaussian(t + period, rs, eta, period)
        )


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

    def test_array_forward(self):
        dim = 10

        descriptor = self._test_initialisation()

        G = descriptor.calculate_descriptor(
            np.random.rand(dim),
            np.random.rand(dim),
            np.random.rand(dim)
        )

        np.testing.assert_array_equal(
            (descriptor.number_of_descriptors, dim),
            G.shape
        )


class TestSphereSectionDescriptor(Wrapper.TestAngularDescriptor):

    def setUp(self):
        from SCFInitialGuess.descriptors.models import make_uniform
        
        
        self.radial_model = make_uniform(5)
        self.number_polar_sections = 2
        self.number_azimuthal_sections = 2
    
    def _test_initialisation(self):

        from SCFInitialGuess.descriptors.coordinate_descriptors import \
            SphereSectionDescriptor, Gaussians

        try:
            descriptor = SphereSectionDescriptor(
                self.number_polar_sections,
                self.number_azimuthal_sections,
                Gaussians(*self.radial_model)
            )
            
        except Exception as ex:
            self.fail("Initialisation failed: " + str(ex))

        return descriptor

        
    def test_with_fixed_radius(self):


        r = 1
        phi = np.random.rand(30) * 7 % 2 * np.pi
        theta = np.random.rand(30) * 4 % np.pi

        try:
            descriptor = self._test_initialisation()
            
            values = descriptor.calculate_descriptor(r, phi[0], theta[0])

            activation = \
                descriptor.calculate_inverse_descriptor(r, phi, theta, values)
        except Exception as ex:
            self.fail("Inverse calculation failed: " + str(ex))

        return r, phi, theta

    def test_with_fixed_theta(self):

        r = np.random.rand(30) * 10
        phi = np.random.rand(30) * 7 % 2 * np.pi
        theta = np.pi / 2

        R, Phi = np.meshgrid(r, phi)
        R = R.reshape(-1)
        Phi = Phi.reshape(-1)

        try:
            descriptor = self._test_initialisation()
            
            values = descriptor.calculate_descriptor(r[0], phi[0], theta)

            activation = \
                descriptor.calculate_inverse_descriptor(R, Phi, theta, values)
        except Exception as ex:
            self.fail("Inverse calculation failed: " + str(ex))

        return r, phi, theta

    def test_calculate_section(self):
        from SCFInitialGuess.descriptors.coordinate_descriptors import \
            SphereSectionDescriptor

        # context: period=4, number of sections = 4, lets test values and check
        # if they end up in the expected sections

        #--- section 1---
        self.assertEqual(
            0,
            SphereSectionDescriptor._calculate_section(0, 4, 4.0)
        )
        
        self.assertEqual(
            0,
            SphereSectionDescriptor._calculate_section(0.4, 4, 4.0)
        )
        #---

        #--- other sections ---
        self.assertEqual(
            1,
            SphereSectionDescriptor._calculate_section(0.5, 4, 4.0)
        )
        
        self.assertEqual(
            1,
            SphereSectionDescriptor._calculate_section(1.0, 4, 4.0)
        )

        self.assertEqual(
            2,
            SphereSectionDescriptor._calculate_section(2.0, 4, 4.0)
        )

        self.assertEqual(
            3,
            SphereSectionDescriptor._calculate_section(3.0, 4, 4.0)
        )
        #---

        #--- check periodicity ---
        self.assertEqual(
            0,
            SphereSectionDescriptor._calculate_section(3.5, 4, 4.0)
        )

        self.assertEqual(
            0,
            SphereSectionDescriptor._calculate_section(4.0, 4, 4.0)
        )

        self.assertEqual(
            0,
            SphereSectionDescriptor._calculate_section(4.4, 4, 4.0)
        )
        #---

        #--- negative values ---
        self.assertEqual(
            0,
            SphereSectionDescriptor._calculate_section(-0.4, 4, 4.0)
        )

        self.assertEqual(
            2,
            SphereSectionDescriptor._calculate_section(-2.0, 4, 4.0)
        )
        #---


if __name__ == '__main__':
    unittest.main()