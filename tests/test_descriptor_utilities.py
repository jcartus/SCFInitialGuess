"""This module contains all tests for SCFInitialGuess.descriptor part of the
package. 

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import unittest
import numpy as np

from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.descriptors.utilities import \
    carthesian_to_spherical_coordinates


class TestCarthesianToSphericalCoordinates(unittest.TestCase):


    def testCase1(self):
        # (1,0,0)
        np.testing.assert_almost_equal(
            np.array(carthesian_to_spherical_coordinates(np.array([1, 0, 0]))),
            np.array([1.0, 0.0, np.pi/2])
        )

    def testCase2(self):
        # (0,1,0)
        np.testing.assert_almost_equal(
            np.array(carthesian_to_spherical_coordinates(np.array([0, 1, 0]))),
            np.array([1.0, np.pi/2, np.pi/2])
        )
    
    def testCase3(self):
        # (0,0,1)
        np.testing.assert_almost_equal(
            np.array(carthesian_to_spherical_coordinates(np.array([0, 0, 1]))),
            np.array([1.0, 0.0, 0.0])
        )

    def testCase4(self):
        # (1,1,0)
        np.testing.assert_almost_equal(
            np.array(carthesian_to_spherical_coordinates(np.array([1, 1, 0]))),
            np.array([np.sqrt(2), np.pi/4, np.pi/2])
        )

    def testCase5(self):
        # (1,1,1)
        np.testing.assert_almost_equal(
            np.array(carthesian_to_spherical_coordinates(np.array([
                1/np.sqrt(2), 
                1/np.sqrt(2), 
                1
            ]))),
            np.array([np.sqrt(2), np.pi/4, np.pi/4])
        )

    def testCase6(self):
        # (-1,0,0)
        np.testing.assert_almost_equal(
            np.array(carthesian_to_spherical_coordinates(np.array([-1, 0, 0]))),
            np.array([1, np.pi, np.pi/2])
        )

    def testCase7(self):
        # (0,-1,0)
        np.testing.assert_almost_equal(
            np.array(carthesian_to_spherical_coordinates(np.array([0, -1, 0]))),
            np.array([1, np.pi*3/2, np.pi/2])
        )

    def testCase8(self):
        # (0,-1,-1)
        np.testing.assert_almost_equal(
            np.array(carthesian_to_spherical_coordinates(np.array([0, -1, -1]))),
            np.array([np.sqrt(2), np.pi*3/2, np.pi*3/4])
        )


class TestRealSphericalHarmonics(unittest.TestCase):
    """Reference value from 
    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    """


    def test_s(self):
        from SCFInitialGuess.descriptors.utilities import real_spherical_harmonics

        l = 0
        m = 0

        np.testing.assert_almost_equal(
            complex(real_spherical_harmonics(
                np.random.rand(),
                np.random.rand(),
                l,
                m
            )),
            0.5 * np.sqrt(1/np.pi)
        )

    def test_p(self):
        from SCFInitialGuess.descriptors.utilities import \
            real_spherical_harmonics
        from scipy.special import sph_harm

        l = 1

        phi = np.random.normal(0, np.pi)
        theta = np.random.normal(0, np.pi / 2)

        #--- m = 1 -> px ---
        m = +1
        np.testing.assert_almost_equal(
            complex(real_spherical_harmonics(
                phi,
                theta,
                l,
                m
            )),
            np.sqrt(1 / 2) * \
                (sph_harm(-1, l, phi, theta) - sph_harm(+1, l, phi, theta)),
            decimal=7 

        )
        #---


        #--- m = 0 -> pz ---
        m = 0
        np.testing.assert_almost_equal(
            complex(real_spherical_harmonics(
                phi,
                theta,
                l,
                m
            )),
            sph_harm(0, l, phi, theta),
            decimal=7 
        )
        #---

        #--- m = -1 -> py ---
        m = -1
        np.testing.assert_almost_equal(
            complex(real_spherical_harmonics(
                phi,
                theta,
                l,
                m
            )),
            1j * np.sqrt(1 / 2) * \
                (sph_harm(-1, l, phi, theta) + sph_harm(+1, l, phi, theta)),
            decimal=7 
        )
        #---


    def test_d(self):
        from SCFInitialGuess.descriptors.utilities import \
            real_spherical_harmonics
        from scipy.special import sph_harm

        l = 2

        phi = np.random.normal(0, np.pi)
        theta = np.random.normal(0, np.pi / 2)

        #--- m = 2 ---
        m = +2
        np.testing.assert_almost_equal(
            complex(real_spherical_harmonics(
                phi,
                theta,
                l,
                m
            )),
            np.sqrt(1 / 2) * \
                (sph_harm(-2, l, phi, theta) + sph_harm(+2, l, phi, theta)),
            decimal=7 

        )
        #---

        #--- m = 0 ---
        m = 0
        np.testing.assert_almost_equal(
            complex(real_spherical_harmonics(
                phi,
                theta,
                l,
                m
            )),
            sph_harm(0, l, phi, theta),
            decimal=7 
        )
        #---

        #--- m = -1 ---
        m = -1
        np.testing.assert_almost_equal(
            complex(real_spherical_harmonics(
                phi,
                theta,
                l,
                m
            )),
            1j * np.sqrt(1 / 2) * \
                (sph_harm(-1, l, phi, theta) + sph_harm(+1, l, phi, theta)),
            decimal=7 

        )
        #---

if __name__ == '__main__':
    unittest.main()