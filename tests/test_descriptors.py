"""This module contains all tests for SCFInitialGuess.descriptor part of the
package. 

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import unittest
import numpy as np

from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.descriptors.coordinate_descriptors import \
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


if __name__ == '__main__':
    unittest.main()