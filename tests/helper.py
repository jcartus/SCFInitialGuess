"""This module contains helper funcitons and utilties for the
testing.
"""

import unittest

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