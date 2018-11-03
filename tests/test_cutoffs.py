"""This module contains all tests for SCFInitialGuess.descriptors.cutoffs 
part of the package. 

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import unittest
import numpy as np



class TestBehler1(unittest.TestCase):

    def setUp(self):

        self.R_c = 5

    def test_inner_function(self):
        from SCFInitialGuess.descriptors.cutoffs import behler_cutoff_1

        #--- try execution ---
        t = np.linspace(0, self.R_c * 1.5, 200)
        try:
            y = behler_cutoff_1(t, self.R_c)
        except Exception as ex:
            self.fail("Executing cutoff function raised an error: " + str(ex))
        #---

        #--- try goes to zero after R_c ---
        self.assertEqual(
            behler_cutoff_1(self.R_c * 1.1, self.R_c),
            0.0
        )
        #---
        
    def test_highlevel_interfact(self):
        from SCFInitialGuess.descriptors.cutoffs import BehlerCutoff1

        co = BehlerCutoff1(self.R_c)


        G = np.random.rand(10)
        
        #--- after R_c weighting should be exactly 0 ---
        np.testing.assert_allclose(
            co.apply(
                G, 
                self.R_c * 1.2
            ),
            0 * G,
        )
        #---

        #--- at R_c/ 2 weighting should be exactly 0.5 ---
        np.testing.assert_allclose(
            co.apply(
                G, 
                self.R_c / 2
            ),
            0.5 * G,
        )
        #---
        

if __name__ == '__main__':
    unittest.main()
