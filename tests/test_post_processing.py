"""This module contains all tests for SCFInitialGuess.nn.tess_post_processing

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import unittest

import numpy as np

class TestRescale(unittest.TestCase):

    def test_rescale(self, p=None, s=None, dim = 3):
        from SCFInitialGuess.nn.post_processing import rescale

        

        # matrices
        if s is None:
            s = np.random.rand(dim, dim)

        if p is None:
            p = np.random.rand(dim, dim)

        # what tr(p@s) should result to
        n_electrons = np.random.rand(1) * 5
        
        # do transformation
        p_rescaled = rescale(p, s, n_electrons)

        np.testing.assert_almost_equal(
            np.trace(p_rescaled @ s), 
            n_electrons, 
            decimal=10
        )

    def test_rescale_batch(self):
        dim = 5
        P = np.random.rand(10, dim, dim)
        
        for p in P:
            self.test_rescale(p=p, dim=dim)


class TestRescaleDiag(TestRescale):

    def test_rescale(self, p=None, s=None, dim = 3):
        from SCFInitialGuess.nn.post_processing import rescale_diag       

        self.skipTest("Rescale_diag does not really make sense")

        # matrices
        if s is None:
            s = np.random.rand(dim, dim)

        if p is None:
            p = np.random.rand(dim, dim)

        # what tr(p@s) should result to
        n_electrons = np.random.rand(1) * 5
        
        # do transformation
        p_rescaled = rescale_diag(p, s, n_electrons)


        np.testing.assert_almost_equal(
            np.sum(np.trace(p_rescaled @ s)), 
            n_electrons, 
            decimal=10
        )

    def test_rescale_batch(self):
        dim = 5
        P = np.random.rand(10, dim, dim)
        
        for p in P:
            self.test_rescale(p=p, dim=dim)



if __name__ == '__main__':
    unittest.main()