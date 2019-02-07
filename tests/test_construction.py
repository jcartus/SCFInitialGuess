"""This file contains tests for the parts of the construction module.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import unittest
import numpy as np

from SCFInitialGuess.construction.utilities import embed

class TestEmbedding(unittest.TestCase):

    def _embed_test_helper(self, x, y, mask, expected):
        
        x0 = x.copy()
        y0 = y.copy()
        mask0 = mask.copy()

        # make sure embedding yields the right result
        np.testing.assert_allclose(
            embed(x, y, mask),
            expected
        )

        # make sure original matrixces were not changed, e.g. via references
        np.testing.assert_allclose(x, x0)
        np.testing.assert_allclose(y, y0)
        np.testing.assert_allclose(mask, mask0)

    def test_embed_diagonal(self):

        x = np.zeros((5,5))
        y = np.ones((5,5))

        # replace elements on the diagonal
        mask = np.diag(np.ones(5)).astype("bool")

        # result should be a unit matrix
        self._embed_test_helper(x, y, mask, np.identity(5))

    def test_embed_all(self):

        x = np.zeros((5,5))
        y = np.ones(x.shape)

        # replace elements on the diagonal
        mask = np.ones(x.shape).astype("bool")

        # result should be exactly y
        self._embed_test_helper(x, y, mask, y)

class TestMakeMasks(unittest.TestCase):

    def setUp(self):
        from SCFInitialGuess.utilities.dataset import Molecule

        # our test molecule will by H2O
        self.mol = Molecule(
            ["O", "H", "H"],
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0]
            ]
        )

        self.mol.basis = "sto-3g"


    def test_center_mask(self):
        from SCFInitialGuess.construction.utilities import make_center_mask

        expected = np.zeros((7, 7))
        expected[0:5, 0:5] = 1
        expected[5:6, 5:6] = 1
        expected[6:7, 6:7] = 1
        expected = expected.astype("bool")

        actual = make_center_mask(self.mol)

        np.testing.assert_allclose(actual, expected)

    def test_hetero_mask(self):
        from SCFInitialGuess.construction.utilities import make_hetero_mask

        expected = np.zeros((7, 7))
        expected[0:5, 5:7] = 1
        expected[5:7, 0:5] = 1
        expected = expected.astype("bool")

        actual = make_hetero_mask(self.mol)

        np.testing.assert_allclose(actual, expected)

    def test_homo_mask(self):
        from SCFInitialGuess.construction.utilities import make_homo_mask

        expected = np.zeros((7, 7))
        expected[6:7, 5:6] = 1
        expected[5:6, 6:7] = 1
        expected = expected.astype("bool")

        actual = make_homo_mask(self.mol)

        np.testing.assert_allclose(actual, expected)

    def test_atomic_pair_maks_O_self_overlap(self):
        from SCFInitialGuess.construction.utilities import make_atom_pair_mask

        # test hydrogen
        expected = np.zeros((7, 7))
        expected[0:5, 0:5] = 1
        expected = expected.astype("bool")

        actual = make_atom_pair_mask(self.mol, 0, 0)

        np.testing.assert_allclose(actual, expected)

    def test_atomic_pair_maks_OH_overlap(self):
        from SCFInitialGuess.construction.utilities import make_atom_pair_mask

        # test hydrogen
        expected = np.zeros((7, 7))
        expected[5:6, 0:5] = 1
        expected = expected.astype("bool")

        actual = make_atom_pair_mask(self.mol, 1, 0)

        np.testing.assert_allclose(actual, expected)
        
    def test_atomic_pair_maks_HO_overlap(self):
        from SCFInitialGuess.construction.utilities import make_atom_pair_mask

        # test hydrogen
        expected = np.zeros((7, 7))
        expected[0:5, 5:6] = 1
        expected = expected.astype("bool")

        actual = make_atom_pair_mask(self.mol, 0, 1)

        np.testing.assert_allclose(actual, expected)

    def test_atomic_pair_maks_HH_homo_overlap(self):
        from SCFInitialGuess.construction.utilities import make_atom_pair_mask

        # test hydrogen
        expected = np.zeros((7, 7))
        expected[5:6, 6:7] = 1
        expected = expected.astype("bool")

        actual = make_atom_pair_mask(self.mol, 1, 2)

        np.testing.assert_allclose(actual, expected)

    def test_atomic_pair_maks_H_self_overlap(self):
        from SCFInitialGuess.construction.utilities import make_atom_pair_mask

        # test hydrogen
        expected = np.zeros((7, 7))
        expected[6:7, 6:7] = 1
        expected = expected.astype("bool")

        actual = make_atom_pair_mask(self.mol, 2, 2)

        np.testing.assert_allclose(actual, expected)

if __name__ == '__main__':
    unittest.main()