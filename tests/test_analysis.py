"""This file contains tests for the module in SCFInitialguess.utilities.dataset

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np
import unittest

from helper import DatasetMock

class TestErrorMeasurements(unittest.TestCase):

    def test_absolute_error(self):
        from SCFInitialGuess.utilities.analysis import measure_absolute_error
        
        
        lhs = np.array(
            [
                np.arange(2),
                np.ones(2)
            ]
        )

        rhs = np.array(
            [
                np.arange(2),
                np.zeros(2)
            ]
        )

        result = list(measure_absolute_error(
            lhs,
            dataset=DatasetMock(testing=(None, rhs))
        ))

        np.testing.assert_array_equal([0.0, 1.0], result)

    def test_symmetry_error(self):
        from SCFInitialGuess.utilities.analysis import measure_symmetry_error

        dataset = []
        b = np.ones((2, 2))
        dataset.append(b) # error in first batch should be 0

        a = np.ones((2, 2))
        a[1][0] = 3 # average errror in first batch should be 1
        dataset.append(a) 
        dataset = np.array(dataset)

        result = list(measure_symmetry_error(dataset))

        np.testing.assert_array_equal([0.0, 1.0], result)

    def test_idempotence_error(self):
        from SCFInitialGuess.utilities.analysis import measure_idempotence_error

        
        # this matrix is actually idempotent i.e. p = p*p
        p = np.array([[2,-2,-4],[-1,3,4],[1,-2,-3]])

        # thus p = p*E*p. To fulfil: p*2 = p*s*p, s must be 2*E
        s = np.eye(len(p)) * 2

        result = list(measure_idempotence_error(
            [p],
            [s]
        ))

        np.testing.assert_array_equal([0], result)

if __name__ == '__main__':
    unittest.main()