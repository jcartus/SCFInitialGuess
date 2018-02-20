"""This script will run all unit test for the project.
It was taken from 
https://github.com/ralf-meyer/NeuralNetworks/tests/run_all_tests.py.

Author:
    - Ralf Meyer, QCIEP, TU Graz
"""

import os
import sys
import unittest

if __name__ == '__main__':
    testsuite = unittest.TestLoader().discover(os.path.dirname(os.path.abspath(__file__)))
    test_runner = unittest.TextTestRunner(verbosity=1).run(testsuite)
    if len(test_runner.failures) > 0 or len(test_runner.errors) > 0:
        sys.exit(1)