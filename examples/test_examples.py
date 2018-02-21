"""This is a unit test that triggers all example scripts, to see if they
still run.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os import listdir
from os.path import isfile, splitext


import unittest, importlib

class TestExamples(unittest.TestCase):

    def setUp(self):
        self.examples = \
            list(map(lambda x: splitext(x)[0], listdir("./examples/")))

    def test_initialguess_benchmark(self):
        import initialguess_benchmark as mod

        try:
            mod.main()
        except Exception as ex:
            self.fail(msg="Example failed: " + str(ex))

    def test_train_and_save_network(self):
        import train_and_save_network as mod

        try:
            mod.main()
        except Exception as ex:
            self.fail(msg="Example failed: " + str(ex))
    
    
    def _test_all_examples(self):

        for example in self.examples:

            if isfile("examples/" + example + ".py"):
                mod = importlib.import_module("." + example, "examples")
            
                try:
                    print("Testing mod " + example)
                    mod.main()
                except Exception as ex:
                    self.fail(msg=example + " failed: " + str(ex))

if __name__ == '__main__':
    unittest.main()
