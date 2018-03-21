"""This module contains all tests for SCFInitialGuess.nn.training

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os import remove, mkdir
from os.path import isdir

import tensorflow as tf
import numpy as np
import unittest

from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.dataset import Dataset
from SCFInitialGuess.nn.networks import EluTrNNN
from SCFInitialGuess.nn.training import train_network, Trainer

class TestTrainer(unittest.TestCase):

    def setUp(self):
        
        msg.print_level = 0

        self.structure = [1, 4, 1]
        self.nsamples = 100

        x = np.random.rand(self.nsamples, self.structure[0]) * 10
        y = np.sum(x**2, axis=1)

        self.dataset = Dataset(
            x.reshape(self.nsamples, self.structure[0]), 
            y.reshape(self.nsamples, self.structure[-1])
        )


    def test_training_default_options(self):

        try:
            trainer = Trainer(EluTrNNN(self.structure))
        except:
            self.fail("Instantiation of trainer failed")
        
        try:
            trainer.setup()
        except:
            self.fail("Trainer setup failed")
        
        try:
            trainer.train(self.dataset)
        except:
            self.fail("Trainer with trainer failed.")

    def test_training_w_logging(self):

        save_dir = "tests/tmp_log/" 

        if not isdir(save_dir):
            mkdir(save_dir)

        try:
            try:
                trainer = Trainer(EluTrNNN(self.structure))
            except:
                self.fail("Instantiation of trainer failed")
        

            try:
                trainer.setup()
            except:
                self.fail("Trainer setup failed")
            
            try:
                trainer.train(
                    self.dataset,
                    summary_save_path=save_dir
                )
            except:
                self.fail("Trainer with trainer failed.")
        finally:
            rmtree(save_dir)

class TestTraining(unittest.TestCase):

    def setUp(self):

        msg.print_level = 0

        self.input_dim = 5
        self.output_dim = 5
        nsamples = 100
        
        x = np.linspace(-2, 2, nsamples * self.input_dim)
        y = np.sin(x)


        self.dataset = Dataset(
            x.reshape(nsamples, self.input_dim), 
            y.reshape(nsamples, self.output_dim)
        )

    def test_train_network(self):
        tf.reset_default_graph()

        structure = [self.input_dim, 10, self.output_dim]
        network = EluTrNNN(structure)

        try:
            _, sess = train_network(
                network,
                self.dataset,
                convergence_threshold=1e-1
            )
            sess.close()
        except Exception as ex:
            self.fail("Traning failed: " + str(ex))

    def test_train_network_w_logging(self):

        tf.reset_default_graph()

        structure = [self.input_dim, 10, self.output_dim]
        network = EluTrNNN(structure)

        save_dir = "tests/tmp_log/"
        
        if not isdir(save_dir):
            mkdir(save_dir)

        try:
            _, sess = train_network(
                network,
                self.dataset,
                summary_save_path=save_dir
            )   
            sess.close()
        except Exception as ex:
            self.fail("Training failed!")
        finally:
            rmtree(save_dir)

if __name__ == '__main__':
    unittest.main()