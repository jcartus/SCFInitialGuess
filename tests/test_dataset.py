"""This module contains all tests for SCFInitialGuess.utilites part of the
package. 

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import normpath, join

import numpy as np

import unittest
from helper import AbstractTest

from SCFInitialGuess.utilities import Molecule, XYZFileReader
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.dataset import Dataset, QChemResultsReader


    
class TestQChemReader(AbstractTest):

    def test_read_file(self):

        test_file = normpath("tests/data/ethan.out")

        geometries = QChemResultsReader.read_file(test_file)

        self.assertEqual(201, len(list(geometries)))

    def test_read_folder(self):

        test_folder = normpath("tests/data/")

        geometries = QChemResultsReader.read_folder(test_folder)

        self.assertEqual(201, len(list(list(geometries)[0])))



class TestXYZFileReader(AbstractTest):
    """This class tests XYZFileReader. """

    def setUp(self):
        self.data_folder = normpath("tests/data")

        # all molecules read in will have this geometry (with variing species)
        self.reference_positions = [
            [0.0, 0.0,  0.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0,  2.0]
        ]

    def test_read_molecule_from_file(self):
        
        mol = XYZFileReader.read_molecule_from_file(
            file_name=join(self.data_folder, "water.xyz"),
            name="TestName"
        )

        # check species
        self.assertListEqual(["O", "H", "H"], mol.species)

        # check positions
        self.assertListEqual(self.reference_positions, mol.positions)

        #check Name
        self.assertEqual("TestName", mol.full_name)

    def test_read_folder(self):
        
        mols = XYZFileReader.read_folder(self.data_folder)

        # check names
        self.assertEqual("CO2", mols[0].full_name)
        self.assertEqual("water", mols[1].full_name)

        # check species
        self.assertListEqual(["C", "O", "O"], mols[0].species)
        self.assertListEqual(["O", "H", "H"], mols[1].species)

        # check positions
        self.assertListEqual(self.reference_positions, mols[0].positions)
        self.assertListEqual(self.reference_positions, mols[1].positions)

    def test_read_tree(self):

        mols = XYZFileReader.read_tree("tests")

        # check names
        self.assertEqual("CO2", mols[0].full_name)
        self.assertEqual("water", mols[1].full_name)

        # check species
        self.assertListEqual(["C", "O", "O"], mols[0].species)
        self.assertListEqual(["O", "H", "H"], mols[1].species)

        # check positions
        self.assertListEqual(self.reference_positions, mols[0].positions)
        self.assertListEqual(self.reference_positions, mols[1].positions)

class TestMolecule(AbstractTest):

    def setUp(self):
        self.reference_species = ["C", "O", "O"]

        self.reference_positions = [
            [0.0, 0.0,  0.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0,  2.0]
        ]


    def test_geometry(self):
        mol = Molecule(self.reference_species, self.reference_positions)

        self.assert_geometries_match(
            zip(self.reference_species, self.reference_positions),
            mol.geometry
        )

class TestDataset(AbstractTest):
    
    def setUp(self):

        self.tolerance = 1e-1
        msg.print_level = 1

    def test_setup(self):
        nsamples = 200

        # create x/y with matching values
        x = np.arange(nsamples)
        y = np.arange(nsamples)

        # dataset should be splitted pure traing 100, validation 50, test 50
        candidate = Dataset(
            x, y, 
            split_test=0.25, 
            split_validation=1.0/3.0,
            normalize_input=False    
        )

        # check if splitting was correct
        self.assertEqual(50, len(candidate.testing[0]))
        self.assertEqual(50, len(candidate.testing[1]))
        self.assertEqual(50, len(candidate.validation[0]))
        self.assertEqual(50, len(candidate.validation[1]))
        self.assertEqual(100, len(candidate.training[0]))
        self.assertEqual(100, len(candidate.training[1]))

        # check if x-y pairs are still correct
        np.testing.assert_array_equal(*candidate.testing)
        np.testing.assert_array_equal(*candidate.validation)
        np.testing.assert_array_equal(*candidate.training)


        

    def test_normalisation(self):
 
        dim = 5
        mu = 3
        sigma = 2
        nsamples = 1000

        x = np.random.randn(nsamples, dim) * sigma + mu

        #--- check normlisation with calculated params ---
        x_norm = Dataset.normalize(x)[0]

        self.assertAlmostEqual(0, np.mean(x_norm), delta=self.tolerance)
        self.assertAlmostEqual(1, np.std(x_norm), delta=self.tolerance)
        #---

        #--- check normalisation with given params ---
        x_norm_given_params = Dataset.normalize(x, mean=mu, std=sigma)[0]
        self.assertAlmostEqual(
            0, 
            np.mean(x_norm_given_params), 
            delta=self.tolerance
        )
        self.assertAlmostEqual(
            1, 
            np.std(x_norm_given_params), 
            delta=self.tolerance
        )
        #---


if __name__ == '__main__':
    unittest.main()
