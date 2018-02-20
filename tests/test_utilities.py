"""This module contains all tests for SCFInitialGuess.utilites part of the
package. 

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import normpath, join

import unittest
from helper import AbstractTest

from SCFInitialGuess.utilities import Molecule, XYZFileReader


    


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

    def test_read_database(self):
        
        mols = XYZFileReader.read_database(self.data_folder)
        
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

if __name__ == '__main__':
    unittest.main()
