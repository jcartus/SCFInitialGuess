"""This module contains all tests for SCFInitialGuess.utilites part of the
package. 

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import normpath, join

import numpy as np

import unittest
from helper import \
    AbstractTest, make_target_matrix_mock, DescriptorMock, DataMock, DescriptorMockSum

from SCFInitialGuess.utilities import Molecule, XYZFileReader
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.dataset import Dataset, QChemResultsReader
from SCFInitialGuess.utilities.dataset import extract_triu, reconstruct_from_triu
from SCFInitialGuess.utilities.constants import \
    number_of_basis_functions as N_BASIS


    
class TestQChemReader(AbstractTest):

    def test_read_file(self):

        test_file = normpath("tests/data/ethan.out")

        geometries = QChemResultsReader.read_file(test_file)

        self.assertEqual(201, len(list(geometries)))

    def test_read_folder(self):

        test_folder = normpath("tests/data/")

        geometries = QChemResultsReader.read_folder(test_folder)

        self.assertEqual(201, len(list(list(geometries)[0])))

class TestHelperFunctions(AbstractTest):

    def testExtractTriu(self):
        
        dim = 3

        A = np.arange(dim**2).reshape(dim, dim)

        v = [0, 1, 2, 4, 5, 8]

        np.testing.assert_array_equal(v, extract_triu(A, dim))


    def reconstruct_from_triu(self):

        dim = 3

        A = np.arange(dim**2).reshape(dim, dim)

        v = [0, 1, 2, 4, 5, 8]

        np.testing.assert_array_equal(A, reconstruct_from_triu(v, dim))

        


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

    def _check_molecules_in_test_folder(self, mols):

        # check names
        self.assertEqual("CO2", mols[0].full_name)
        self.assertEqual("QM9", mols[1].full_name)
        self.assertEqual("water", mols[2].full_name)

        # check species
        self.assertListEqual(["C", "O", "O"], mols[0].species)
        self.assertListEqual(["N", "C", "O"], mols[1].species)
        self.assertListEqual(["O", "H", "H"], mols[2].species)

        # check positions
        self.assertListEqual(self.reference_positions, mols[0].positions)
        self.assertListEqual(self.reference_positions, mols[2].positions)



    def test_read_folder(self):
        
        mols = XYZFileReader.read_folder(self.data_folder)

        self._check_molecules_in_test_folder(mols)
        
    def test_read_tree(self):

        mols = XYZFileReader.read_tree("tests")
        
        self._check_molecules_in_test_folder(mols)

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


class TestDatasetBlockExtractorCallBacks(unittest.TestCase):

    def setUp(self):
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

        # target matrix mock
        self.T = make_target_matrix_mock(self.mol)

        # standard descriptor
        self.descriptor = DescriptorMock()

    def _make_expected_symmetry_vector(self, pairs):
        
        G_expected = []

        for p in pairs:

            # check if center block of off-diag is required
            if isinstance(p, int):
                G_expected.append(
                    [p, self.mol.species[p]]
                )
            else:
                G_expected.append(
                    [p[0], self.mol.species[p[0]], p[1], self.mol.species[p[1]]]
                )

        return G_expected

    def _assert_results_ok(self, G, T, pair):
        """pair marks the expected regions. With descritor and target matrix mock
        the expected results are clear."""
        from SCFInitialGuess.construction.utilities import \
            make_atom_pair_mask

        # check symmetry vector G
        self.assertListEqual(G, self._make_expected_symmetry_vector(pair))

        # check matrix
        for (p, t) in zip(pair, T):
            if isinstance(p, int):
                mask = make_atom_pair_mask(self.mol, p, p)
                
                np.testing.assert_equal(
                    extract_triu(
                        self.T.copy()[mask], 
                        N_BASIS[self.mol.basis][self.mol.species[p]]
                    ),
                    t
                )
            else:

                # exptract block from LOWER triu (thus min/max)
                mask = make_atom_pair_mask(self.mol, np.max(p), np.min(p))
                np.testing.assert_equal(self.T[mask], t)

    

    def test_extract_center_blocks_H(self):
        from SCFInitialGuess.utilities.dataset import \
            extract_center_block_dataset_pairs

        species = "H"
        G, T = extract_center_block_dataset_pairs(
            self.descriptor, 
            [self.mol], 
            [self.T],
            species
        )

        pair = [1,2]

        self._assert_results_ok(G, T, pair)

    def test_extract_center_blocks_O(self):
        from SCFInitialGuess.utilities.dataset import \
            extract_center_block_dataset_pairs

        species = "O"
        G, T = extract_center_block_dataset_pairs(
            self.descriptor, 
            [self.mol], 
            [self.T],
            species
        )

        pair = [0,]

        self._assert_results_ok(G, T, pair)

    def test_extract_HOMO_blocks_H(self):
        from SCFInitialGuess.utilities.dataset import \
            extract_HOMO_block_dataset_pairs

        species = "H"
        G, T = extract_HOMO_block_dataset_pairs(
            self.descriptor, 
            [self.mol], 
            [self.T],
            species
        )

        pair = [[2,1]]

        self._assert_results_ok(G, T, pair)

    def test_extract_HOMO_blocks_O(self):
        from SCFInitialGuess.utilities.dataset import \
            extract_HOMO_block_dataset_pairs

        species = "O"
        G, T = extract_HOMO_block_dataset_pairs(
            self.descriptor, 
            [self.mol], 
            [self.T],
            species
        )

        self.assertEqual([], G)
        self.assertEqual([], T)
        
            
    def test_extract_Hetero_blocks_OH(self):
        from SCFInitialGuess.utilities.dataset import \
            extract_HETERO_block_dataset_pairs

        species = ["O", "H"]
        #random order of species
        ind = np.arange(2)
        #np.random.shuffle(ind)

        G, T = extract_HETERO_block_dataset_pairs(
            [self.descriptor, self.descriptor], 
            [self.mol], 
            [self.T],
            species
            #[species[ind[0]], species[ind[1]]]
        )

        pair = [[1, 0], [2, 0]]

        self._assert_results_ok(G, T, pair)
        

    def test_extract_Hetero_blocks_HO(self):
        from SCFInitialGuess.utilities.dataset import \
            extract_HETERO_block_dataset_pairs

        species = ["H", "O"]
        #random order of species
        ind = np.arange(2)
        #np.random.shuffle(ind)

        G, T = extract_HETERO_block_dataset_pairs(
            [self.descriptor, self.descriptor], 
            [self.mol], 
            [self.T],
            species
            #[species[ind[0]], species[ind[1]]]
        )

        pair = [[0, 1], [ 0, 2]]

        self._assert_results_ok(G, T, pair)
            
    
class TestMakeBlockDataset(unittest.TestCase):

    def setUp(self):

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

        self.T = np.ones((7,7))

        self.descriptor = DescriptorMockSum()

    def test_center_works_like_deprecated_H(self):
        """Make sure generic function works like deprecated special function 
        for center blocks"""
        from SCFInitialGuess.utilities.dataset import \
            make_center_block_dataset, make_block_dataset, \
                extract_center_block_dataset_pairs


        data = DataMock(
            molecules_test=self.mol,
            T_test=self.T
        )

        expected = make_center_block_dataset(
            self.descriptor,
            [[], [], [self.mol]],
            [[], [], [self.T]],
            "H"
        )

        actual = make_block_dataset(
            self.descriptor,
            [[], [], [self.mol]],
            [[], [], [self.T]],
            "H",
            extract_center_block_dataset_pairs
        )

        #--- testing ---
        np.testing.assert_allclose(
            actual.testing[0],
            expected.testing[0]
        )
        np.testing.assert_allclose(
            actual.testing[1],
            expected.testing[1]
        )
        #---

        #--- validation ---
        np.testing.assert_allclose(
            actual.validation[0],
            expected.validation[0]
        )
        np.testing.assert_allclose(
            actual.validation[1],
            expected.validation[1]
        )
        #---

        #--- training ---
        np.testing.assert_allclose(
            actual.training[0],
            expected.training[0]
        )
        np.testing.assert_allclose(
            actual.training[1],
            expected.training[1]
        )
        #---


if __name__ == '__main__':
    unittest.main()
