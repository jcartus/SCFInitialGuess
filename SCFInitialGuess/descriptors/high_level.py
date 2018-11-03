"""This file contains front-end descriptor classes that use the direction of one 
atom to another in the molecule to describe the environment.

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np
from SCFInitialGuess.descriptors.utilities import \
    carthesian_to_spherical_coordinates 

#-------------------------------------------------------------------------------
#   Top level descriptor classes
#-------------------------------------------------------------------------------


class AbstractCoordinateDescriptor(object):
    """This class takes a SCFInitialGuess.utilities.dataset.Molecule
    and calculates a descriptor vector that captures the atomic environment.
    """

    def __init__(self, 
            radial_descriptor, 
            angular_descriptor,
            cut_off
        ):      

        self.radial_descriptor = radial_descriptor
        self.angular_descriptor = angular_descriptor
        self.cut_off = cut_off

    @property
    def number_of_descriptors(self):
        return self.radial_descriptor.number_of_descriptors + \
            self.angular_descriptor.number_of_descriptors

    def calculate_descriptors_batch(self, molecules):
        """Calculate all descriptors for a batch of molecules"""
        
        descriptors = []
        for molecule in molecules:
            descriptors.append(self.calculate_all_descriptors(molecule))
        
        return np.array(descriptors).reshape(len(molecules), -1)

    def calculate_all_descriptors(self, molecule):
        """Calculates all descriptors for all atoms in the molecule"""
        
        geometry = molecule.geometry

        molecular_descriptor = []

        # calculate descriptor for each atom
        for i in range(molecule.number_of_atoms):
            
            molecular_descriptor.append(
                self.calculate_atom_descriptor(
                    i, 
                    molecule, 
                    self.number_of_descriptors
                )
            )
        return np.array(molecular_descriptor)

    def calculate_atom_descriptor(self, index_atom, molecule, number_of_descriptors):
        """Calculates the descriptor of the atom with index index_atom in the 
        molecule
        """

        geometry = molecule.geometry

        atomic_descriptor = np.zeros(number_of_descriptors)
            
        for j, geom_j in enumerate(geometry):
            if index_atom == j:
                continue
            
            atomic_descriptor += self.calculate_atomic_descriptor_contribution(
                    geometry[index_atom], 
                    geom_j
                )

        return atomic_descriptor




    def calculate_atomic_descriptor_contribution(self, geom_i, geom_j):
        """Calculates the contribution to the atomic descriptor of atom i
        for the atom j.
        """

        raise NotImplementedError(
            "AbstractCoordinateDescriptor is an abstract class!"
        )

    def apply_coordinate_descriptors(self, R):
        """This function is called by calculate_atomic_descriptor_contribution
        and avalues the repective coordinate descriptors of a vector R
        (probably the path from one atom to another in the molecule).
        """
        
        r, phi, theta = carthesian_to_spherical_coordinates(R)

        G = []

        G += list(self.radial_descriptor.calculate_descriptor(r))
        G += list(self.angular_descriptor.calculate_descriptor(r, phi, theta))
        
        return self.cut_off.apply(np.array(G), r, phi, theta)




class NonWeighted(AbstractCoordinateDescriptor):
    """This class calculates a desriptor of the vectors of the atoms
    to each other with gaussians in the radial direction.
    """

    def calculate_atomic_descriptor_contribution(self, geom_i, geom_j):
        """Calculates the contribution to the atomic descriptor of atom i
        for the atom j.
        """

        R = np.array(geom_j[1]) -  np.array(geom_i[1])

        return self.apply_coordinate_descriptors(R)



class AtomicNumberWeighted(AbstractCoordinateDescriptor):
    """This class calculates a desriptor of the vectors of the atoms
    to each other with gaussians in the radial direction.
    """

    def calculate_atomic_descriptor_contribution(self, geom_i, geom_j):
        """Calculates the contribution to the atomic descriptor of atom i
        for the atom j.
        """
        from SCFInitialGuess.utilities.constants import atomic_numbers as Z

        R = np.array(geom_i[1]) -  np.array(geom_j[1])

        return self.apply_coordinate_descriptors(R) * Z[geom_j[0]]

class ElectronegativityWeighted(AbstractCoordinateDescriptor):
    """This class calculates a desriptor of the vectors of the atoms
    to each other with gaussians in the radial direction.
    """

    def calculate_atomic_descriptor_contribution(self, geom_i, geom_j):
        """Calculates the contribution to the atomic descriptor of atom i
        for the atom j.
        """
        from SCFInitialGuess.utilities.constants import electronegativities as Chi

        R = np.array(geom_i[1]) -  np.array(geom_j[1])

        return self.apply_coordinate_descriptors(R) * Chi[geom_j[0]]