"""This module contains functions to calculate descriptor values for 
butadien molecules. I snatched most of it from Daniel Lukic's (Github: danlukic)
bachelor prject.

Author:
    Daniel Lukic, TU Graz
    vlt Ralf Meyer, QCIEP, TU Graz
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np

def fetch_descriptor_values(molecules):
    
    geometries = [mol.positions for mol in molecules]
    
    descriptor_values = []
    for geometry in geometries:
        descriptor_values.append(xyz2zmat(geometry))
        
    return np.array(descriptor_values)
    
def fetch_descriptor_values_over_time(molecules):
    
    geometries = [mol.positions for mol in molecules]
    
    distances = []
    angles = []
    dihedrals = []
    for geometry in geometries:
        descriptors = xyz2zmatSeparate(geometry)
        distances.append(descriptors[0])
        angles.append(descriptors[1])
        dihedrals.append(descriptors[2])

        
    return np.array(distances), np.array(angles), np.array(dihedrals)


# Some helper functions to transform the xyz geometries
def xyz2zmat(geometry):
    # Calculates 24 internal coordinates from the xyz geometry
    internal_coordinates = []
    internal_coordinates.append(calc_distance(geometry[0], geometry[1])) # distance of C1 to C2
    internal_coordinates.append(calc_distance(geometry[1], geometry[2])) # distance of C2 to C3
    internal_coordinates.append(calc_angle(geometry[0], geometry[1], geometry[2])) # angle between C1, C2 and C3
    internal_coordinates.append(calc_distance(geometry[2], geometry[3])) # distance of C3 to C4
    internal_coordinates.append(calc_angle(geometry[1], geometry[2], geometry[3])) # angle between C2, C3 and C4
    internal_coordinates.append(calc_dihedral(geometry[0], geometry[1], geometry[2], geometry[3])) # dihedral between C1, C2, C3 and C4

    internal_coordinates.append(calc_distance(geometry[0], geometry[4])) # distance of C1 to H1
    internal_coordinates.append(calc_angle(geometry[4], geometry[0], geometry[1])) # angle between H1, C1 and C2
    internal_coordinates.append(calc_dihedral(geometry[4], geometry[0], geometry[1], geometry[2])) # dihedral between H1, C1, C2 and C3

    internal_coordinates.append(calc_distance(geometry[0], geometry[5])) # distance of C1 to H2
    internal_coordinates.append(calc_angle(geometry[5], geometry[0], geometry[1])) # angle between H2, C1 and C2
    internal_coordinates.append(calc_dihedral(geometry[5], geometry[0], geometry[1], geometry[2])) # dihedral between H2, C1, C2 and C3

    internal_coordinates.append(calc_distance(geometry[1], geometry[6])) # distance of C2 to H3
    internal_coordinates.append(calc_angle(geometry[6], geometry[1], geometry[0])) # angle between H3, C2 and C1
    internal_coordinates.append(calc_dihedral(geometry[6], geometry[1], geometry[2], geometry[3])) # dihedral between H3, C2, C3 and C4

    internal_coordinates.append(calc_distance(geometry[2], geometry[7])) # distance of C3 to H4
    internal_coordinates.append(calc_angle(geometry[7], geometry[2], geometry[3])) # angle between H4, C3 and C4
    internal_coordinates.append(calc_dihedral(geometry[7], geometry[3], geometry[2], geometry[1])) # dihedral between H3, C3, C2 and C1

    internal_coordinates.append(calc_distance(geometry[3], geometry[8])) # distance of C4 to H5
    internal_coordinates.append(calc_angle(geometry[8], geometry[3], geometry[2])) # angle between H5, C4 and C3
    internal_coordinates.append(calc_dihedral(geometry[8], geometry[3], geometry[2], geometry[1])) # dihedral between H5, C4, C3 and C2

    internal_coordinates.append(calc_distance(geometry[3], geometry[9])) # distance of C4 to H6
    internal_coordinates.append(calc_angle(geometry[9], geometry[3], geometry[2])) # angle between H6, C4 and C3
    internal_coordinates.append(calc_dihedral(geometry[9], geometry[3], geometry[2], geometry[1])) # dihedral between H6, C4, C3 and C2
    return internal_coordinates

def xyz2zmatSeparate(geometry):
    # Calculates 24 internal coordinates from the xyz geometry

    distances = []
    angles = []
    dihedrals = []

    internal_coordinates = []
    distances.append(calc_distance(geometry[0], geometry[1])) # distance of C1 to C2
    distances.append(calc_distance(geometry[1], geometry[2])) # distance of C2 to C3
    angles.append(calc_angle(geometry[0], geometry[1], geometry[2])) # angle between C1, C2 and C3
    distances.append(calc_distance(geometry[2], geometry[3])) # distance of C3 to C4
    angles.append(calc_angle(geometry[1], geometry[2], geometry[3])) # angle between C2, C3 and C4
    dihedrals.append(calc_dihedral(geometry[0], geometry[1], geometry[2], geometry[3])) # dihedral between C1, C2, C3 and C4

    distances.append(calc_distance(geometry[0], geometry[4])) # distance of C1 to H1
    angles.append(calc_angle(geometry[4], geometry[0], geometry[1])) # angle between H1, C1 and C2
    dihedrals.append(calc_dihedral(geometry[4], geometry[0], geometry[1], geometry[2])) # dihedral between H1, C1, C2 and C3

    distances.append(calc_distance(geometry[0], geometry[5])) # distance of C1 to H2
    angles.append(calc_angle(geometry[5], geometry[0], geometry[1])) # angle between H2, C1 and C2
    dihedrals.append(calc_dihedral(geometry[5], geometry[0], geometry[1], geometry[2])) # dihedral between H2, C1, C2 and C3

    distances.append(calc_distance(geometry[1], geometry[6])) # distance of C2 to H3
    angles.append(calc_angle(geometry[6], geometry[1], geometry[0])) # angle between H3, C2 and C1
    dihedrals.append(calc_dihedral(geometry[6], geometry[1], geometry[2], geometry[3])) # dihedral between H3, C2, C3 and C4

    distances.append(calc_distance(geometry[2], geometry[7])) # distance of C3 to H4
    angles.append(calc_angle(geometry[7], geometry[2], geometry[3])) # angle between H4, C3 and C4
    dihedrals.append(calc_dihedral(geometry[7], geometry[3], geometry[2], geometry[1])) # dihedral between H3, C3, C2 and C1

    distances.append(calc_distance(geometry[3], geometry[8])) # distance of C4 to H5
    angles.append(calc_angle(geometry[8], geometry[3], geometry[2])) # angle between H5, C4 and C3
    dihedrals.append(calc_dihedral(geometry[8], geometry[3], geometry[2], geometry[1])) # dihedral between H5, C4, C3 and C2

    distances.append(calc_distance(geometry[3], geometry[9])) # distance of C4 to H6
    angles.append(calc_angle(geometry[9], geometry[3], geometry[2])) # angle between H6, C4 and C3
    dihedrals.append(calc_dihedral(geometry[9], geometry[3], geometry[2], geometry[1])) # dihedral between H6, C4, C3 and C2
    return distances, angles, dihedrals

  
def calc_distance(a1,a2):
    # Calculates the distance between two atoms a1 and a2 
    
    a1 = np.array(a1)
    a2 = np.array(a2)
    
    return np.linalg.norm(a2-a1)

def calc_angle(a1,a2,a3):
    # Calculates the angle inbetween the vectors a1-a2 and a3-a2
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    
    r1 = a1-a2
    r2 = a3-a2
    #return np.arccos(np.dot(r1,r2)/(np.linalg.norm(r1)*np.linalg.norm(r2)))
    return np.arctan2(np.linalg.norm(np.cross(r1,r2)),np.dot(r1,r2))

def calc_dihedral(a1,a2,a3,a4):
    # Calculates the dihedral angle of a chain of four atoms
    
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    a4 = np.array(a4)
    
    r1 = a1-a2
    r2 = a3-a2
    r3 = a4-a3
    n1 = np.cross(r1,r2)
    n2 = np.cross(r2,r3)
    #return np.arccos(np.dot(n1,n2)/(np.linalg.norm(n1)*np.linalg.norm(n2)))
    return np.arctan2(np.dot(np.cross(n1,n2),r2/np.linalg.norm(r2)),np.dot(n1,n2))