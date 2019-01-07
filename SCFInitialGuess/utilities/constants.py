"""This module stores or fetches value for contants, e.g. electronegativites

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from .usermessages import Messenger as msg
from os.path import normpath, join, isfile, isdir


# valence electrons
valence_electrons = {
    'H':  1,
    'Li': 1,
    'Na': 1,
    'Be': 2,
    'Mg': 2,
    'B':  3,
    'Al': 3,
    'C':  4,
    'Si': 4,
    'N':  5,
    'P':  5,
    'O':  6,
    'S':  6,
    'F':  7,
    'Cl': 7,
    'I':  7,
    'Br': 7,
    'He': 8,
    'Ne': 8,
    'Ar': 8
}


# atomic number of the lements in the periodic table.
atomic_numbers = {
    'H': 1,
    'He': 2,
    'Li': 3, 
    'Be': 4,
    'B': 5, 
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14, 
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
    'Br': 35,
    'I': 53
}

# electronegativity values (pauling scala)
#https://en.wikipedia.org/wiki/Electronegativities_of_the_elements_(data_page)
electronegativities = {
    'H': 2.2,
    'Li': 0.98,
    'B': 2.04,
    'C': 2.55,
    'N': 3.04,
    'O': 3.44,
    'F': 3.98,
    'Na': 0.93,
    'Mg': 1.31,
    'Al': 1.61,
    'Si': 1.9,
    'P': 2.19,
    'S': 2.58,
    'Cl': 3.16,
    'Br': 2.96,
    'I': 2.66,
    'Ar': None
}


# the number of basis functions in 6-311++G** for a specified element
number_of_basis_functions = {
    "6-311++g**": {
        "H": 7,
        "N": 22,
        "F": 22,
        "O": 22,
        "S": 30,
        "C": 22,
        "Si": 30,
        "Cl": 30,
        "P": 30,
        "Br": 48,
        "I": 62
    },
    "sto-3g": {
        "H": 1,
        "C": 5,
    },
    "3-21g*": {
        "H": 2,
        "C": 9,
    },
    "3-21++g*": {
        "H": 3,
        "C": 13
    },
    "6-31g**": {
        "H": 5,
        "C": 14,
    }
}
