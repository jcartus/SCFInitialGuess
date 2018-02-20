"""This module stores or fetches value for contants, e.g. electronegativites

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from utilities.usermessages import Messenger as msg
from os.path import normpath, join, isfile, isdir


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
}
