"""This module stores or fetches value for contants, e.g. electronegativites

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from utilities.usermessages import Messenger as msg
from os.path import normpath, join, isfile, isdir


class ConstantProvider(object):
    """This class will be a wrapper to all canstants and fetch them from the 
    right files / variables in this module.
    
    Attributes:
        - chi <dict<str, float>>: electronegativies of the atoms. Keys are atom 
        symbols. Alias: electronegativity
        - number_of_basis_functions <dict<str, int>>: the number of basis 
        functions for each atom in the 6-311++G(d,p) basis set. Key are the 
        atom symbols.
    """

    def __init__(self, data_folder):
        """Constructor:

        Args:
            data_folder <str>: the full path to the folder in which data files
            from which some of the constants are read are stored.
        """

        if not isdir(data_folder):
            raise ValueError("Data folder was not found at " + data_folder)

        self._data_folder = data_folder

    #--- electro negativity ---
    @property
    def chi(self):
        return fetch_electronegativites_from_file(
            join(self._data_folder, "Electronegativities.txt")
        )

    @property
    def electronegativites(self):
         return self.chi
    #---

    #--- basis fnctions ---
    @property
    def number_of_basis_functions(self):
        return number_of_basis_functions
    #---


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
    'Cl': 3.16
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
    "Br": 48
}