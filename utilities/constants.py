"""This module stores or fetches value for contants, e.g. electronegativites

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

from utilities.usermessages import Messenger as msg
from os.path import normpath, isfile

def fetch_electronegativites_from_file(
    file=None
    ):
    
    if file is None:
        file = normpath("../Electronegativities.txt")

    if not isfile(file):
        raise IOError("File not found at " + file)

    msg.info("Fetching electronegativity values from file ...")

    with open(file, 'r') as f:
        # read lines from file and dropp commtes and column header
        lines = f.readlines()[5:]

    # get list of tuples (element symbol, value)
    chi = [(line[1], line[2]) for line in lines]

    return dict(chi)

electronegativites = fetch_electronegativites_from_file()

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