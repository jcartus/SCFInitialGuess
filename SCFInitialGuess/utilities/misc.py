"""In this module all kinds of helper routines, that do not belong to any of the other topics will be stored. They can be imported from utilities directly.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from os import getcwd, chdir
from os.path import expanduser

class cd:
    """Context manager for changing the current working directory
    Got this snippet from stackoverflow:
    https://stackoverflow.com/questions/431684/how-do-i-cd-in-python
    """
    def __init__(self, newPath):
        self.newPath = expanduser(newPath)

    def __enter__(self):
        self.savedPath = getcwd()
        chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        chdir(self.savedPath)