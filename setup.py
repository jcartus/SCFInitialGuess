"""This script installs the package.abs

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from distutils.core import setup

name = 'SCFInitialGuess'

setup(
    name=name,
    version='0.0',
    description='A neural network approach to create initial guesses for scf calculations',
    author='Johannes Cartus',
    packages=[name, name + '.utilities'],
    package_dir={
        name: 'SCFInitialGuess',
        name + '.utilities': 'SCFInitialGuess/utilities'
    }
)