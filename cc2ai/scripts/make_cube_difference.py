"""This script will create the difference between two cube files
for the same molecule

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np

def main(cube_1, cube_2, outfile):
    
    with open(cube_1, 'r') as f1:
        lines1 = f1.readlines()

    with open(cube_2, 'r') as f2:
        lines2 = f2.readlines()

    number_of_atoms = int(lines1[2].split()[0])

    end_of_header = 6 + number_of_atoms # first row after header
    header = lines1[:(end_of_header)]

    def extract(x):
        return np.array(list(map(float, x.split())))

    def format(x):
        return "{:12.6f}".format(x)

    with open(outfile, 'a') as fout:
        fout.write("".join(header))

        for (lhs, rhs) in zip(lines1[end_of_header:], lines2[end_of_header:]):
            diff = extract(lhs) - extract(rhs)

            fout.write("".join(map(format, diff)) + "\n")




if __name__ == '__main__':
    folder = "cc2ai/ethen/"
    main(
        folder + "cube_180_converged.cube",
        folder + "cube_180_sap.cube",
        folder + "cube_diff_converged-sap.cube"
    )