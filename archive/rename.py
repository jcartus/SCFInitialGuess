
from os import listdir, rename
from os.path import join, isdir, isfile

root = "database/GMTKN55"

db = [join(root, d) for d in listdir(root) if isdir(join(root, d))]

molecules = [(d, m)  for d in db for m in listdir(d) if isdir(join(d, m))]

for mol in molecules:
    f = join(mol[0], mol[1], "struc.xyz")
    if isfile(f):
        print(f)
        rename(f, join(mol[0], mol[1], mol[1] + ".xyz"))
