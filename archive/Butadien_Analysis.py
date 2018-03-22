from os import listdir
from os.path import join, normpath

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from butadien.load_data import load_data
from pyscf.scf import hf

from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.dataset import Dataset, Molecule
from SCFInitialGuess.nn.networks import EluTrNNN, EluFixedValue
from SCFInitialGuess.nn.training import train_network

dim = 26
model_save_path = "butadien/model.npy"
source = "butadien/data"

msg.info("Welcome", 2)



#--- fetching the network ---
msg.info("Loading Network", 2)

sess = tf.Session()
model = np.load(model_save_path)
network = EluFixedValue(*model)
network.setup()
sess.run(tf.global_variables_initializer())
#---

#--- fetching the molecules ---
msg.info("Fetching the molecules", 2)
def grep_molecule(input_file):
    import re
    
    with open(input_file) as f:
            
        molecule = re.search(r"\$molecule.*\$end", f.read(), re.DOTALL)
        if molecule is None:
            raise ValueError("No molecule found in " + f.name)
        else:
            molecule = molecule.group(0)

            # cut out geometries
            geometries = molecule.splitlines()[2:-1]

    # from geometries take the species and positions
    species, positions = [], []
    for line in geometries:
        splits = line.split()
        species.append(splits[0])
        positions.append(splits[1:])

    return species, positions

def fetch_molecules(folder):
    
    files = [file for file in listdir(folder) if ".inp" in file][:10]
    
    molecules = []

    for i, file in enumerate(files):
        
        msg.info("Fetching: " + str(i + 1) + "/" + str(len(files)))

        mol = Molecule(*grep_molecule(join(folder, file)))
        
        mol.basis = "sto-3g"
        
        molecules.append(mol)
    return molecules
             
molecules = fetch_molecules("butadien/data")
#---

#--- do scf ---
msg.info("Running the SCF calculations", 2)
iterations = []
for i, molecule in enumerate(molecules):
    
    mol = molecule.get_pyscf_molecule()
    
    
    msg.info("Calculating: " + str(i + 1) + "/200.")
    
    # assemble pyscf initial guesses
    P_1e = hf.init_guess_by_1e(mol)
    P_atom = hf.init_guess_by_atom(mol)
    P_minao = hf.init_guess_by_minao(mol)
    
    print("die normalen sind fertig")

    # nn guess
    S = hf.get_ovlp(mol).reshape(1, dim**2)
    P_NN = network.run(sess, S).reshape(dim, dim).astype('float64')


    print("P_NN fertig")
    
    iterations_molecule = []
    for guess in [P_1e, P_atom, P_minao, P_NN]:#, P_NN]:
        print("SCF")
        mf = hf.RHF(mol)
        mf.verbose = 1
        mf.kernel(dm0=guess)
        iterations_molecule.append(mf.iterations)
    

    print("SCFs done")
    iterations.append(iterations_molecule)

    print("Next step")

iterations = np.array(iterations)
#---


#--- statistics ---
fig, axes = plt.subplots(2,2)

bins = 1 # todo hier kann man auch ein array angeben

for i, name in enumerate(['1e', 'atom', 'P_minao']):
    
    hist, bins = np.histogram(iterations[:,i])
    center = (bins[:-1] + bins[1:]) / 2
    axes[i].bar(center, hist, label=name)
    
plt.legend()
plt.show()

#---