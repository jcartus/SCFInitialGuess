from functools import reduce
import argparse
from datetime import date, datetime

import numpy as np
import tensorflow as tf

from pyscf.scf import hf
from pyscf.tools import cubegen

from SCFInitialGuess.nn.networks import EluFixedValue
from SCFInitialGuess.utilities.dataset import extract_triu, reconstruct_from_triu, \
    make_butadien_dataset
from SCFInitialGuess.utilities.usermessages import Messenger as msg


DIM = {
    "ethan": 58,
    "ethen": 48,
    "ethin": 38
}

N_ELECTRONS = {
    "ethan": 18,
    "ethen": 16,
    "ethin": 14
}

def fetch_dataset(molecule_type, is_triu=True, index=None):

        dim = DIM[molecule_type]

        def load_triu(S, P, dim):
            
            return [extract_triu(s, dim) for s in S], [extract_triu(p, dim) for p in P]

        if is_triu:
            dataset, molecules = make_butadien_dataset(
                np.load(
                    "cc2ai/" + \
                    molecule_type + "/molecules_" + molecule_type + "_6-31g**.npy"
                ),
                *load_triu(*np.load(
                    "cc2ai/" + \
                    molecule_type + "/dataset_" + molecule_type + "_6-31g**.npy"
                ), dim), 
                index=index
            )
        else:
            dataset, molecules = make_butadien_dataset(
                np.load(
                    "cc2ai/" + \
                    molecule_type + "/molecules_" + molecule_type + "_6-31g**.npy"
                ),
                *np.load(
                    "cc2ai/" + \
                    molecule_type + "/dataset_" + molecule_type + "_6-31g**.npy"
                ),
                index=index
            )

        return dataset, molecules

def mc_wheeny_purification(p,s):
    return (3 * np.dot(np.dot(p, s), p) - np.dot(np.dot(np.dot(np.dot(p, s), p), s), p)) / 2

def multi_mc_wheeny(p, s, n_max=4):
    for i in range(n_max):
        p = mc_wheeny_purification(p, s)
    return p

def main(molecule_type, sample_index):

    msg.print_level = 2

    msg.info("Hi. Measurements for " + molecule_type, 2)

    #--- fetch dataset and constants ---
    msg.info("Fetching sample " + str(sample_index) + " from datase", 2)

    dim = DIM[molecule_type]

    molecules = np.load(
        "cc2ai/" + \
        molecule_type + "/molecules_" + molecule_type + "_6-31g**.npy"
    )
    S, P = np.load(
        "cc2ai/" + \
        molecule_type + "/dataset_" + molecule_type + "_6-31g**.npy"
    )
    dataset = make_butadien_dataset(molecules, S, P)[0]

    molecule = molecules[sample_index].get_pyscf_molecule()
    s = S[sample_index].reshape(dim, dim)
    s_normed = extract_triu(
        dataset.inverse_input_transform(s.reshape(1, -1)), 
        dim
    ).reshape(1,-1)
    p = P[sample_index].reshape(dim, dim)
    #---

    #--- fetch network ---
    msg.info("Fetching pretained network ", 2)
    graph = tf.Graph()
    structure, weights, biases = np.load(
        "cc2ai/" + molecule_type + "/network_" + molecule_type + ".npy", 
        encoding="latin1"
    )
    
    with graph.as_default():
        sess = tf.Session()
        network = EluFixedValue(structure, weights, biases)
        network.setup()
        sess.run(tf.global_variables_initializer())
    #---

    #--- calculate guesses ---
    msg.info("Calculating guesses ...",2)
    
    msg.info("Neural network ", 1)
    p_nn = reconstruct_from_triu(network.run(sess, s_normed), dim).reshape(dim, dim)

    msg.info("McWheenys", 1)
    p_mcw1 = multi_mc_wheeny(p_nn, s, n_max=1)
    p_mcw5 = multi_mc_wheeny(p_nn, s, n_max=5)

    msg.info("Classics", 1)
    p_sap = hf.init_guess_by_atom(molecule)
    
    p_minao = hf.init_guess_by_minao(molecule)
    
    p_gwh = hf.init_guess_by_wolfsberg_helmholtz(molecule)
    #--- 

    #--- Measureing & print ---
    outfile = "cc2ai/" + molecule_type + "/cube_" + str(sample_index) + "_"
    
    msg.info("Results Converged ", 1)
    cubegen.density(molecule, outfile + "converged.cube", p.astype("float64"))

    msg.info("Results NN: ", 1)
    cubegen.density(molecule, outfile + "nn.cube", p_nn.astype("float64"))
    
    msg.info("Results McWheeny 1: ",1)
    cubegen.density(molecule, outfile + "mcw1.cube", p_mcw1.astype("float64"))
    
    msg.info("Results McWheeny 5: ", 1)
    cubegen.density(molecule, outfile + "mcw5.cube", p_mcw5.astype("float64"))

    msg.info("Results SAP: ", 1)
    cubegen.density(molecule, outfile + "sap.cube", p_sap)

    msg.info("Results MINAO: ", 1)
    cubegen.density(molecule, outfile + "minao.cube", p_minao)

    msg.info("Results GWH: ", 1)
    cubegen.density(molecule, outfile + "gwh.cube", p_gwh)
    #---

   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG',
        description="This program will read molecule geometries from a data" + 
            "base folder, generate a few geometries and do md runs with qChem on them" 
        
    )

    parser.add_argument(
        "--molecule", 
        required=False,
        help="name of the molecule for which to train a network",
        dest="molecule",
        choices=["ethan", "ethen", "ethin"],
        default="ethan"
    )

    parser.add_argument(
        "--sample", 
        required=False,
        help="Index of the molecule for which the densities are plotted",
        dest="index",
        type=int,
        default=180
    )


    args = parser.parse_args()

    main(args.molecule, args.index)