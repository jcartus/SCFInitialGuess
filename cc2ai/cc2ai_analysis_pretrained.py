from functools import reduce
import argparse

import numpy as np
import tensorflow as tf

from pyscf.scf import hf

from SCFInitialGuess.nn.networks import EluFixedValue
from SCFInitialGuess.utilities.dataset import make_butadien_dataset, \
    extract_triu, make_matrix_batch
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.analysis import measure_all_quantities, \
    measure_iterations, make_results_str, statistics,\
    mf_initializer, mf_initializer_damping, mf_initializer_diis


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

def fetch_dataset(molecule_type):

        dim = DIM[molecule_type]

        def load_triu(S, P, dim):
            
            return [extract_triu(s, dim) for s in S], [extract_triu(p, dim) for p in P]

        dataset, molecules = make_butadien_dataset(
            np.load(
                "cc2ai/" + \
                molecule_type + "/molecules_" + molecule_type + "_6-31g**.npy"
            ),
            *load_triu(*np.load(
                "cc2ai/" + \
                molecule_type + "/dataset_" + molecule_type + "_6-31g**.npy"
            ), dim), 
        )

        return dataset, molecules

def mc_wheeny_purification(p,s):
    return (3 * np.dot(np.dot(p, s), p) - np.dot(np.dot(np.dot(np.dot(p, s), p), s), p)) / 2

def multi_mc_wheeny(p, s, n_max=4):
    for i in range(n_max):
        p = mc_wheeny_purification(p, s)
    return p

def measure_and_display(p, dataset, molecules, molecule_type):
    def format_results(result):
        if isinstance(result, list):
            out = list(map(
                lambda x: "{:0.5E} +- {:0.5E}".format(*x),
                result
            ))
            out = "\n".join(out)
        else:
            out =  "{:0.5E} +- {:0.5E}".format(*result)
        return out

    dim = DIM[molecule_type]

    result = make_results_str(measure_all_quantities(
        p,
        dataset,
        molecules[1],
        N_ELECTRONS[molecule_type],
        mf_initializer,
        dim,
        is_triu=True
    ))

    result += "--- Iterations Damped ---\n" + \
        format_results(statistics(list(measure_iterations(
            mf_initializer_damping,
            make_matrix_batch(p, dim, is_triu=True).astype('float64'),
            molecules[1]
        ))))

    result += "\n" + "--- Iterations DIIS ---\n" + \
        format_results(statistics(list(measure_iterations(
            mf_initializer_diis,
            make_matrix_batch(p, dim, is_triu=True).astype('float64'),
            molecules[1]
        ))))

    msg.info("\n" +  result, 1)
    

def main(molecule_type):

    msg.print_level = 2

    msg.info("Hi. Measurements for " + molecule_type, 2)

    msg.info("Fetching dataset", 2)
    dataset, molecules = fetch_dataset(molecule_type)
    dim = DIM[molecule_type]

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
    s_raw = make_matrix_batch(dataset.inverse_input_transform(dataset.testing[0]), dim, True)
    
    msg.info("Neural network ", 1)
    p_nn = network.run(sess, dataset.testing[0])

    msg.info("McWheenys", 1)
    p_batch = make_matrix_batch(p_nn, dim, True)
    p_mcw1 = np.array(list(map(lambda x: multi_mc_wheeny(x[0], x[1], n_max=1), zip(p_batch, s_raw))))
    p_mcw5 = np.array(list(map(lambda x: multi_mc_wheeny(x[0], x[1], n_max=5), zip  (p_batch, s_raw))))

    msg.info("Classics", 1)
    p_sap = [
        hf.init_guess_by_atom(mol.get_pyscf_molecule()) for mol in molecules[1]
    ]
    p_minao = [
        hf.init_guess_by_minao(mol.get_pyscf_molecule()) for mol in molecules[1]
    ]
    p_gwh = [
        hf.init_guess_by_wolfsberg_helmholtz(mol.get_pyscf_molecule()) for mol in molecules[1]
    ]
    #--- 

    #--- Measureing:
    msg.info("Results NN: ", 1)
    measure_and_display(p_nn, dataset, molecules, molecule_type)

    msg.info("Results McWheeny 1: ",1)
    measure_and_display(
        list(map(lambda x: extract_triu(x, dim), p_mcw1)), 
        dataset, molecules, molecule_type)

    msg.info("Results McWheeny 5: ", 1)
    measure_and_display(
        list(map(lambda x: extract_triu(x, dim), p_mcw5)), 
        dataset, molecules, molecule_type)

    msg.info("Results SAP: ", 1)
    measure_and_display(
        list(map(lambda x: extract_triu(x, dim), p_gwh)), 
        dataset, molecules, molecule_type)

    msg.info("Results MINAO: ", 1)
    measure_and_display(
        list(map(lambda x: extract_triu(x, dim), p_minao)), 
        dataset, molecules, molecule_type)

    msg.info("Results GWH: ", 1)
    measure_and_display(
        list(map(lambda x: extract_triu(x, dim), p_gwh)), 
        dataset, molecules, molecule_type)

   
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

    args = parser.parse_args()

    main(args.molecule)