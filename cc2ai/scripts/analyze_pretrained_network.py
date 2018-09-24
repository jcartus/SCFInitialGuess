from functools import reduce
import argparse
from datetime import date, datetime

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

def measure_and_display(p, dataset, molecules, molecule_type, is_triu, log_file):
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
        is_triu=is_triu
    ))

    result += "--- Iterations Damped ---\n" + \
        format_results(statistics(list(measure_iterations(
            mf_initializer_damping,
            make_matrix_batch(p, dim, is_triu=is_triu).astype('float64'),
            molecules[1]
        ))))

    result += "\n" + "--- Iterations DIIS ---\n" + \
        format_results(statistics(list(measure_iterations(
            mf_initializer_diis,
            make_matrix_batch(p, dim, is_triu=is_triu).astype('float64'),
            molecules[1]
        ))))

    msg.info(result, 1)
    with open(log_file, "a+") as f:
        f.write(result)

def main(molecule_type):

    msg.print_level = 2

    msg.info("Hi. Measurements for " + molecule_type, 2)

    #--- fetch dataset and constants ---
    msg.info("Fetching dataset", 2)
    dataset_triu, molecules = fetch_dataset(molecule_type, True)
    dataset, _ = fetch_dataset(molecule_type, False)
    dim = DIM[molecule_type]
    #---

    #--- fetch network ---
    msg.info("Fetching pretained network ", 2)
    graph = tf.Graph()
    structure, weights, biases = np.load(
        "cc2ai/" + molecule_type + "/network_" + molecule_type + ".npy", 
        encoding="latin1"
    )
    #---

    with graph.as_default():
        sess = tf.Session()
        network = EluFixedValue(structure, weights, biases)
        network.setup()
        sess.run(tf.global_variables_initializer())
    #---

    #--- calculate guesses ---
    msg.info("Calculating guesses ...",2)
    s_raw = make_matrix_batch(dataset_triu.inverse_input_transform(dataset_triu.testing[0]), dim, True)
    
    msg.info("Neural network ", 1)
    p_nn = network.run(sess, dataset_triu.testing[0])

    msg.info("McWheenys", 1)
    p_batch = make_matrix_batch(p_nn, dim, True)
    p_mcw1 = np.array(list(map(lambda x: multi_mc_wheeny(x[0], x[1], n_max=1), zip(p_batch, s_raw))))
    p_mcw5 = np.array(list(map(lambda x: multi_mc_wheeny(x[0], x[1], n_max=5), zip  (p_batch, s_raw))))

    msg.info("Classics", 1)
    p_1e = np.array([
        hf.init_guess_by_1e(mol.get_pyscf_molecule()) for mol in molecules[1]
    ])
    p_sap = np.array([
        hf.init_guess_by_atom(mol.get_pyscf_molecule()) for mol in molecules[1]
    ])
    p_minao = np.array([
        hf.init_guess_by_minao(mol.get_pyscf_molecule()) for mol in molecules[1]
    ])
    p_gwh = np.array([
        hf.init_guess_by_wolfsberg_helmholtz(mol.get_pyscf_molecule()) for mol in molecules[1]
    ])
    #--- 

    #--- Measureing & print ---
    log_file = "cc2ai/" + molecule_type + "/pretrained_" + str(date.today()) + ".log"
    with open(log_file, "a+") as f:
        f.write("##### Analysis of " + str(datetime.now()) + " #####\n")
    msg.info("Results NN: ", 1)
    with open(log_file, "a+") as f:
        f.write("\n+++++ Plain NN +++++\n")
    measure_and_display(
        p_nn, dataset_triu, molecules, molecule_type, True, log_file
    )
    
    with open(log_file, "a+") as f:
        f.write("\n\n+++++ McW 1 +++++\n")
    msg.info("Results McWheeny 1: ",1)
    measure_and_display(
        p_mcw1.reshape(-1, dim**2), dataset, molecules, molecule_type, False, log_file
    )
    
    with open(log_file, "a+") as f:
        f.write("\n\n+++++ McW 5 +++++\n")
    msg.info("Results McWheeny 5: ", 1)
    measure_and_display(
        p_mcw5.reshape(-1, dim**2), dataset, molecules, molecule_type, False, log_file
    )

    with open(log_file, "a+") as f:
        f.write("\n\n+++++ H_Core +++++\n")
    msg.info("Results H_Core: ", 1)
    measure_and_display(
        p_1e.reshape(-1, dim**2), dataset, molecules, molecule_type, False, log_file
    )

    with open(log_file, "a+") as f:
        f.write("\n\n+++++ SAP +++++\n")
    msg.info("Results SAP: ", 1)
    measure_and_display(
        p_sap.reshape(-1, dim**2), dataset, molecules, molecule_type, False, log_file
    )

    with open(log_file, "a+") as f:
        f.write("\n\n+++++ MINAO +++++\n")
    msg.info("Results MINAO: ", 1)
    measure_and_display(
        p_minao.reshape(-1, dim**2), dataset, molecules, molecule_type, False, log_file
    )

    with open(log_file, "a+") as f:
        f.write("\n\n+++++ GWH +++++\n")
    msg.info("Results GWH: ", 1)
    measure_and_display(
        p_gwh.reshape(-1, dim**2), dataset, molecules, molecule_type, False, log_file
    )

   
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