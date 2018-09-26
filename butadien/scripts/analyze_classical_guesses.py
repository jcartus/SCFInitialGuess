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
from SCFInitialGuess.nn.post_processing import multi_mc_wheeny


DIM = 26

N_ELECTRONS = 30

def main():

    ############################################################################
    #         Inputs
    ############################################################################
    data_folder = "butadien/data/"
    postfix = ""#400"
    log_file = "butadien/results/classical_guess_performance_" + str(date.today()) + ".log"
    ############################################################################



    msg.print_level = 2

    msg.info("Hi. Classical guess performance for any stupid dataset",2)

    #--- fetch dataset and constants ---
    msg.info("Fetching dataset", 2)
    dataset, molecules = fetch_dataset(data_folder, postfix)
    dim = DIM
    s_raw = make_matrix_batch(dataset.inverse_input_transform(dataset.testing[0]), DIM, True)
    #---


    #--- Measureing & print ---
    with open(log_file, "a+") as f:
        f.write("##### Analysis of " + str(datetime.now()) + " #####\n")
        f.write("Datafolder: " + data_folder + "\n")
        f.write("Postfix: " + postfix + "\n")

    do_analysis(dataset, molecules, s_raw, log_file)



def fetch_dataset(folder, postfix, is_triu=True, index=None):

    dim = DIM

    def load_triu(S, P, dim):
        
        return [extract_triu(s, dim) for s in S], [extract_triu(p, dim) for p in P]

    if is_triu:
        dataset, molecules = make_butadien_dataset(
            np.load(
                folder + "molecules" + postfix + ".npy"
            ),
            *load_triu(
                np.load(folder + "S" + postfix + ".npy"),
                np.load(folder + "P" + postfix + ".npy"), 
                dim
            ), 
            index=index,
            test_samples=100
        )
    else:
        raise NotImplementedError("Only triu networks!!")

    return dataset, molecules


    
def do_analysis(dataset, molecules, s_raw, log_file):

    #--- calculate guesses ---
    msg.info("Calculating guesses ...",2)

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

    with open(log_file, "a+") as f:
        f.write("\n\n+++++ H_Core +++++\n")
    msg.info("Results H_Core: ", 1)
    measure_and_display(
        p_1e.reshape(-1, DIM**2), dataset, molecules, False, log_file, s=s_raw
    )

    with open(log_file, "a+") as f:
        f.write("\n\n+++++ SAP +++++\n")
    msg.info("Results SAP: ", 1)
    measure_and_display(
        p_sap.reshape(-1, DIM**2), dataset, molecules, False, log_file, s=s_raw
    )

    with open(log_file, "a+") as f:
        f.write("\n\n+++++ MINAO +++++\n")
    msg.info("Results MINAO: ", 1)
    measure_and_display(
        p_minao.reshape(-1, DIM**2), dataset, molecules, False, log_file, s=s_raw
    )

    with open(log_file, "a+") as f:
        f.write("\n\n+++++ GWH +++++\n")
    msg.info("Results GWH: ", 1)
    measure_and_display(
        p_gwh.reshape(-1, DIM**2), dataset, molecules, False, log_file, s=s_raw
    )


def measure_and_display(p, dataset, molecules, is_triu, log_file, s):
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

    dim = DIM

    result = make_results_str(measure_all_quantities(
        p,
        dataset,
        molecules[1],
        N_ELECTRONS,
        mf_initializer,
        dim,
        is_triu=is_triu,
        is_dataset_triu=True,
        s=s
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

   
if __name__ == '__main__':
    main()