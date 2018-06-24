from functools import reduce
import argparse
from datetime import date, datetime

from os.path import join

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

from SCFInitialGuess.descriptors.coordinate_descriptors import NonWeighted, \
    RADIAL_GAUSSIAN_MODELS, AZIMUTHAL_GAUSSIAN_MODELS, POLAR_GAUSSIAN_MODELS, Gaussians


from analyze_pretrained_network_SDescriptor import do_analysis

DIM = 26

N_ELECTRONS = 30

def main():

    ############################################################################
    #         Inputs
    ############################################################################
    network_path = "butadien/data/networks/networkGaussians400EquidistantBroadening.npy"
    #network_path = "butadien/data/networks/networkSMatrixBigDataset.npy"
    data_folder = "butadien/data/400/"
    postfix = "400"

    log_file = "butadien/results/pretrained_" + str(date.today()) + ".log"
    ############################################################################



    msg.print_level = 2

    msg.info("Hi. Measurements for butadien",2)

    #--- fetch dataset and constants ---
    msg.info("Fetching dataset", 2)
    dataset, molecules, S = fetch_dataset(data_folder, postfix)
    dim = DIM
    s_test = make_matrix_batch(S, DIM, False)
    #---


    #--- Measureing & print ---
    with open(log_file, "a+") as f:
        f.write("##### Analysis of " + str(datetime.now()) + " #####\n")
        f.write("Network: " + network_path + "\n")
        f.write("Datafolder: " + data_folder + "\n")
        f.write("Postfix: " + postfix + "\n")

    do_analysis(network_path, dataset, molecules, s_test, log_file)


def fetch_dataset(folder, postfix, is_triu=True, index=None, test_samples=100):

    dim = DIM

    if is_triu:
        molecules = np.load(folder + "molecules" + postfix + ".npy") 
        
        descriptor = NonWeighted(
            Gaussians(*RADIAL_GAUSSIAN_MODELS["Equidistant-Broadening_1"]),
            Gaussians(*AZIMUTHAL_GAUSSIAN_MODELS["Equisitant_1"]),
            Gaussians(*POLAR_GAUSSIAN_MODELS["Equisitant_1"])
        )

        
        descriptor_values = descriptor.calculate_descriptors_batch(molecules)

        dataset, molecules = make_butadien_dataset(
            molecules,
            descriptor_values,
            [extract_triu(p, DIM) for p in np.load(folder + "P" + postfix + ".npy")],
            index=index,
            test_samples=test_samples
        )

        S = np.load(folder + "S" + postfix + ".npy")

        if index is None:
            index = np.arange(len(S))
        S_test = S[index[test_samples:]]

    else:
        raise NotImplementedError("Only triu networks!!")

    return dataset, molecules, S_test


   
if __name__ == '__main__':
    main()