import numpy as np
import argparse

from SCFInitialGuess.utilities.dataset import Dataset, assemble_batch
from SCFInitialGuess.utilities.constants import number_of_basis_functions as N_BASIS
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.nn.networks import EluTrNNN
from SCFInitialGuess.nn.training import train_network

def main(
    species,
    structure,
    save_path=None,
    source=None
    ):


    if structure[0] != N_BASIS[species] or structure[-1] != N_BASIS[species]:
        raise ValueError(
            "Invalid structure. Bad Input/Output dim (should be " + \
            "{0} but was {1}/{2}!".format(
                N_BASIS[species], structure[0], structure[-1]
            ) 
        )

    if source is None:
        source = ["../dataset/PyQChem/s22"]
    

    msg.info("Assmbling dataset ...", 2)
    dataset = Dataset(*assemble_batch(source, species))
    
    msg.info("Training model ...", 2)
    network = EluTrNNN(structure)
    network, sess = train_network(network, dataset)

    if not save_path is None:    
        msg.info("Storing model ...", 2)
        save_object = [
            network.structure,
            network.weights_values(sess),
            network.biases_values(sess)
        ]

        np.save(
            save_path,
            save_object
        )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='PROG',
        description="This program will train a neural network model."
    )

    parser.add_argument(
        "--species", 
        required=False,
        type=str,
        help="Chemical symbol of atomic species for which the model shall be created.",
        metavar="species",
        dest="species",
        default="H"
    )

    parser.add_argument(
        "--structure", 
        required=True,
        nargs='+',
        type=int,
        help="The structure of the network.",
        metavar="Network structure",
        dest="structure"
    )

    parser.add_argument(
        "--out", 
        required=False,
        type=str,
        help="The path (incl. name) where to store the trained model.",
        metavar="Save Path",
        dest="save_path",
        default=None
    )

    parser.add_argument(
        "--source", 
        required=True,
        type=str,
        help="The path to the root dir of the dataset database.",
        metavar="Top level folder",
        dest="source"
    )

    args = parser.parse_args()

    main(
        species=args.species,
        structure=args.structure,
        save_path=args.save_path,
        source=args.source
    )