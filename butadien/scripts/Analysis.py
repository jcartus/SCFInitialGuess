
import numpy as np
import tensorflow as tf

from SCFInitialGuess.nn.networks import EluTrNNN
from SCFInitialGuess.nn.training import Trainer
from SCFInitialGuess.utilities.dataset import Dataset
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.analysis import NetworkAnalyzer

dim = 26
msg.print_level = 1

def fetch_dataset():
    #--- the dataset ---
    S, P = np.load("butadien/data/dataset.npy")

    ind_cut = 150
    index = np.arange(200)
    np.random.shuffle(index)

    S_test = np.array(S)[index[ind_cut:]]
    P_test = np.array(P)[index[ind_cut:]]

    S_train = np.array(S)[index[:ind_cut]]
    P_train = np.array(P)[index[:ind_cut]]

    dataset = Dataset(np.array(S_train), np.array(P_train), split_test=0.0)

    dataset.testing = (Dataset.normalize(S_test, mean=dataset.x_mean, std=dataset.x_std)[0], P_test)
    #---
    return dataset

def main():

    dataset = fetch_dataset()
    molecules = np.load("butadien/data/molecules.npy")[150:]

    trainer = Trainer(EluTrNNN([dim**2, dim**2, dim**2]))
    trainer.setup()
    analyzer = NetworkAnalyzer(trainer)
    analyzer.setup(dim, 30)
    results = analyzer.measure(dataset, molecules)

    print(analyzer.make_results_str(results))


if __name__ == '__main__':
    main()