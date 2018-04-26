from functools import reduce
from datetime import date, datetime

import numpy as np
import tensorflow as tf

from pyscf.scf import hf

from SCFInitialGuess.nn.networks import EluTrNNN
from SCFInitialGuess.nn.training import Trainer
from SCFInitialGuess.nn.cost_functions import MSE
from SCFInitialGuess.utilities.dataset import Dataset, reconstruct_from_triu, extract_triu
from SCFInitialGuess.utilities.usermessages import Messenger as msg
from SCFInitialGuess.utilities.analysis import statistics

# ethan

#msg.print_level = 1

class TriuNetworkAnalyzer(object):

    def __init__(self, trainer, dim, n_electrons):
        
        self.trainer = trainer    
        self.dim = dim
        self.vector_dim = dim*(dim+1)/2 
        self.n_electrons = n_electrons

    @staticmethod
    def mf_initializer(mol):
        mf = hf.RHF(mol)
        mf.diis = None
        mf.verbose = 1

        return mf

    @staticmethod
    def measure_iterations(mf_initializer, guesses, molecules):
        
        iterations = []
        for i, (p, molecule) in enumerate(zip(guesses, molecules)):

            msg.info("Iteration calculation: " + str(i))

            mf = mf_initializer(molecule.get_pyscf_molecule())
            mf.kernel(dm0=p)

            iterations.append(mf.iterations)

        return iterations
    
    @staticmethod
    def makeMatrixBatch(vector_batch, dim):
        """Turns a batch of flatted out matrices into a batch of actual matrices
        i.e. reshapes the vectors into dim x dim matrices again.
        TODO describe inputs
        """

        vector_batch = np.array(list(map(
            lambda x: reconstruct_from_triu(x, dim), 
            vector_batch
        )))

        return vector_batch.reshape([-1, dim, dim])

    def measure(
        self, 
        dataset, 
        molecules, 
        number_of_measurements=10,
        convergence_threshold=1e-6
        ):
        
        err_abs = []
        err_sym = []
        err_idem = []
        err_occ = []
        iterations = []

        s_raw = self.makeMatrixBatch(
            dataset.inverse_input_transform(dataset.testing[0]),
            self.dim
        )

        for i in range(number_of_measurements):
            
            msg.info("Network: " + str(i), 2)
            msg.info("train ... " + str(i), 1)

            network, sess = self.trainer.train(
                dataset,
                convergence_threshold=convergence_threshold
            )

            with self.trainer.graph.as_default():
                
                msg.info("calculate quantities ...", 1)

                p = network.run(sess, dataset.testing[0])
                p_batch = self.makeMatrixBatch(p, self.dim)

                err_abs.append(statistics(list(
                    self.measure_absolute_error(p, dataset)
                )))

                err_sym.append(statistics(list(
                    self.measure_symmetry_error(p_batch)
                )))

                err_idem.append(statistics(list(
                    self.measure_idempotence_error(p_batch, s_raw)
                )))

                err_occ.append(statistics(list(
                    self.measure_occupance_error(p_batch, s_raw, self.n_electrons)
                )))

                iterations.append(statistics(list(
                    self.measure_iterations(
                        self.mf_initializer, 
                        p_batch.astype('float64'), 
                        molecules
                    )
                )))
            
        
        return (
            np.array(err_abs),
            np.array(err_sym),
            np.array(err_idem),
            np.array(err_occ),
            np.array(iterations)
        )
    
    @staticmethod
    def measure_absolute_error(p, dataset):

        return np.mean(np.abs(p - dataset.testing[0]), 1)

    @staticmethod
    def measure_symmetry_error(p_batch):
        for p in p_batch:
            yield np.mean(np.abs(p - p.T))

    @staticmethod
    def measure_idempotence_error(p_batch, s_batch):
        for (p, s) in zip(p_batch, s_batch):
            yield np.mean(np.abs(2 * p - reduce(np.dot, (p, s, p))))

    @staticmethod
    def measure_occupance_error(p_batch, s_batch, n_electrons):
        for (p, s) in zip(p_batch, s_batch):
            yield np.mean(np.abs(np.trace(np.dot(p, s)) - n_electrons))

    @staticmethod
    def make_results_str(results):
        
        out = ""

        def format_results(result):
            out = list(map(
                lambda x: "{:0.5E} +- {:0.5E}".format(*x),
                result
            ))
            return "\n".join(out)

        out += "--- Absolute Error ---\n"
        out += format_results(results[0])
        out += "\n"
        out += "--- Symmetry Error ---\n"
        out += format_results(results[1])
        out += "\n"
        out += "--- Idempotence Error ---\n"
        out += format_results(results[2])
        out += "\n"
        out += "--- Occupance Error ---\n"
        out += format_results(results[3])
        out += "\n"
        out += "--- Avg. Iterations ---\n"
        out += format_results(results[4])
        out += "\n"

        return out


def fetch_dataset(path, dim):
    #--- the dataset ---
    S, P = np.load(path)

    
    ind_cut = 150
    index = np.arange(200)
    np.random.shuffle(index)

    S_triu = list(map(lambda x: extract_triu(x, dim), S))
    P_triu = list(map(lambda x: extract_triu(x, dim), P))

    S_test = np.array(S_triu)[index[150:]]
    P_test = np.array(P_triu)[index[150:]]
    
    S_train = np.array(S_triu)[index[:150]]
    P_train = np.array(P_triu)[index[:150]]
    
    dataset = Dataset(np.array(S_train), np.array(P_train), split_test=0.0)

    dataset.testing = (Dataset.normalize(S_test, mean=dataset.x_mean, std=dataset.x_std)[0], P_test)

    return dataset


def main():

    dim = {
        "ethan": 58,
        "ethen": 48,
        "ethin": 38
    }

    log_file = "cc2ai/results/" + str(date.today()) + ".log"
    with open(log_file, "a+") as f:
        f.write("##### Analysis of " + str(datetime.now()) + " #####\n")

    for key, value in dim.items():
        dim_triu = int(value * (value + 1) / 2)

        msg.info("Starting " + key, 2)

        msg.info("Fetch data ...", 2)
        dataset = fetch_dataset(
            "cc2ai/" + key + "/dataset_" + key + "_6-31g**.npy", 
            value
        )
        molecules = np.load(
            "cc2ai/" + key +"/molecules_" + key + "_6-31g**.npy"
        )[150:]

        msg.info("Setup trainer ...", 2)
        trainer = Trainer(
            EluTrNNN([dim_triu, dim_triu, dim_triu]),
            cost_function=MSE()
        )
        trainer.setup()
        
        msg.info("Setup analyzer ...", 2)
        analyzer = TriuNetworkAnalyzer(trainer, value, 30)

        msg.info("Do measurements ...", 2)
        results = analyzer.measure(dataset, molecules)

        msg.info("Plot results ...", 2)

        with open(log_file, "a+") as f:
            f.write(analyzer.make_results_str(results))


if __name__ == '__main__':
    main()