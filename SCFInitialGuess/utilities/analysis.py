"""This file contains everyting required to generate plots.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from pandas import DataFrame
from pyscf.scf import hf

#from SCFInitialGuess.utilities.dataset import reconstruct_from_triu
from SCFInitialGuess.nn.cost_functions import absolute_error, symmetry_error
from SCFInitialGuess.nn.cost_functions import idempotence_error, predicted_occupance, makeMatrixBatch  

def statistics(x):
    return np.mean(x), np.std(x)

def matrix_error(error, xlabel="index", ylabel="index", ButadienMode=False, **kwargs):
    
    ax = sns.heatmap(error, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


    if ButadienMode:
        C_labels = ["1s  ", "2s  ", "2px", "2py", "2pz"]
        H_labels = ["1s  "]
        labels = [
            str(ci) + ". C: " + orbital \
                for ci in range(1,5) for orbital in C_labels
        ] + [
            str(hi) + ". H: " + orbital \
                for hi in range(1,7) for orbital in H_labels
        ]


        plt.yticks(np.arange(26), labels) 
        plt.xticks(np.arange(26), labels, rotation='vertical') 


    return ax
    

def prediction_scatter(
        actual,
        predicted, 
        xlabel="actual", 
        ylabel="predicted", 
        **kwargs
    ):

    data = DataFrame({xlabel: actual, ylabel: predicted})
    ax = sns.regplot(x=xlabel, y=ylabel, data=data, **kwargs)
    return ax
    
def iterations_histogram(
        dict_iterations, 
        xlabel="iterations / 1", 
        ylabel="count / 1", 
        **kargs
    ):

    data = DataFrame(dict_iterations)
    ax = sns.countplot(x=xlabel, y=ylabel, data=data)
    return ax

def plot_summary_scalars(
    file_label_dicts, 
    xlabel="steps / 1", 
    ylabel="costs / 1"
    ):
    """This function is used to plot data of scalars exported from
    tensorboard.

    Args:
        file_label_dicts <dict<str, str>>: dictionary with labels for plot and
        file path for data to be plotted.
        xlabel/ylabel <str>: plot axis labels.
    """
    
    fig = plt.figure()

    for label, fpath in file_label_dicts.items():
        with open(fpath, "r") as f:
            lines = f.readlines()[1:]

        steps, scalar = [], []
        for line in lines:
            splits = line.split(",")
            steps.append(int(splits[1]))
            scalar.append(float(splits[2]))

        plt.semilogy(steps, scalar, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend()

    return fig

class NetworkAnalyzer(object):

    def __init__(self, trainer):
        
        self.trainer = trainer
        self.graph = trainer.graph
        self.network = trainer.network

    def setup(self, dim, N_electrons, isUpperTriangle=False):
        
        self.dim = dim
        self.vector_dim = vector_dim = dim*(dim+1)/2 if isUpperTriangle else dim**2

        with self.graph.as_default():
            self.x = self.network.input_tensor
            self.f = self.network.output_tensor
            self.y = tf.placeholder(tf.float32, [None, vector_dim], "y")
            self.s = tf.placeholder(tf.float32, [None, vector_dim], "s")

            self.f_batch = makeMatrixBatch(self.f, dim, isUpperTriangle)
            self.s_batch = makeMatrixBatch(self.s, dim, isUpperTriangle)

            self.absolute_error = absolute_error(self.f, self.y)
            self.symmetry_error = symmetry_error(self.f_batch)
            self.idempotence_error = \
                idempotence_error(self.f_batch, self.s_batch)
            self.predicted_occupance_error = \
                predicted_occupance(self.f_batch, self.s_batch) - N_electrons

    @staticmethod
    def mf_initializer(mol):
        mf = hf.RHF(mol)
        mf.diis = None
        mf.verbose = 1

        return mf

    def measure_iterations(self, sess, network, dataset, molecules):
        
        iterations = []
        for (s_norm, molecule) in zip(dataset.testing[0], molecules):

            p = sess.run(
                self.f_batch, 
                {self.x: s_norm.reshape(1,-1)}
            ).reshape(self.dim, self.dim).astype('float64')

            mf = self.mf_initializer(molecule.get_pyscf_molecule())
            mf.kernel(dm0=p)

            iterations.append(mf.iterations)

        return iterations

    def measure(self, dataset, molecules, number_of_measurements=10):
        
        err_abs = []
        err_sym = []
        err_idem = []
        err_occ = []
        iterations = []

        s_raw = dataset.inverse_input_transform(dataset.testing[0])

        for i in range(number_of_measurements):
            network, sess = self.trainer.train(
                dataset,
                convergence_threshold=1e-6
            )

            with self.graph.as_default():
                err_abs.append(statistics(
                    sess.run(
                        self.absolute_error, 
                        {self.x: dataset.testing[0], self.y: dataset.testing[1]}
                    )
                ))

                err_sym.append(statistics(
                    sess.run(self.symmetry_error, {self.x: dataset.testing[0]})
                ))

                err_idem.append(statistics(
                    sess.run(self.idempotence_error, 
                    {self.x: dataset.testing[0], self.s: s_raw})
                ))

                err_occ.append(statistics(
                    sess.run(
                        self.predicted_occupance_error, 
                        {self.x: dataset.testing[0], self.s: s_raw}
                    )
                ))

                iterations.append(statistics(self.measure_iterations(
                    sess, network, dataset, molecules
                )))
            
        
        return (
            np.array(err_abs),
            np.array(err_sym),
            np.array(err_idem),
            np.array(err_occ),
            np.array(iterations)
        )
    
    @staticmethod
    def make_results_str(results):
        
        out = ""

        def format_results(result):
            out = list(map(
                lambda x, y: "{:0.5E} +- {:0.5E}".format(x, y),
                result
            ))
            return "\n".join(out)

        out += "--- Absolute Error ---\n"
        out += format_results(results[0])
        out += "--- Symmetry Error ---\n"
        out += format_results(results[1])
        out += "--- Idempotence Error ---\n"
        out += format_results(results[2])
        out += "--- Occupance Error ---\n"
        out += format_results(results[3])
        out += "--- Avg. Iterations ---\n"
        out += format_results(results[4])

        return out

if __name__ == '__main__':
    dim = 26
    A = np.random.rand(dim, dim)
    matrix_error(A, orbitalTicks=True)
    plt.show()