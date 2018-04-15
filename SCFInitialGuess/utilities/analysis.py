"""This file contains everyting required to generate plots.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


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

if __name__ == '__main__':
    dim = 26
    A = np.random.rand(dim, dim)
    matrix_error(A, orbitalTicks=True)
    plt.show()