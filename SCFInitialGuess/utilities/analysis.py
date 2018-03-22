"""This file contains everyting required to generate plots.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

def matrix_error(error, xlabel="index", ylabel="index", **kwargs):
    
    ax = sns.heatmap(error, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    
def iterations_histogram(dict_iterations, 
    xlabel="iterations / 1", 
    ylabel="count / 1", 
    **kargs
    ):

    data = DataFrame(dict_iterations)
    ax = sns.countplot(x=xlabel, y=ylabel, data=data)
    return ax

    