

import numpy as np
import matplotlib.pyplot as plt

from SCFInitialGuess.utilities.analysis import plot_summary_scalars

def main():
    prefix = "butadien/notebooks/log/plain/run_.-tag-mse_with_l2_regularisation_"

    filedicts = {
        "MSE": prefix + "error.csv",
        "Regularisation": prefix + "weight_decay.csv",
        "Total error": prefix + "total_loss.csv"
    }

    plot_summary_scalars(filedicts)

    plt.savefig("/home/jo/Repos/MastersThesis/InitialGuess/Butadien/figures/ButadienPlainNNTrainingCost.png")

    plt.show()

if __name__ == '__main__':
    main()
