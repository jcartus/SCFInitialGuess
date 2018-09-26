import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

def extract_triu_batch(A_batch, dim):
    return np.array(
        [extract_triu(A, dim) for A in A_batch]
    )

def extract_triu(A, dim):
    """Extracts the upper triangular part of the matrix.
    Input can be matrix, will be reshaped if it is not.
    """
    return A.reshape(dim, dim)[np.triu_indices(dim)]


def normalize(x, std_tolerance=1e-20, mean=None, std=None):

        if mean is None or std is None:
            mean = np.average(x, 0)
            std = np.std(x, 0)

        # handle dvision by zero if std == 0
        return (
            (x - mean) / np.where(np.abs(std) < std_tolerance, 1, std),
            mean,
            std
        )

def make_dataset(dim):
    
    #path = "./TSmall/"
    #S = np.load(path + "STSmall.npy")
    #P = np.load(path + "PTSmall.npy")
    path = "../butadien/data/400/"

    S = np.load(path + "S400.npy")
    P = np.load(path + "P400.npy")


    s_triu = extract_triu_batch(S, dim)
    p_triu = extract_triu_batch(P, dim)

    s_triu_norm, mu, sigma = normalize(s_triu)

    return s_triu_norm, p_triu

def build_nn(dim):
    
    dim_triu = dim * (dim + 1) // 2
    model = keras.Sequential()

    activation = tf.nn.selu

    # input
    model.add(keras.layers.Dense(dim_triu, activation=activation, input_dim=dim_triu))
    
    # hidden
    for i in range(8):
        model.add(keras.layers.Dense(dim_triu, activation=activation))
    
    # output
    model.add(keras.layers.Dense(dim_triu))

    model.compile(
        optimizer=tf.train.AdamOptimizer(0.0001), 
        loss='MSE', 
        metrics=['mae', 'mse']
    )

    return model

def main():
    dim = 26#70
    epochs = 1000

    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #session = tf.Session(config=config)
    #keras.backend.set_session(session)

    data = make_dataset(dim)
    model = build_nn(dim)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error", 
        min_delta=1e-4, 
        patience=3
    )

    history = model.fit(
        data[0], data[1], 
        epochs=epochs, 
        validation_split=0.2, 
        verbose=1, 
        #callbacks=[early_stopping]
    )


    with open('log.json', 'w') as f:
        f.write(json.dumps(history.history))

if __name__ == '__main__':
    main()