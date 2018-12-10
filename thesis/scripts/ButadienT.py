import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

from SCFInitialGuess.utilities.usermessages import Messenger as msg

from os.path import join



#-------------------------------------------------------------------------------
# Fetch dataset
#-------------------------------------------------------------------------------

msg.info("Fetching dataset",1)

from SCFInitialGuess.utilities.dataset import extract_triu_batch, AbstractDataset
from sklearn.model_selection import train_test_split

# fetch dataset
data_path = "thesis/dataset/TSmall_sto3g/"
postfix = "TSmall_sto3g"
dim = 26
#data_path = "../butadien/data/"
#postfix = ""
#dim = 26


def split(x, y, ind):
    return x[:ind], y[:ind], x[ind:], y[ind:]

S = np.load(join(data_path, "S" + postfix + ".npy"))
P = np.load(join(data_path, "P" + postfix + ".npy"))
#F = np.load(join(data_path, "F" + postfix + ".npy"))

molecules = np.load(join(data_path, "molecules" + postfix + ".npy"))



ind = int(0.8 * len(molecules))
molecules_train, molecules_test = (molecules[:ind], molecules[ind:])

s_triu = extract_triu_batch(S, dim)
p_triu = extract_triu_batch(P, dim)

s_train, p_train, s_test, p_test = split(s_triu, p_triu, ind)
#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------
# Make descriptor
#-------------------------------------------------------------------------------
from SCFInitialGuess.descriptors.high_level import     AtomicNumberWeighted
from SCFInitialGuess.descriptors.coordinate_descriptors import     Gaussians, SPHAngularDescriptor
from SCFInitialGuess.descriptors.cutoffs import     BehlerCutoff1
from SCFInitialGuess.descriptors.models import     RADIAL_GAUSSIAN_MODELS, make_uniform
import pickle     
descriptor = AtomicNumberWeighted(
    Gaussians(*make_uniform(100, 100, 30)),
    #Gaussians(*RADIAL_GAUSSIAN_MODELS["Man"]),
    SPHAngularDescriptor(6),
    BehlerCutoff1(5)
)

pickle.dump(descriptor, open("thesis/models/ButadienTDescriptor/descriptor.npy", "wb"))

msg.info("Numer of descriptors per atom: " + \
    str(descriptor.number_of_descriptors))


msg.info("Number of descriptors in total: " + str(
    descriptor.calculate_all_descriptors(molecules[0]).shape
))


msg.info("Calulating symmetry vectors", 1)
G = []
for mol in molecules:
    G.append(
        descriptor.calculate_all_descriptors(mol).flatten()
    )
G = np.asarray(G)

# normalize
G_norm, mu, std = AbstractDataset.normalize(G)
np.save("thesis/models/ButadienTDescriptor/normalisation.npy", (mu, std))


g_train, g_test = G_norm[:ind], G_norm[ind:]

msg.info("Descriptors shape: " + str(
G.shape))
#-------------------------------------------------------------------------------






#-------------------------------------------------------------------------------
# Networks stuff
#-------------------------------------------------------------------------------

dim_in = G.shape[1]
dim_triu = dim * (dim + 1) // 2

msg.info("Dimensions: " + str((dim**2, dim_triu, dim_in)))




keras.backend.clear_session()


activation = "elu"
learning_rate = 1e-5
intializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

model = keras.Sequential()

# input layer
model.add(keras.layers.Dense(700, activation=activation, input_dim=dim_in, kernel_initializer=intializer, bias_initializer='zeros'))

# hidden
#for i in range(3):

model.add(keras.layers.Dense(
    500, 
    activation=activation, 
    kernel_initializer=intializer, 
    #bias_initializer='zeros',
    #kernel_regularizer=keras.regularizers.l2(1e-8)
))


model.add(keras.layers.Dense(
        400, 
        activation=activation, 
        kernel_initializer=intializer, 
        #bias_initializer='zeros',
        #kernel_regularizer=keras.regularizers.l2(1e-8)
))


#output
model.add(keras.layers.Dense(dim_triu))

model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='MSE', metrics=['mae', 'mse'])

model.summary()



filepath = "thesis/models/model_descriptos_" + postfix + ".h5"

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_mean_squared_error", 
    min_delta=1e-7, 
    patience=200, 
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_mean_squared_error', 
    factor=0.5, 
    patience=50, 
    verbose=1, 
    mode='auto', 
    min_delta=1e-4, 
    cooldown=50, 
    min_lr=1e-10
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, 
    monitor='val_mean_squared_error', 
    verbose=1, 
    save_best_only=False, 
    save_weights_only=False, 
    mode='auto', 
    period=1
)

log_dir = "./thesis/log//ButadienT" 
tensorboard = keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1, 
    batch_size=32, 
    #update_freq='epoch'
)
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
# Network training
#-------------------------------------------------------------------------------
epochs = 10000

while True:
    keras.backend.set_value(model.optimizer.lr, 1e-4)

    history = model.fit(
        x = g_train,
        y = p_train,
        epochs=epochs,
        shuffle=True,
        validation_data=(g_test, p_test), 
        verbose=1, 
        callbacks=[
            early_stopping, 
            reduce_lr, 
            checkpoint,
            tensorboard
        ]
    )
    
    msg.info("\n\n\n\n\nReset!\n\n\n")

