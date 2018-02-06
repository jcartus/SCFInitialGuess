
from os import walk
from os.path import normpath, join

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utilities.dataset import Result, normalize
from utilities.constants import number_of_basis_functions as N_BASIS





database = normpath("/home/jcartus/Documents/SCFInitialGuess/dataset/s22")


#--- create and preprocess a dataset for C ---
tree = walk(database)
batches = []

x, y = [], []
for directory, _, files in list(tree)[1:]:
    result = Result(directory)
    data = result.create_batch("C")
    x += data[0] 
    y += list(map(np.diag, data[1])) #todo: maybe the cast to list is not necessary
        
x_raw, x_mean, x_std = normalize(np.array(x))
y_raw = np.array(y)

# todo refactor to a function in dataset
ind_train = int(np.floor(0.8 * len(x_raw)))
x_train, y_train = x_raw[:ind_train], y_raw[:ind_train]
x_test, y_test = x_raw[ind_train:], y_raw[:ind_train:]
#---


sess = tf.Session()

#--- set up the network ---
dim = N_BASIS['C']

x = tf.placeholder("float", shape=[None, dim])
y = tf.placeholder("float", shape=[None, dim])

def weight_init(dim_in, dim_out):
    weights = tf.truncated_normal([dim_in, dim_out], stddev=0.1)
    return tf.Variable(weights)

def bias_init(dim):
    bias = tf.truncated_normal([dim], stddev=0.01)
    return tf.Variable(bias)

def layer(x, dim_in, dim_out, weight_init=weight_init, bias_init=bias_init, activation=None):
       
    w = weight_init(dim_in, dim_out)
    b = bias_init(dim_out)

    pre_activation = tf.matmul(x, w) + b

    if activation is None:
        return pre_activation
    else:
        return activation(pre_activation)

z1 = layer(x, dim, 100, activation=tf.nn.elu)
z2 = layer(z1, 100, 80, activation=tf.nn.elu)
z3 = layer(z2, 80, 50, activation=tf.nn.elu)
output = layer(z3, 50, dim) #linear activation
#---

#--- train the net ---
loss = tf.losses.mean_squared_error(y, output)
optimizer = tf.train.AdamOptimizer()
training = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())


# for plotting
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_autoscaley_on(True)
line1, = ax.semilogy([], [], label="train")
line2, = ax.semilogy([], [], label="test")
plt.xlabel("steps / 1"); plt.ylabel("error / 1"); plt.legend()
ax.relim()
ax.autoscale_view()
fig.canvas.draw()
fig.canvas.flush_events()
training_cost = []
validation_cost = []

max_iterations = int(1e5)
delta_convergence = 1e-7

# do actual training and plotting
for i in range(max_iterations):
    # train and calulate costs
    sess.run(training, {x: x_train, y: y_train})
    training_cost.append(sess.run(loss, {x: x_train, y: y_train}))
    validation_cost.append(sess.run(loss, {x: y_test, y: y_test}))

    print(training_cost[-1], validation_cost[-1])

    # visualize 
    line1.set_data(np.arange(0, len(training_cost)), training_cost)
    line2.set_data(np.arange(0, len(training_cost)), validation_cost)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    if i > 1:
        if np.abs(training_cost[-1] - training_cost[-2]) < delta_convergence:
            print("---------------------------------\n")
            print("Iteration stopped after {0} steps.\n".format(i))
            break
#---

print("\n\nFinal training error: {0}\nFinal validation error: {1}".format(
    training_cost[-1],
    validation_cost[-1]
))




