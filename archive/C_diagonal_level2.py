from os import walk
from os.path import normpath, join

import numpy as np
import tensorflow as tf

from utilities.constants import number_of_basis_functions as N_BASIS
from utilities.dataset import Result, normalize, assemble_batch, split_dataset, \
    random_subset_by_fraction
from utilities.usermessages import Messenger as msg
from nn.networks import TruncatedNormalEluNN

project_dir = normpath("/home/jcartus/Documents/SCFInitialGuess/")
database = join(project_dir, "dataset/")
log_dir = join(project_dir, "log/")
#db_names = ["a24", "ala27", "s22", "l7", "p76", "shbc"]
db_names = ["a24"]

#--- create and preprocess a dataset for C ---
folders = [join(database, name) for name in db_names]
x_raw, y_raw, _, _ = assemble_batch(folders, "C")

x_train, y_train, x_test, y_test = split_dataset(x_raw, y_raw)
#---

#--- train network ---
dim = N_BASIS['C']

tf.reset_default_graph()
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, dim])
y = tf.placeholder(tf.float32, shape=[None, dim])

network = TruncatedNormalEluNN([dim, 40,40, 40, dim], log_histograms=True)
f = network.setup(x)

# define loss
with tf.name_scope("optimization"):
    error = tf.losses.mean_squared_error(y, f) # sum_i (f8(x_i) - y_i)^2
    weight_decay = tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(0.1),
        network.weights
    )
    weight_decay = tf.nn.l2_loss(network.weights[1]) # todo all weights should be regularized
    loss = error #+ weight_decay

    tf.summary.scalar("error", loss)
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(
        learning_rate=0.04,
        beta1=0.9,
        beta2=0.999
    ).minimize(loss)

summ = tf.summary.merge_all()
#---

#--- do training ---
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
writer = tf.summary.FileWriter(join(log_dir, "test4"))
writer.add_graph(sess.graph)



msg.info("Starting  traing ...", 2)
for i in range(10000):
    batch = random_subset_by_fraction(x_train, y_train, 0.1)

    # write down summary
    if i % 10 == 0:
        s = sess.run(summ, feed_dict={x: batch[0], y: batch[1]})
        writer.add_summary(s, i)

    # save graph and report error
    if i % 200 == 0:
        validation_error = sess.run(error, feed_dict={x: x_test, y: y_test})
        saver.save(sess, log_dir, i)
        msg.info("Validation error: " + str(validation_error))

    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})






