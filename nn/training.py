"""This module will contain everything needed to train a neural Network.

Authors:
 - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import join

import tensorflow as tf
import numpy as np

from utilities.usermessages import Messenger as msg

class CustomAdam(tf.train.AdamOptimizer):
    """A custom verison of tensorflow's AdamOptimizer. The only
    difference is the new __str__ method, to get a neatly prinable
    representation"""


    def __init__(self, *args, **kwargs):
    
        super(CustomAdam, self).__init__(*args, **kwargs)

    @property
    def print_name(self):
        return "Adam_eta-{0}_b1-{1}_b2-{2}".format(
            self._lr, self._beta1, self._beta2
        )
    
    

class Model(object):

    def __init__(self, name,network, optimizer):

        self.name = name 
        self.network = network
        self.optimizer = optimizer

    def __str__(self):
        return self.name  + "_" + str(self.network) + "_" + self.optimizer.print_name
        


def network_benchmark(models, dataset, logdir, max_training_steps=100000):

    for model in models:

        msg.info("Investigating model " + model.name, 1)

        print(str(model))
        save_path = join(logdir, str(model))
        
        # make new session and build graph
        tf.reset_default_graph()
        sess = tf.Session()

        dim_in = model.network.structure[0]
        dim_out = model.network.structure[-1]

        x = tf.placeholder(tf.float32, shape=[None, dim_in])
        y = tf.placeholder(tf.float32, shape=[None, dim_out])
        f = model.network.setup(x)

        with tf.name_scope("loss"):
            error = tf.losses.mean_squared_error(y, f) # sum_i (f8(x_i) - y_i)^2
            weight_decay = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(0.1),
                model.network.weights
            )
            loss = error +  weight_decay

            tf.summary.scalar("error", error)
            tf.summary.scalar("total_loss", loss)

        # define loss
        with tf.name_scope("train"):
            train_step = model.optimizer.minimize(loss)

        summary = tf.summary.merge_all()
        #saver = tf.train.Saver()
        writer = tf.summary.FileWriter(save_path)
        writer.add_graph(sess.graph)

        msg.info("Start training ... ", 1)
        old_error = 1e13

        sess.run(tf.global_variables_initializer())

        for step in range(max_training_steps):
            batch = dataset.sample_minibatch(0.2) 

            # log progress
            if step % 10 == 0:
                writer.add_summary(sess.run(
                    summary, 
                    feed_dict={x: batch[0], y: batch[1]}
                ))

            # save graph and report error
            if step % 200 == 0:
                validation_error = sess.run(
                    error, 
                    feed_dict={x: dataset.validation[0], y: dataset.validation[1]}
                )
                #saver.save(sess, log_dir, step)

                diff = np.abs(old_error - validation_error)
                msg.info("Error difference to prev. step: {:0.4E}".format(diff))
                if diff < 1e-5:
                    msg.info(
                        "Convergence reached after " + str(step) + " steps.", 1
                    )
                else:
                    old_error = validation_error
            
            if step + 1 == max_training_steps:
                msg.info("Max iterations exceeded.", 1)

            sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
            
        test_error = sess.run(
            error, 
            feed_dict={x: dataset.validation[0], y: dataset.validation[1]}
        )
        msg.info("Test error: " + str(test_error))




