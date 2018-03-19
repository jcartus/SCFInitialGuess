"""This module will contain everything needed to train a neural Network.

Authors:
 - Johannes Cartus, QCIEP, TU Graz
"""

from os.path import join
from uuid import uuid4

import tensorflow as tf
import numpy as np

from SCFInitialGuess.utilities.usermessages import Messenger as msg

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
        

def mse_with_l2_regularisation(
        network, 
        expectation_tensor,
        regularisation_parameter=0.001
    ):
    
    with tf.name_scope("mse_with_l2_regularisation"):

        error = tf.losses.mean_squared_error(
            network.output_tensor, 
            expectation_tensor
        )

        regularisation = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(regularisation_parameter),
            network.weights
        )

        cost = error + regularisation

        tf.summary.scalar("weight_decay", regularisation)
        tf.summary.scalar("error", error)
        tf.summary.scalar("total_loss", cost)

    return cost, error, regularisation

class MSE(object):

    def function(self, network, y_placeholder):

        error = tf.losses.mean_squared_error(
            network.output_tensor,
            y_placeholder
        )

        cost = error
        
        tf.summary.scalar("error", error)
        
        return error

class RegularizedMSE(MSE):

    def __init__(self, alpha=0.001):

        self.alpha = alpha

    def function(self, network, y_placeholder):

        error = super(RegularizedMSE, self).function(network, y_placeholder)

        regularisation = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(self.alpha),
            network.weights
        )

        cost = error + regularisation

        tf.summary.scalar("weight_decay", regularisation)
        tf.summary.scalar("total_loss", cost)

        return cost




class Trainer(object):

    def __init__(
        self, 
        network,
        optimizer=None,
        error_function=None,
        cost_function=None
    ):

        self.network = network

        if optimizer is None:
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=0.001
            )
        else:
            self.optimizer = optimizer

        if cost_function is None:
            self.cost_function = RegularizedMSE()
        else:
            self.cost_function = cost_function
        
        if error_function is None:
            self.error_function = MSE()
        else:
            self.error_function = error_function


        self.training_step = None
        self.test_error = None

    def setup(self, target_graph=None):

        if target_graph is None:
            msg.info("No target graph specified for Trainer setup. " + \
                "Creating new graph ...", 1)
            self.graph = tf.Graph()
        else:
            msg.info("Appending to graph: " + str(target_graph))

        
        with self.graph.as_default():
            
            msg.info("Setting up the training in the target graph ...", 1)

            # placeholder for dataset target-values
            self.target_placeholder = tf.placeholder(
                dtype="float32", 
                shape=[None, self.network.structure[-1]],
                name="y"
            )

            msg.info("network ...", 1)
            with tf.name_scope("network/"):
                network_output = self.network.setup()
                self.input_placeholder = self.network.input_tensor

            msg.info("error function ...", 1)
            with tf.name_scope("error_function/"):
                self.error = self.error_function.function(
                    self.network, 
                    self.target_placeholder
                )
                
            msg.info("cost function ...", 1)
            with tf.name_scope("cost_function/"):
                self.cost = self.cost_function.function(
                    self.network, 
                    self.target_placeholder
            )

            msg.info("training step", 1)
            with tf.name_scope("training/"):
                self.training_step = self.optimizer.minimize(self.cost)

        return self.graph, self.network, self.target_placeholder
    
    
    def train(
            self,
            dataset,
            max_steps=100000,
            evaluation_period=200,
            mini_batch_size=0.2,
            convergence_threshold=1e-5,
            summary_save_path=None
        ):

        sess = tf.Session(graph=self.graph)

        if self.training_step is None:
            self.setup()


        #--- prep the writer ---
        if not summary_save_path is None:
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(summary_save_path)
            writer.add_graph(sess.graph)
        #---

        #--- train the network ---
        old_error = 1e10

        sess.run(tf.global_variables_initializer())


        msg.info("Starting network training ...", 1)        
        for step in range(max_steps):
            mini_batch = dataset.sample_minibatch(mini_batch_size)

            if step % np.ceil(evaluation_period / 10):
                if not summary_save_path is None:
                    writer.add_summary(
                        sess.run(
                            summary, 
                            feed_dict={
                                self.input_placeholder: mini_batch[0], 
                                self.target_placeholder: mini_batch[1]
                            }
                        ), 
                        step
                    )

            if step % evaluation_period == 0:
                error = sess.run(
                    self.error,
                    feed_dict={
                        self.input_placeholder: dataset.validation[0], 
                        self.target_placeholder: dataset.validation[1]
                    }
                )

                # compare to previous error
                diff = np.abs(error - old_error)

                # convergence check
                if diff < convergence_threshold:
                    msg.info(
                        "Convergence reached after " + str(step) + " steps.", 
                        1
                    )

                    break
                else:
                    msg.info(
                        "Validation cost: {:0.5E}. Diff to prev.: {:0.1E}".format(
                            error,
                            diff
                        )
                    )

                    old_error = error
                

            # do training step
            sess.run(
                self.training_step, 
                feed_dict={
                    self.input_placeholder: mini_batch[0], 
                    self.target_placeholder: mini_batch[1]
                }
            )
        #---

        if not summary_save_path is None:
            writer.close()


        test_error = sess.run(
            self.error,
            feed_dict={
                self.input_placeholder: dataset.testing[0], 
                self.target_placeholder: dataset.testing[1]
            }
        )

        self.test_error = test_error

        msg.info("Test error: {:0.5E}".format(test_error), 1)

        return self.network, sess


def train_network(
    network,
    dataset,
    sess=None,
    learning_rate=0.001,
    regularisation_parameter=0.01,
    max_steps=100000,
    evaluation_period=200,
    mini_batch_size=0.2,
    convergence_threshold=1e-5,
    summary_save_path=None
    ):
    """Train a neural Neutwork from nn.networks with the AdamOptimizer,
    to minimize the mean squared error with l2 regularisation.

    Args:
        - network <nn.networks.AbstractNeuralNetwork>: the network to be trained.
        - dataset <utilities.dataset.Dataset>: the dataset to train the net on.
        - learning_rate <float>: the learning rate to use for training w/
        AdamOptimizer
        - regularisation_parameter <float>: the factor with which the 
        regularisation is added to the total cost.
        - max_steps <int>: max number of learning steps to take if convergence 
        not met before.
        - evaluation_period <int>: period of training steps after which there
        will be a check for convergence.
        mini_batch_size <int>: size of the minibatch that is randomly sampled 
        from the training dataset in every training step.
        - convergence_threshold <float>: training convergence is reached if 
        difference in error drops below this value.
        - summary_save_path <str>: the full path to a folder in which the 
        tensorboard data will be written. If None given nothing will be exported.

    Returns:
        - the trained network
        - the session
    """

    if sess is None:
        sess = tf.Session()

    #--- set up the graph ---
    msg.info("Setting up the graph ...", 1)
    network_output = network.setup()
    x = network.input_tensor
    y = tf.placeholder(
            dtype="float32", 
            shape=[None, network.structure[-1]],
            name="y"
        )


    # cost is mse w/ l2 regularisation
    cost, mse, _ = mse_with_l2_regularisation(
        network,
        expectation_tensor=y,
        regularisation_parameter=regularisation_parameter
    )

    #optimizer and training
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = optimizer.minimize(cost) 
    #---

    #--- prep the writer ---
    if not summary_save_path is None:
        msg.warn("Careful! If more than 1 network is in current graph, " + \
            "it should be cleared before merging the summary!"
        )
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(summary_save_path)
        writer.add_graph(sess.graph)
    #---

    #--- train the network ---
    msg.info("Starting network training ...", 1)
    old_error = 1e10

    sess.run(tf.global_variables_initializer())

    for step in range(max_steps):
        mini_batch = dataset.sample_minibatch(mini_batch_size)

        if step % np.ceil(evaluation_period / 10) == 0:
            if not summary_save_path is None:
                writer.add_summary(
                    sess.run(
                        summary, 
                        feed_dict={
                            x: mini_batch[0], 
                            y: mini_batch[1]
                        }
                    ), 
                    step
                )

        if step % evaluation_period == 0:
            error = sess.run(
                mse,
                feed_dict={x: dataset.validation[0], y: dataset.validation[1]}
            )

            # compare to previous error
            diff = np.abs(error - old_error)

            # convergence check
            if diff < convergence_threshold:
                msg.info(
                    "Convergence reached after " + str(step) + " steps.", 
                    1
                )

                break
            else:
                msg.info(
                    "Validation cost: {:0.5E}. Diff to prev.: {:0.1E}".format(
                        error,
                        diff
                    )
                )

                old_error = error
            

        # do training step
        sess.run(train_step, feed_dict={x: mini_batch[0], y: mini_batch[1]})
    #---

    if not summary_save_path is None:
        writer.close()

    test_error = sess.run(
        mse,
        feed_dict={x: dataset.testing[0], y: dataset.testing[1]}
    )
    msg.info("Test error: {:0.5E}".format(test_error), 1)


    return network, sess

def network_benchmark(
        models, 
        dataset, 
        logdir, 
        steps_report=250,
        max_training_steps=100000,
        convergence_eps=1e-7
    ):

    for model in models:

        msg.info("Investigating model " + str(model), 2)

        save_path = join(logdir, str(model))
        
        # make new session and build graph
        tf.reset_default_graph()
        sess = tf.Session()

        dim_in = model.network.structure[0]
        dim_out = model.network.structure[-1]

        f = model.network.setup()
        x = model.input_tensor
        y = tf.placeholder(tf.float32, shape=[None, dim_out])
        

        with tf.name_scope("loss"):
            error = tf.losses.mean_squared_error(y, f) / dim_out # sum_i (f8(x_i) - y_i)^2
            weight_decay = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(0.001),
                model.network.weights
            )
            loss = error +  weight_decay

            tf.summary.scalar("weight_decay", weight_decay)
            tf.summary.scalar("error_per_element", error)
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
            if step % 50 == 0:
                writer.add_summary(sess.run(
                    summary, 
                    feed_dict={x: batch[0], y: batch[1]}
                ), step)

            # save graph and report error
            if step % steps_report == 0:
                validation_error = sess.run(
                    error, 
                    feed_dict={x: dataset.validation[0], y: dataset.validation[1]}
                ) / dim_out
                #saver.save(sess, log_dir, step)

                diff = np.abs(old_error - validation_error)
                msg.info("Error: {:0.4E}. Diff to before: {:0.4E}".format(
                    validation_error,
                    diff
                ))
                if diff < convergence_eps:
                    msg.info(
                        "Convergence reached after " + str(step) + " steps.", 1
                    )
                    break
                else:
                    old_error = validation_error
            
            if step + 1 == max_training_steps:
                msg.info("Max iterations exceeded.", 1)

            sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
            
        test_error = sess.run(
            error, 
            feed_dict={x: dataset.validation[0], y: dataset.validation[1]}
        ) / dim_out
        msg.info("Test error: {:0.1E}".format(test_error))




