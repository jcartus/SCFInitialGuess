"""This module contains all the neural network components, needed to
set up and train a neural network.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

import tensorflow as tf

class AbstractNeuralNetwork(object):
    """This is an abstract template for a neural network. Components like
    the activation function or the initialization method can be replace
    individually.
    """

    def __init__(self, structure, log_histograms=False):
        self.structure = structure

        self._log_histograms = log_histograms

        self._graph = None
        self.weights = []
        self.biases = []

        self._name_string = ""

    def __str__(self):

        if len(self.structure) == 2:
            structure = str(self.structure[-1])
        else:
            structure = "x".join(map(str, self.structure[1:-1]))

        return self._name_string + structure
    def run(self, session):
        """Evaluate the neural network"""
        session.run(self._graph)

    def setup(self, input_tensor):
        
        # set up input layer
        with tf.name_scope("input_layer"):
            self._graph = input_tensor


        # hidden layers
        for layer in range(1, len(self.structure) - 1):
            self._graph, w, b = self._add_layer(
                self._graph, 
                self.structure[layer - 1], 
                self.structure[layer],
            )
            self.weights.append(w)
            self.biases.append(b)    

        # output layer
        self._graph, w, b = self._add_output_layer(
            self._graph,
            self.structure[-2],
            self.structure[-1]
        )
        self.weights.append(w)
        self.biases.append(b)

        return self._graph

    def _add_layer(self, x, dim_in, dim_out, name="hidden_layer"):
        
        with tf.name_scope(name):
            w = tf.Variable(self._initialization([dim_in, dim_out]), name="w")
            b = tf.Variable(self._initialization([dim_out]), name="b")

            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)

            act = self._activation(self._preactivation(x, w, b))

            tf.summary.histogram("activations", act)

            return act, w, b


    def _add_output_layer(self, x, dim_in, dim_out):
        
        with tf.name_scope("output_layer"):

            w = tf.Variable(self._initialization([dim_in, dim_out]), name="w")
            b = tf.Variable(self._initialization([dim_out]), name="b")

            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)

            out = self._preactivation(x, w, b)

            tf.summary.histogram("outputs", out)

            return out, w, b


    def _initialization(self, shape, **kwargs):
        raise NotImplementedError("This is just the abstract class!")
        
    def _preactivation(self, x, w, b):
        return tf.matmul(x, w) + b

    def _activation(self, preactivation):
        raise NotImplementedError("This is just an abstract class")


class TruncatedNormalNN(AbstractNeuralNetwork):
    """This is a Neural Network with weights/biases initialized truncated normal.
    """

    def __init__(self, structure, log_histograms=False, mu=0, std=0.01):
        """Ctor

        Args:
            - structure <list<int>>: list of number of nodes. First used for 
            input, last element for output.
            - mu <float>: mean used for weights/bias initialisation
            - sistdgma <float>: standard deviation used for weight/bias 
            initialisation
        """

        super(TruncatedNormalNN, self).__init__(structure, log_histograms)

        self.mu = mu
        self.std = std
        self._name_string += "TrN_mu-{0}_std-{1}_Elu_".format(mu, std)

    def _initialization(self, shape, **kwargs):
        return tf.truncated_normal(
            shape=shape, 
            mean=self.mu,
            stddev=self.std,
            **kwargs
        )

class EluTrNNN(TruncatedNormalNN):
    """This is a Neural Network with weights/biases initialized truncated normal
    and the activations being elus
    """

    def __init__(self, *args, **kwargs):
        super(EluTrNNN, self).__init__(*args, **kwargs)
        self._name_string = "Elu_" + self._name_string

    def _activation(self, preactivation):
        return tf.nn.elu(preactivation)

class ReluTrNNN(TruncatedNormalNN):
    """This is a Neural Network with weights/biases initialized truncated normal
    and the activations being Relus
    """
    def __init__(self, *args, **kwargs):
        super(ReluTrNNN, self).__init__(*args, **kwargs)
        self._name_string = "Relu_" + self._name_string

    def _activation(self, preactivation):
        return tf.nn.relu(preactivation)