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

        #todo: maybe a property for input too.
        self.input_tensor = None

    def __str__(self):

        if len(self.structure) == 2:
            structure = str(self.structure[-1])
        else:
            structure = "x".join(map(str, self.structure[1:-1]))

        return self._name_string + structure
    
    @property
    def output_tensor(self):
        if self._graph is None:
            self.setup()

        return self._graph

    def weights_values(self, session):
        if self._graph is None:
            raise RuntimeError("Notwork not initialized!")
        return [session.run(w) for w in self.weights]
    
    def biases_values(self, session):
        if self._graph is None:
            raise RuntimeError("Notwork not initialized!")
        return [session.run(b) for b in self.biases]

    def run(self, session, inputs):
        """Evaluate the neural network"""
        session.run(self._graph, feed_dict={self.input_tensor: inputs})

    def setup(self):

        # set up input placeholder    
        self.input_tensor = tf.placeholder(
                dtype="float32", shape=[None, self.structure[0]]
            )
    
        # set up input layer
        with tf.name_scope("input_layer"):
            self._graph = self.input_tensor

        # hidden layers
        if len(self.structure) > 2:
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

    def _add_layer(self, x, dim_in, dim_out, name="hidden_layer", **kwargs):
        
        with tf.name_scope(name):
            w = tf.Variable(
                self._initialization([dim_in, dim_out], **kwargs), 
                name="w"
            )
            b = tf.Variable(
                self._initialization([dim_out], **kwargs), 
                name="b"
            )

            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)

            act = self._activation(self._preactivation(x, w, b))

            tf.summary.histogram("activations", act)

            return act, w, b


    def _add_output_layer(self, x, dim_in, dim_out, **kwargs):
        
        with tf.name_scope("output_layer"):

            w = tf.Variable(
                self._initialization([dim_in, dim_out], **kwargs), 
                name="w"
            )
            b = tf.Variable(
                self._initialization([dim_out], **kwargs), 
                name="b"
            )

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

class FixedValueNN(AbstractNeuralNetwork):
    """This neural network is initialized by values not by specifying the 
    distribution.
    """


    def __init__(self, structure, weights, biases, log_histograms=False):
        """Ctor

        Args:
            - structure <list<int>>: list of number of nodes. First used for 
            input, last element for output.
            - weights <list<np.array<float>>>: list of values to initialize 
            weights by.
            - biases <list<np.array<float>>>: list of values to initialize 
            biases by.
        """

        super(FixedValueNN, self).__init__(structure, log_histograms)

        self._weight_values = weights
        self._biases_values = biases

        self._name_string += "Loaded_Model_"

    def setup(self):

        # set up input placeholder    
        self.input_tensor = tf.placeholder(
                dtype="float32", shape=[None, self.structure[0]]
            )
    
        # set up input layer
        with tf.name_scope("input_layer"):
            self._graph = self.input_tensor


        # hidden layers
        for layer in range(1, len(self.structure) - 1):
            self._graph, w, b = self._add_layer(
                self._graph, 
                self.structure[layer - 1], 
                self.structure[layer],
                layer=layer
            )
            self.weights.append(w)
            self.biases.append(b)    

        # output layer
        self._graph, w, b = self._add_output_layer(
            self._graph,
            self.structure[-2],
            self.structure[-1],
            layer=-1
        )
        self.weights.append(w)
        self.biases.append(b)

        return self._graph

    def _initialization(self, shape, **kwargs):
        if len(shape) == 1:
            return self._biases_values[kwargs["layer"]]
        elif len(shape) == 2:
            return self._weight_values[kwargs["layer"]]


class EluFixedValue(FixedValueNN):
    def __init__(self, *args, **kwargs):
        super(EluFixedValue, self).__init__(*args, **kwargs)
        self._name_string = "Elu_" + self._name_string

    def _activation(self, preactivation):
        return tf.nn.elu(preactivation)    

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