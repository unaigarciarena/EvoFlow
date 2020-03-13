import numpy as np
import tensorflow as tf
from VALP.descriptor import ConvDecoderDescriptor, DecoderDescriptor, GenericDescriptor, DiscreteDescriptor, ConvolutionDescriptor, init_functions, act_functions
import os
import math


class Network:
    def __init__(self, network_descriptor, ident):

        self.id = ident
        self.descriptor = network_descriptor
        self.List_layers = []
        self.List_weights = []

    def reset_network(self):
        self.List_layers = []
        self.List_weights = []


class MLP(Network):
    def __init__(self, descriptor, ident):
        super().__init__(descriptor, ident)
        self.List_bias = []
        self.result = None

    def create_hidden_layer(self, in_size, out_size, init_w_function, layer_name):

        w = tf.Variable(init_w_function(shape=[in_size, out_size]), name="W"+layer_name)
        b = tf.Variable(tf.zeros(shape=[out_size]), name="b"+layer_name)

        self.List_weights.append(w)
        self.List_bias.append(b)

    def initialization(self):
        self.create_hidden_layer(self.descriptor.input_dim, self.descriptor.dims[0], self.descriptor.init_functions[0], str(0))

        for lay in range(1, self.descriptor.number_hidden_layers):
            self.create_hidden_layer(self.descriptor.dims[lay-1], self.descriptor.dims[lay], self.descriptor.init_functions[lay], str(lay))

        self.create_hidden_layer(self.descriptor.dims[self.descriptor.number_hidden_layers-1], self.descriptor.output_dim, self.descriptor.init_functions[self.descriptor.number_hidden_layers], str(self.descriptor.number_hidden_layers))

    def building(self, layer, load):

        if load:
            self.load_weights()
        else:
            self.initialization()

        for lay in range(self.descriptor.number_hidden_layers+1):
            act = self.descriptor.act_functions[lay]
            layer = tf.matmul(layer, self.List_weights[lay]) + self.List_bias[lay]

            if lay in self.descriptor.batch_norm:
                layer = tf.layers.batch_normalization(layer)

            if act is not None and lay < self.descriptor.number_hidden_layers:
                layer = act(layer)

            if lay in self.descriptor.dropout:
                layer = tf.layers.dropout(layer)
            self.List_layers.append(layer)

        self.result = layer

    def variables(self):

        tensors = {}
        for ind, var in enumerate(self.List_bias):
            tensors[self.id + "-B-" + str(ind)] = var
        for ind, var in enumerate(self.List_weights):
            tensors[self.id + "-B-" + str(ind)] = var
        return tensors

    def load_weights(self, path="/home/unai/Escritorio/MultiNetwork/"):
        if os.path.isfile(path + str(self.id) + ".npy"):
            self.List_weights, self.List_bias = np.load(str(self.id) + ".npy")

            for i in range(len(self.List_weights)):
                self.List_weights[i] = tf.Variable(self.List_weights[i])
                self.List_bias[i] = tf.Variable(self.List_bias[i])

                if i == 0:  # If input
                    if self.List_weights[i].shape[0] < self.descriptor.input_dim:  # In case the input size of a network has been increased, we complete the weights of the first layer
                        if "uniform" in self.descriptor.List_init_functions[i].__name__:
                            new = tf.Variable(self.descriptor.List_init_functions[i](shape=[self.descriptor.input_dim-self.List_weights[i].shape[0].value, self.List_weights[i].shape[1].value], minval=-0.1, maxval=0.1), name="W"+str(i))
                        elif "normal" in self.descriptor.List_init_functions[i].__name__:
                            new = tf.Variable(self.descriptor.List_init_functions[i](shape=[self.descriptor.input_dim-self.List_weights[i].shape[0].value, self.List_weights[i].shape[1].value], mean=0, stddev=0.03), name="W"+str(i))
                        else:
                            new = tf.Variable(self.descriptor.List_init_functions[i](shape=[self.descriptor.input_dim-self.List_weights[i].shape[0].value, self.List_weights[i].shape[1].value]), name="W"+str(i))
                        self.List_weights[i] = tf.concat((self.List_weights[i], new), axis=0)

                    elif self.List_weights[i].shape[0] > self.descriptor.input_dim:  # In case the input size of a network has been reduced, we discard part of the weights
                        self.List_weights[i] = self.List_weights[i][:self.descriptor.input_dim, :]

                if i == len(self.List_weights)-1:  # If output

                    if self.List_weights[i].shape[1] < self.descriptor.output_dim:  # If the ouput size of a network has been increased, we complete the last weight matrix and bias vector
                        if "uniform" in self.descriptor.List_init_functions[i].__name__:
                            new = tf.Variable(self.descriptor.List_init_functions[i](shape=[self.List_weights[i].shape[0].value, self.descriptor.output_dim-self.List_weights[i].shape[1].value], minval=-0.1, maxval=0.1), name="W"+str(i))
                        elif "normal" in self.descriptor.List_init_functions[i].__name__:
                            new = tf.Variable(self.descriptor.List_init_functions[i](shape=[self.List_weights[i].shape[0].value, self.descriptor.output_dim-self.List_weights[i].shape[1].value], mean=0, stddev=0.03), name="W"+str(i))
                        else:
                            new = tf.Variable(self.descriptor.List_init_functions[i](shape=[self.List_weights[i].shape[0].value, self.descriptor.output_dim-self.List_weights[i].shape[1].value]), name="W"+str(i))
                        self.List_weights[i] = tf.concat((self.List_weights[i], new), axis=1)
                        self.List_bias[i] = tf.concat((self.List_bias[i], tf.zeros(self.descriptor.output_dim-self.List_bias[i].shape[0].value)), axis=0)

                    elif self.List_weights[i].shape[1] > self.descriptor.output_dim:  # If the output size of a network has been decreased, we discard part of the weights and biases
                        self.List_weights[i] = self.List_weights[i][:, :self.descriptor.output_dim]
                        self.List_bias[i] = self.List_bias[i][:self.descriptor.output_dim]
            self.List_weights = self.List_weights.tolist()
            self.List_bias = self.List_bias.tolist()
        else:
            self.initialization()

    def save_weights(self, sess):
        ws = sess.run(self.List_weights)
        bs = sess.run(self.List_bias)

        np.save(self.id, [ws, bs], allow_pickle=True)


class CNN(Network):

    def __init__(self, network_descriptor, ident):
        super().__init__(network_descriptor, ident)

    def initialization(self):

        last_c = self.descriptor.input_dim[-1]
        for ind, layer in enumerate(self.descriptor.layers):

            if layer == 2:  # If the layer is convolutional
                if self.descriptor.init_fns[ind] == 0:
                    w = tf.Variable(np.random.uniform(-0.1, 0.1, size=[self.descriptor.sizes[ind][0], self.descriptor.sizes[ind][1], last_c, self.descriptor.sizes[ind][2]]).astype('float32'), name="W"+str(ind))
                else:
                    w = tf.Variable(np.random.normal(0, 0.03, size=[self.descriptor.sizes[ind][0], self.descriptor.sizes[ind][1], last_c, self.descriptor.sizes[ind][2]]).astype('float32'), name="W"+str(ind))
                self.List_weights += [tf.Variable(w)]
                last_c = self.descriptor.sizes[ind][2]

            else:  # In case the layer is pooling, no need of weights
                self.List_weights += [tf.Variable(-1)]

    def building(self, layer, graph):

        with graph.as_default():
            for ind, lay in enumerate(self.descriptor.layers):
                if lay == 2:  # If the layer is convolutional
                    layer = tf.nn.conv2d(layer, self.List_weights[ind], (1, self.descriptor.strides[ind], self.descriptor.strides[ind], 1), padding="VALID")
                elif lay == 0:  # If the layer is average pooling
                    layer = tf.nn.avg_pool(layer, (1, self.descriptor.sizes[ind, 0], self.descriptor.sizes[ind, 1], 1), (1, self.descriptor.strides[ind], self.descriptor.strides[ind], 1), "SAME")
                else:
                    layer = tf.nn.max_pool(layer, (1, self.descriptor.sizes[ind, 0], self.descriptor.sizes[ind, 1], 1), (1, self.descriptor.strides[ind], self.descriptor.strides[ind], 1), "SAME")
                if self.descriptor.act_fns[ind] is not None:
                    layer = self.descriptor.act_fns[ind](layer)

                self.List_layers += [layer]

        return layer

    def conv_save_weights(self, sess, name=""):
        ws = sess.run(self.List_weights)

        np.save(name + str(self.id), ws, allow_pickle=True)

    def conv_load_weights(self, path="/home/unai/Escritorio/MultiNetwork/"):
        if os.path.isfile(path + str(self.id) + ".npy"):
            self.List_weights = np.load(str(self.id) + ".npy")
            for i in range(len(self.List_weights)):
                self.List_weights[i] = tf.Variable(self.List_weights[i])
        else:
            self.initialization()

# ################  The following classes contain the necessary tools to implement the different networks. In tensorflow level


class Decoder(MLP):

    def __init__(self, descriptor, ident):
        super().__init__(descriptor, ident)
        self.z_log_sigma_sq = None
        self.z_mean = None

    def building(self, inputs, load):

        z = []
        cond = []

        for i in range(len(inputs)):
            if i in self.descriptor.rands:  # This contains the indices of the decoder inputs that are deleted when sampling
                z += [tf.layers.flatten(inputs[i])]
            else:
                cond += [tf.layers.flatten(inputs[i])]

        z = tf.concat(z, axis=1)  # This contains the input that will be deleted when sampling
        self.z_log_sigma_sq = z[:, :math.floor(z.shape[1].value/2)]
        self.z_log_sigma_sq = tf.clip_by_value(self.z_log_sigma_sq, clip_value_min=-10, clip_value_max=10)

        self.z_mean = z[:, math.ceil(z.shape[1].value/2):]

        eps = tf.random_normal(shape=tf.shape(self.z_mean), mean=0, stddev=1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        z_samples = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        z_samples_cond = tf.concat([z_samples] + cond, axis=1)
        self.descriptor.input_dim = z_samples_cond.shape[1].value

        super().initialization()

        super().building(z_samples_cond, load)


class ConvDecoder(Network):  # Not tested, code is very similar to common decoder
    def __init__(self, descriptor, ident):

        super().__init__(descriptor, ident)


class Generic(MLP):
    def __init__(self, descriptor, ident):
        super().__init__(descriptor, ident)
        self.x = None

    def building(self, inputs, load):

        for inpt in range(len(inputs)):
            inputs[inpt] = tf.layers.flatten(inputs[inpt])
        self.x = tf.concat(inputs, axis=1)

        self.descriptor.input_dim = self.x.shape[1].value

        super().initialization()

        super().building(self.x, load)


class Discrete(MLP):
    def __init__(self, descriptor, ident):
        super().__init__(descriptor, ident)
        self.x = None

    def building(self, inputs, load):

        for inpt in range(len(inputs)):
            inputs[inpt] = tf.layers.flatten(inputs[inpt])
        self.x = tf.concat(inputs, axis=1)

        self.descriptor.input_dim = self.x.shape[1].value

        super().building(self.x, load)


class CNN(object):
    def __init__(self, descriptor, inp, ident, load):
        self.descriptor = descriptor
        self.network = None
        self.inp = inp
        max_shape = [0, 0, 0]

        for inpt in inp:
            max_shape[0] = max(max_shape[0], inpt.shape[1].value)
            max_shape[1] = max(max_shape[1], inpt.shape[2].value)
            max_shape[2] = max(max_shape[2], inpt.shape[3].value)

        for inpt in range(len(inp)):
            pad = np.zeros((4, 2))
            for dim in range(1, 3):
                pad[dim, 0] = math.floor((max_shape[dim-1]-inp[inpt].shape[dim].value)/2)
                pad[dim, 1] = math.ceil((max_shape[dim-1]-inp[inpt].shape[dim].value)/2)
            inp[inpt] = tf.pad(inp[inpt], pad)

        self.inp = tf.concat(self.inp, axis=3)
        self.in_ch = self.inp.shape[-1].value
        self.result = self.create_network(ident, load)

    def create_network(self, ident, load):
        self.network = Network(self.descriptor.network, ident, self.in_ch)

        if load:
            self.network.conv_load_weights()
        else:
            self.network.convolutional_initialization()
        return self.network.convolutional_evaluation(self.inp)

    def variables(self):
        return self.network.variables()

    def save_weights(self, sess):
        self.network.conv_save_weights(sess)


def random_batch(x, y, size):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param y: Fitness scores of the population, intended to be fed to the net in the output
    :param size: Size of the batch desired
    :return: A random batch of the data (x, y) of the selected size
    """
    indices = np.random.randint(x.shape[0], size=size)
    return x[indices, :], y[indices]


def batch(x, size, i):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param size: Size of the batch desired
    :param i: Index of the last solution used in the last epoch
    :return: The index of the last solution in the batch (to be provided to this same
             function in the next epoch, the solutions in the actual batch, and their
             respective fitness scores
    """

    if i + size > x.shape[0]:  # In case there are not enough solutions before the end of the array
        index = i + size-x.shape[0]  # Select all the individuals until the end and restart
        return np.concatenate((x[i:, :], x[:index, :]))
    else:  # Easy case
        index = i+size
        return x[i:index, :]


# ################### Random descriptor initializers
# These functions return random network descriptors


def decoder_descriptor(z_dim, x_dim):

    dim_list = np.random.randint(4, 50, np.random.randint(1, 5))

    n_hidden = len(dim_list)

    decoder_init_functions = np.random.choice(init_functions, size=n_hidden+1)   # List or random init functions for decoder

    decoder_act_functions = np.random.choice(act_functions, size=n_hidden+1)
    decoder_act_functions[n_hidden] = None

    return DecoderDescriptor(z_dim, x_dim, 0, n_hidden, dim_list, decoder_init_functions,  decoder_act_functions, [[], []])


def generic_descriptor(input_dim, output_dim):

    dim_list = np.random.randint(4, 50, np.random.randint(1, 5))

    n_hidden = len(dim_list)

    generic_init_functions = np.random.choice(init_functions, size=n_hidden+1)   # List or random init functions for encoder
    generic_act_functions = np.random.choice(act_functions, size=n_hidden+1)
    generic_act_functions[n_hidden] = None

    return GenericDescriptor(input_dim, output_dim, 0, n_hidden, dim_list, generic_init_functions,  generic_act_functions, 0)


def convolutional_descriptor(input_dim, output_dim, previous, model_ins):

    layers = np.random.choice([0, 1, 2], size=2)  # Number of layers = 2

    filters = [np.random.randint(1, 5) if layers[i] == 2 else -1 for i in range(layers.shape[0])]
    init = [np.random.randint(0, 2) if layers[i] == 2 else -1 for i in range(layers.shape[0])]
    act = [np.random.randint(0, 2) if layers[i] == 2 else -1 for i in range(layers.shape[0])]

    sizes = np.concatenate((np.random.randint(2, 4, size=(layers.shape[0], 2)), np.random.randint(1, 3, size=(layers.shape[0], 1))), axis=1)
    strides = np.random.randint(1, 2, size=layers.shape[0])

    return ConvolutionDescriptor(input_dim, output_dim, 0, layers, filters, strides, sizes, act, init, 0)


def conv_dec_descriptor(in_dim, out_dim):

    layers = np.random.randint(2, 10)
    filters = []
    strides = []

    outs = [out_dim]

    for i in range(layers-1):
        size = np.random.randint(2, 7)
        in_ch = np.random.randint(1, 7)
        str_cands = [i for i in range(2, out_dim[0]) if out_dim[0] % i == 0]
        if len(str_cands) == 0:
            break
        stride = np.random.choice(str_cands[:3])
        strides.insert(0, (1, stride, stride, 1))
        filters.insert(0, (size, size, out_dim[2], in_ch))
        out_dim = (out_dim[0]//stride, out_dim[1]//stride, in_ch)
        outs.insert(0, out_dim)

    layers = len(filters)

    init = np.random.randint(0, 2, layers)
    act = np.random.randint(0, 7, layers+1)

    return ConvDecoderDescriptor(in_dim, out_dim, 0, layers, filters, strides, act, init, outs, [[], []])


def discrete_descriptor(input_dim, output_dim):

    dim_list = np.random.randint(4, 50, np.random.randint(1, 5))

    n_hidden = len(dim_list)

    generic_init_functions = np.random.choice(init_functions, size=n_hidden+1)   # List or random init functions for encoder
    generic_act_functions = np.random.choice(act_functions, size=n_hidden+1)
    generic_act_functions[n_hidden] = tf.nn.softmax

    return DiscreteDescriptor(input_dim, output_dim, 0, n_hidden, dim_list, generic_init_functions,  generic_act_functions, 0)
