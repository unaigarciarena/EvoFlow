import tensorflow as tf
import numpy as np
from functools import reduce
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def xavier_init(fan_in=None, fan_out=None, shape=None):
    """ Xavier initialization of network weights"""
    if fan_in is None or shape is not None:
        fan_in = shape[0]
        fan_out = shape[1]
    # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    lim = np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-lim, maxval=lim, dtype=tf.float32)


activations = [None, tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh]
initializations = [xavier_init, tf.random_uniform, tf.random_normal]


class NetworkDescriptor:

    def __init__(self, number_hidden_layers=1, input_dim=1, output_dim=1, init_functions=None, act_functions=None, dropout=(), dropout_probs=(), batch_norm=()):
        """
        This class implements the descriptor of a generic network. Subclasses of this are the ones evolved.
        :param number_hidden_layers: Number of hidden layers in the network
        :param input_dim: Dimension of the input data (can have one or more dimensions)
        :param output_dim: Expected output of the network (similarly, can have one or more dimensions)
        :param init_functions: Weight initialization functions. Can have different ranges, depending on the subclass
        :param act_functions: Activation functions to be applied after each layer
        :param dropout: A 0-1 array of the length number_hidden_layers indicating  whether a dropout "layer" is to be
        applied AFTER the activation function
        :param batch_norm: A 0-1 array of the length number_hidden_layers indicating  whether a batch normalization
        "layer" is to be applied BEFORE the activation function
        """
        self.number_hidden_layers = number_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_functions = init_functions
        self.act_functions = act_functions
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_probs = dropout_probs

    def remove_layer(self, _):  # Defined just in case the user redefines classes and forgets to define this function
        pass

    def change_dimensions_in_layer(self, _, __):  # Defined just in case the user redefines classes and forgets to define this function
        pass

    def remove_random_layer(self):
        layer_pos = np.random.randint(self.number_hidden_layers)
        self.remove_layer(layer_pos)

    def change_activation(self, layer_pos, new_act_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.act_functions[layer_pos] = new_act_fn

    def change_weight_init(self, layer_pos, new_weight_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.init_functions[layer_pos] = new_weight_fn

    def change_dimensions(self, max_layer_size):
        # Select random layer and execute change
        layer_pos = np.random.randint(self.number_hidden_layers)
        new_dim = np.random.randint(max_layer_size)+1
        self.change_dimensions_in_layer(layer_pos, new_dim)

    def change_dropout(self):
        # Select random layer and flip dropout
        rnd = np.random.choice(np.arange(0, self.dropout.shape[0]), size=np.random.randint(0, self.dropout.shape[0]), replace=False)
        self.dropout[rnd] -= 1
        self.dropout[rnd] = self.dropout[rnd]**2

    def change_batch_norm(self):
        # Select random layer and flip batch normalization
        rnd = np.random.choice(np.arange(0, self.batch_norm.shape[0]), size=np.random.randint(0, self.batch_norm.shape[0]), replace=False)
        self.batch_norm[rnd] -= 1
        self.batch_norm[rnd] = self.batch_norm[rnd]**2


class MLPDescriptor(NetworkDescriptor):
    def __init__(self, number_hidden_layers=1, input_dim=1, output_dim=1,  dims=None, init_functions=None, act_functions=None, dropout=(), batch_norm=()):
        """

        :param number_hidden_layers: Number of hidden layers in the network
        :param input_dim: Dimension of the input data (can have one or more dimensions)
        :param output_dim: Expected output of the network (similarly, can have one or more dimensions)
        :param init_functions: Weight initialization functions. Can have different ranges, depending on the subclass
        :param dims: Number of neurons in each layer
        :param act_functions: Activation functions to be applied after each layer
        :param dropout: A 0-1 array of the length number_hidden_layers indicating  whether a dropout "layer" is to be
        applied AFTER the activation function
        :param batch_norm: A 0-1 array of the length number_hidden_layers indicating  whether a batch normalization
        "layer" is to be applied BEFORE the activation function
        """

        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, init_functions=init_functions, act_functions=act_functions, dropout=dropout, batch_norm=batch_norm)
        self.dims = dims  # Number of neurons in each layer

    def random_init(self, input_size=None, output_size=None, nlayers=None, max_layer_size=None, _=None, __=None, no_drop=None, no_batch=None):

        # If the incoming/outgoing sizes have more than one dimension compute the size of the flattened sizes
        if input_size is not None:
            if hasattr(input_size, '__iter__'):
                self.input_dim = reduce(lambda x, y: x*y, input_size)
            else:
                self.input_dim = input_size
        if output_size is not None:
            if hasattr(output_size, '__iter__'):
                self.output_dim = reduce(lambda x, y: x*y, output_size)
            else:
                self.output_dim = output_size

        # Random initialization
        if nlayers is not None and max_layer_size is not None:
            self.number_hidden_layers = np.random.randint(nlayers)+1
            self.dims = [np.random.randint(4, max_layer_size)+1 for _ in range(self.number_hidden_layers)]
            self.init_functions = np.random.choice(initializations, size=self.number_hidden_layers+1)
            self.act_functions = np.random.choice(activations, size=self.number_hidden_layers+1)
        if no_batch is not None:
            if no_batch:
                self.batch_norm = np.zeros(self.number_hidden_layers+1)
            else:
                self.batch_norm = np.random.randint(0, 2, size=self.number_hidden_layers+1)
        if no_drop is not None:
            if no_drop:
                self.dropout = np.zeros(self.number_hidden_layers+1)
                self.dropout_probs = np.zeros(self.number_hidden_layers+1)
            else:
                self.dropout = np.random.randint(0, 2, size=self.number_hidden_layers+1)
                self.dropout_probs = np.random.rand(self.number_hidden_layers+1)

    def add_layer(self, layer_pos, lay_dims, init_w_function, init_a_function, dropout, drop_prob, batch_norm):
        """
        This function adds a layer in the layer_pos position
        :param layer_pos: Position of the layer
        :param lay_dims: Number of neurons in the layer
        :param init_w_function: function for initializing the layer
        :param init_a_function: activation function to be applied after the layer
        :param dropout: Whether dropout is applied or not in the layer
        :param drop_prob: probability of dropout
        :param batch_norm: Whether batch normalization is applied after the layer
        :return:
        """

        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos >= self.number_hidden_layers:
            return

        # We create the new layer and add it to the network descriptor
        self.dims = np.insert(self.dims, layer_pos, lay_dims)
        self.init_functions = np.insert(self.init_functions, layer_pos, init_w_function)
        self.act_functions = np.insert(self.act_functions, layer_pos, init_a_function)
        self.number_hidden_layers = self.number_hidden_layers + 1
        self.batch_norm = np.insert(self.batch_norm, layer_pos, batch_norm)
        self.dropout = np.insert(self.dropout, layer_pos, dropout)
        self.dropout_probs = np.insert(self.dropout_probs, layer_pos, drop_prob)

    def remove_layer(self, layer_pos):
        """
        This function deletes a layer
        :param layer_pos: Position of the layer to be deleted
        :return:
        """

        # If not within feasible bounds, return
        if layer_pos <= 1 or layer_pos > self.number_hidden_layers:
            return

        # We set the number of input and output dimensions for the layer to be
        # added and for the ones in the architecture that will be connected to it

        # We delete the layer in pos layer_pos
        self.dims = np.delete(self.dims, layer_pos)
        self.init_functions = np.delete(self.init_functions, layer_pos)
        self.act_functions = np.delete(self.act_functions, layer_pos)
        self.batch_norm = np.delete(self.batch_norm, layer_pos)
        self.dropout = np.delete(self.dropout, layer_pos)
        self.dropout_probs = np.delete(self.dropout_probs, layer_pos)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers - 1

    def change_layer_dimension(self, new_dim):

        layer_pos = np.random.randint(0, self.number_hidden_layers)
        self.dims[layer_pos] = new_dim

    def print_components(self, identifier):
        print(identifier, ' n_hid:', self.number_hidden_layers)
        print(identifier, ' Dims:', self.dims)
        print(identifier, ' Init:', self.init_functions)
        print(identifier, ' Act:', self.act_functions)

    def codify_components(self, max_hidden_layers, ref_list_init_functions, ref_list_act_functions):

        max_total_layers = max_hidden_layers + 1
        # The first two elements of code are the number of layers and number of loops
        code = [self.number_hidden_layers]

        # We add all the layer dimension and fill with zeros all positions until max_layers
        code = code + self.dims + [-1]*(max_total_layers-len(self.dims))

        # We add the indices of init_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_f = []
        for init_f in self.init_functions:
            aux_f.append(ref_list_init_functions.index(init_f))
        code = code + aux_f + [-1]*(max_total_layers-len(aux_f))

        # We add the indices of act_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_a = []
        for act_f in self.act_functions:
            aux_a.append(ref_list_act_functions.index(act_f))
        code = code + aux_a + [-1]*(max_total_layers-len(aux_a))

        return code


class ConvDescriptor(NetworkDescriptor):
    def __init__(self, number_hidden_layers=2, input_dim=(28, 28, 3), output_dim=(7, 7, 1), op_type=(0, 1), filters=((3, 3, 2), (3, 3, 2)), strides=((1, 1, 1), (1, 1, 1)), list_init_functions=(0, 0), list_act_functions=(0, 0), dropout=(), batch_norm=()):
        """
        Descriptor for convolutional cells
        :param number_hidden_layers: Number of hidden layers (it's changed afterwards)
        :param input_dim: Dimension of the input
        :param output_dim: Expected dimension of the output (could be greater)
        :param op_type: Type of layer (Mean pooling, max pooling, or convolutional. it's changed afterwards)
        :param filters: list of dimensions of filters (it's changed afterwards)
        :param strides: list of strides (it's changed afterwards)
        :param list_init_functions: list of initialization functions of the filter weights (it's changed afterwards)
        :param list_act_functions: list of activation functions after filters (it's changed afterwards)
        :param dropout: list of booleans defining whether dropout is applied to each layer (it's changed afterwards)
        :param batch_norm: list of booleans defining whether batch normalization is applied to each layer (it's changed afterwards)
        """

        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, init_functions=list_init_functions, act_functions=list_act_functions, dropout=dropout, batch_norm=batch_norm)
        self.layers = op_type
        self.filters = filters
        self.strides = strides
        self.shapes = []    # This is an important variable which contains the shapes of the blobs. This way we control that the CNN does not produce
                            # too small blobs

    def random_init(self, input_size, output_size, nlayers, _, max_stride, max_filter, no_drop, no_batch):
        """
        This function randomly initializes the descriptor. This function is susceptible of being modified by the user with specific creation needs
        :param input_size:  Dimension of the input
        :param output_size: Expected dimension of the output (could be greater)
        :param nlayers: maximum number of layers
        :param _: unused
        :param max_stride: maximum stride possible (used as 2)
        :param max_filter: maximum filter size possible (used as 3)
        :param no_drop: Whether dropout is a possibility in the network
        :param no_batch: Whether batch normalization is a possibility in the network
        :return:
        """

        self.input_dim = input_size
        self.output_dim = output_size

        self.number_hidden_layers = np.random.randint(nlayers)+1
        self.init_functions = []
        self.act_functions = []
        self.layers = []
        self.strides = []
        self.filters = []
        shape = self.input_dim
        i = 0
        while i < nlayers:

            self.strides += [np.array([np.random.randint(1, max_stride)] * 2 + [1])]
            self.filters += [np.array([np.random.randint(2, max_filter)] * 2 + [np.random.randint(3, 64.5)])]
            shape = compute_output(shape, 0, self.filters[-1], self.strides[-1])
            if shape[0] < 2 or shape[1] < 2 or shape[0] * shape[1] * self.filters[-1][2] < self.output_dim:  # If the blob size is too small
                self.number_hidden_layers = i
                self.strides = self.strides[:-1]
                self.filters = self.filters[:-1]
                break
            i += 1
            self.shapes += [shape]
            self.init_functions += [np.random.choice(initializations[1:])]
            self.act_functions += [np.random.choice(activations)]
            self.layers += [np.random.choice([0, 1, 2])]

        if no_batch:
            self.batch_norm = np.zeros(self.number_hidden_layers+1)
        else:
            self.batch_norm = np.random.randint(0, 2, size=self.number_hidden_layers+1)

        if no_drop:
            self.dropout = np.zeros(self.number_hidden_layers+1)
            self.dropout_probs = np.zeros(self.number_hidden_layers+1)
        else:
            self.dropout = np.random.randint(0, 2, size=self.number_hidden_layers+1)
            self.dropout_probs = np.random.rand(self.number_hidden_layers+1)

    def add_layer(self, layer_pos, lay_type, lay_params):
        """
        This function adds a layer in the layer_pos position
        :param layer_pos: Position of the layer
        :param lay_type: Type of operation (0, 1: pooling, 2 convolutional)
        :param lay_params: sizes of the *filters*.
        :return:
        """
        self.layers.insert(layer_pos, lay_type)
        if self.shapes[-1][0] > 2 and self.shapes[-1][1] > 2:
            self.number_hidden_layers += 1
            if lay_type < 2:
                self.filters.insert(layer_pos, [lay_params[1], lay_params[1], 1])
                self.act_functions.insert(layer_pos, None)
                self.init_functions.insert(layer_pos, None)
                self.strides.insert(layer_pos, [lay_params[0], lay_params[0], 1])
            elif lay_type == 2:
                self.strides.insert(layer_pos, [lay_params[0], lay_params[0], 1])
                self.filters.insert(layer_pos, [lay_params[1], lay_params[1], np.random.randint(0, 65)])
                self.act_functions.insert(layer_pos, lay_params[2])
                self.init_functions.insert(layer_pos, lay_params[3])
            self.reset_shapes()
            return 0
        else:
            return 1

    def reset_shapes(self):
        """
        This function recomputes the self.shapes variable. Used when a mutation is performed to keep the variable updated.
        :return:
        """
        self.shapes = []
        shape = self.input_dim
        for lay in range(self.number_hidden_layers):
            shape = compute_output(shape, 0, self.filters[lay], self.strides[lay])
            self.shapes += [shape]

    def remove_layer(self, layer_pos):
        """
        This function deletes a layer
        :param layer_pos: Position of the layer to be deleted
        :return:
        """

        self.filters.pop(layer_pos)
        self.act_functions.pop(layer_pos)
        self.init_functions.pop(layer_pos)
        self.strides.pop(layer_pos)
        self.number_hidden_layers -= 1
        self.reset_shapes()

    def remove_random_layer(self):
        """
        Select a random layer and execute the deletion
        :return:
        """
        if self.number_hidden_layers > 1:
            layer_pos = np.random.randint(len(self.filters))
            self.remove_layer(layer_pos)
            return 0
        else:
            return -1

    def change_filters(self, layer_pos, new_kernel_size, new_channel):
        """
        Change the size of one layer filter
        :param layer_pos: Position of the filter to be changed
        :param new_kernel_size: Height and width of the filter (only square filters are allowed)
        :param new_channel: Number of output channels
        :return:
        """
        self.filters[layer_pos][0] = new_kernel_size
        self.filters[layer_pos][1] = new_kernel_size
        self.filters[layer_pos][2] = new_channel
        self.reset_shapes()

    def change_stride(self, layer_pos, new_stride):
        """
        Change the stride of a filter in a layer
        :param layer_pos: Layer which stride is changed
        :param new_stride: self-explanatory
        :return:
        """
        self.strides[layer_pos][0] = new_stride
        self.strides[layer_pos][1] = new_stride
        self.reset_shapes()

    def print_components(self, identifier):
        print(identifier, ' n_conv:', len([x for x in self.filters if not x == -1]))
        print(identifier, ' n_pool:', len([x for x in self.filters if x == -1]))
        print(identifier, ' Init:', self.init_functions)
        print(identifier, ' Act:', self.act_functions)
        print(identifier, ' filters:', self.filters)
        print(identifier, ' strides:', self.strides)

    def codify_components(self):

        filters = [str(x) for x in self.filters]
        init_funcs = [str(x) for x in self.init_functions]
        act_funcs = [str(x) for x in self.act_functions]
        sizes = [[str(y) for y in x] for x in self.filters]
        strides = [str(x) for x in self.strides]
        return str(self.input_dim) + "_" + str(self.output_dim) + "_" + ",".join(filters) + "*" + ",".join(["/".join(szs) for szs in sizes]) + "*" + ",".join(strides) + "_" + ",".join(init_funcs) + "_" + ",".join(act_funcs)


class TConvDescriptor(NetworkDescriptor):
    def __init__(self, number_hidden_layers=2, input_dim=(7, 7, 50), output_dim=(28, 28, 3), filters=((3, 3, 2), (3, 3, 2)), strides=((1, 1, 1), (1, 1, 1)), list_init_functions=(0, 0), list_act_functions=(0, 0), dropout=(), batch_norm=()):
        """
       Descriptor for transposed convolutional cells
       :param number_hidden_layers: Number of hidden layers (it's changed afterwards)
       :param input_dim: Dimension of the input
       :param output_dim: Expected dimension of the output (could be greater)
       :param filters: list of dimensions of filters (it's changed afterwards)
       :param strides: list of strides (it's changed afterwards)
       :param list_init_functions: list of initialization functions of the filter weights (it's changed afterwards)
       :param list_act_functions: list of activation functions after filters (it's changed afterwards)
       :param dropout: list of booleans defining whether dropout is applied to each layer (it's changed afterwards)
       :param batch_norm: list of booleans defining whether batch normalization is applied to each layer (it's changed afterwards)
       """

        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, init_functions=list_init_functions, act_functions=list_act_functions, dropout=dropout, batch_norm=batch_norm)
        self.filters = filters
        self.strides = strides
        self.output_shapes = []

    def random_init(self, input_size, output_size, _, __, max_stride, max_filter, no_drop, no_batch):
        """
        This function randomly initializes the descriptor. This function is susceptible of being modified by the user with specific creation needs
        :param input_size:  Dimension of the input
        :param output_size: Expected dimension of the output (could be greater)
        :param _: unused
        :param __: unused
        :param max_stride: maximum stride possible (used as 2)
        :param max_filter: maximum filter size possible (used as 3)
        :param no_drop: Whether dropout is a possibility in the network
        :param no_batch: Whether batch normalization is a possibility in the network
        :return:
        """
        self.input_dim = input_size
        self.output_dim = output_size

        # Random initialization

        self.strides = []
        self.filters = []
        self.init_functions = []
        self.act_functions = []
        shape = [-1] + list(self.input_dim)
        for i in range(300):
            self.strides += [np.array([np.random.randint(1, max_stride)] * 2 + [1])]
            self.filters += [np.array([np.random.randint(2, max_filter)] * 2 + [np.random.randint(3, 65)])]
            shape = [-1] + compute_output(shape[1:], 4, self.filters[-1], self.strides[-1]) + [self.filters[-1][2]]
            self.output_shapes += [shape]
            self.init_functions += [np.random.choice(initializations[1:])]
            self.act_functions += [np.random.choice(activations)]

            if shape[1] >= self.output_dim[0] and shape[2] >= self.output_dim[1]:  # Once the expected shape is exceeded, we have enough layers
                self.filters[-1][2] = self.output_dim[2]
                shape[-1] = self.output_dim[2]
                self.output_shapes[-1] = shape
                self.number_hidden_layers = i+1
                break

        if no_batch:
            self.batch_norm = np.zeros(self.number_hidden_layers+1)
        else:
            self.batch_norm = np.random.randint(0, 2, size=self.number_hidden_layers+1)

        if no_drop:
            self.dropout = np.zeros(self.number_hidden_layers+1)
            self.dropout_probs = np.zeros(self.number_hidden_layers+1)
        else:
            self.dropout = np.random.randint(0, 2, size=self.number_hidden_layers+1)
            self.dropout_probs = np.random.rand(self.number_hidden_layers+1)

    def add_layer(self, layer_pos, lay_params):
        """
        This function adds a layer in the layer_pos position
        :param layer_pos: Position of the layer
        :param lay_params: sizes of the filters.
        :return:
        """

        self.number_hidden_layers += 1
        self.strides.insert(layer_pos, [lay_params[0], lay_params[0], 1])
        self.filters.insert(layer_pos, [lay_params[1], lay_params[1], np.random.randint(0, 65)])
        self.act_functions.insert(layer_pos, lay_params[2])
        self.init_functions.insert(layer_pos, lay_params[3])
        self.reset_shapes()
        return 0

    def reset_shapes(self):
        """
        This function recomputes the self.shapes variable. Used when a mutation is performed to keep the variable updated.
        :return:
        """
        self.output_shapes = []
        shape = self.input_dim
        for lay in range(self.number_hidden_layers):
            shape = compute_output(shape, 4, self.filters[lay], self.strides[lay])
            self.output_shapes += [[-1] + shape + [self.filters[lay][2]]]

    def remove_layer(self, layer_pos):
        """
        This function deletes a layer
        :param layer_pos: Position of the layer to be deleted
        :return:
        """

        self.filters.pop(layer_pos)
        self.act_functions.pop(layer_pos)
        self.init_functions.pop(layer_pos)
        self.strides.pop(layer_pos)
        self.number_hidden_layers -= 1
        self.reset_shapes()

    def remove_random_layer(self):
        """
        Search for a layer which is deletable and (in case it exists) run deletion
        :return:
        """
        layers = np.random.choice(np.arange(0, len(self.filters)), size=len(self.filters), replace=False)

        for layer_pos in layers:
            if self.number_hidden_layers > 1 and (self.output_shapes[-1][1] - 1) / self.strides[layer_pos][0] - self.filters[layer_pos][0] > self.output_dim[0]:
                self.remove_layer(layer_pos)
                return 0
        return -1

    def change_activation(self, layer_pos, new_act_fn):
        self.act_functions[layer_pos] = new_act_fn

    def change_weight_init(self, layer_pos, new_weight_fn):
        self.init_functions[layer_pos] = new_weight_fn

    def change_filters(self, layer_pos, new_kernel_size, new_channel):
        self.filters[layer_pos][0] = new_kernel_size
        self.filters[layer_pos][1] = new_kernel_size
        self.filters[layer_pos][2] = new_channel
        self.reset_shapes()

    def change_stride(self, layer_pos, new_stride):
        self.strides[layer_pos][0] = new_stride
        self.strides[layer_pos][1] = new_stride
        self.reset_shapes()

    def print_components(self, identifier):
        print(identifier, ' n_conv:', len([x for x in self.filters if not x == -1]))
        print(identifier, ' n_pool:', len([x for x in self.filters if x == -1]))
        print(identifier, ' Init:', self.init_functions)
        print(identifier, ' Act:', self.act_functions)
        print(identifier, ' filters:', self.filters)
        print(identifier, ' strides:', self.strides)

    def codify_components(self):

        filters = [str(x) for x in self.filters]
        init_funcs = [str(x) for x in self.init_functions]
        act_funcs = [str(x) for x in self.act_functions]
        sizes = [[str(y) for y in x] for x in self.filters]
        strides = [str(x) for x in self.strides]
        return str(self.input_dim) + "_" + str(self.output_dim) + "_" + ",".join(filters) + "*" + ",".join(["/".join(szs) for szs in sizes]) + "*" + ",".join(strides) + "_" + ",".join(init_funcs) + "_" + ",".join(act_funcs)


class Network:
    def __init__(self, network_descriptor):
        """
        This class contains the tensorflow definition of the networks (i.e., the "implementation" of the descriptors)
        :param network_descriptor: The descriptor this class is implementing
        """
        self.descriptor = network_descriptor
        self.List_layers = []           # This will contain the outputs of all layers in the network
        self.List_weights = []          # This will contain the list of weights/kernels of the layers


class MLP(Network):

    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)
        self.List_bias = []             # This will contain the biases (if necessary) of the layers

    def create_hidden_layer(self, in_size, out_size, init_w_function, layer_name):
        """
        This function creates the necessary tf variables for doing the matmul and addition operations afterwards
        :param in_size: number of neurons in the previous layer
        :param out_size: number of neurons in the current layer
        :param init_w_function: function for randomly initializing the weights
        :param layer_name: Usually the number of the layer
        :return:
        """

        w = tf.Variable(init_w_function(shape=[in_size, out_size]), name="W"+layer_name)
        b = tf.Variable(tf.zeros(shape=[out_size]), name="b"+layer_name)

        self.List_weights.append(w)
        self.List_bias.append(b)

    def initialization(self, graph):
        """
        This function uses create_hidden_layer iteratively to form the necessary variables for the MLP
        :param graph: tensorflow graph on which the variables are created (and where the whole model is implemented)
        :return:
        """
        with graph.as_default():
            self.create_hidden_layer(self.descriptor.input_dim, self.descriptor.dims[0], self.descriptor.init_functions[0], str(0))

            for lay in range(1, self.descriptor.number_hidden_layers):
                self.create_hidden_layer(self.descriptor.dims[lay-1], self.descriptor.dims[lay], self.descriptor.init_functions[lay], str(lay))

            self.create_hidden_layer(self.descriptor.dims[self.descriptor.number_hidden_layers-1], self.descriptor.output_dim, self.descriptor.init_functions[self.descriptor.number_hidden_layers], str(self.descriptor.number_hidden_layers))

    def building(self, layer, graph):
        """
        This function uses the variables created by the initialization function to create the MLP
        :param layer: input of the network
        :param graph: graph on which the variables are defined
        :return:
        """
        with graph.as_default():

            for lay in range(self.descriptor.number_hidden_layers+1):
                act = self.descriptor.act_functions[lay]
                layer = tf.matmul(layer, self.List_weights[lay]) + self.List_bias[lay]

                if self.descriptor.batch_norm[lay] > 0:
                    print("batch")
                    layer = tf.layers.batch_normalization(layer)

                if act is not None and lay < self.descriptor.number_hidden_layers:
                    layer = act(layer)

                if self.descriptor.dropout[lay] > 0:

                    layer = tf.layers.dropout(layer, rate=self.descriptor.dropout_probs)
                self.List_layers.append(layer)

        return layer


class CNN(Network):

    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)

    def initialization(self, graph):
        """
        This function creates all the necessary filters for the CNN
        :param graph: Graph in which the variables are created (and convolutional operations are performed)
        :return:
        """
        last_c = self.descriptor.input_dim[-1]
        with graph.as_default():
            for ind in range(self.descriptor.number_hidden_layers):

                if self.descriptor.layers[ind] == 2:  # If the layer is convolutional
                    if self.descriptor.init_functions[ind] == 0:
                        w = tf.Variable(np.random.uniform(-0.1, 0.1, size=[self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], last_c, self.descriptor.filters[ind][2]]).astype('float32'), name="W"+str(ind))
                    else:
                        w = tf.Variable(np.random.normal(0, 0.03, size=[self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], last_c, self.descriptor.filters[ind][2]]).astype('float32'), name="W"+str(ind))
                    self.List_weights += [tf.Variable(w)]
                    last_c = self.descriptor.filters[ind][2]

                else:  # In case the layer is pooling, no need of weights
                    self.List_weights += [tf.Variable(-1)]

    def building(self, layer, graph):
        """
        Using the filters defined in the initialization function, create the CNN
        :param layer: Input of the network
        :param graph: Graph in which variables were defined
        :return:
        """
        with graph.as_default():
            for ind in range(self.descriptor.number_hidden_layers):

                if self.descriptor.layers[ind] == 2:  # If the layer is convolutional
                    layer = tf.nn.conv2d(layer, self.List_weights[ind], (1, self.descriptor.strides[ind][0], self.descriptor.strides[ind][1], self.descriptor.strides[ind][2]), padding=[[0, 0], [0, 0], [0, 0], [0, 0]])
                elif self.descriptor.layers[ind] == 0:  # If the layer is average pooling
                    layer = tf.nn.avg_pool(layer, (1, self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], 1), (1, self.descriptor.strides[ind][0], self.descriptor.strides[ind][1], 1), padding="VALID")
                else:
                    layer = tf.nn.max_pool(layer, (1, self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], 1), (1, self.descriptor.strides[ind][0], self.descriptor.strides[ind][1], 1), padding="VALID")

                if self.descriptor.act_functions[ind] is not None:  # If we have activation function
                    layer = self.descriptor.act_functions[ind](layer)
                # batch normalization and dropout not implemented (maybe pooling operations should be part of convolutional layers instead of layers by themselves)
                self.List_layers += [layer]

        return layer


class TCNN(Network):
    """
    Almost identical to CNN
    """
    def __init__(self, network_descriptor):
        super().__init__(network_descriptor)

    def initialization(self, graph):

        last_c = self.descriptor.input_dim[-1]
        with graph.as_default():

            for ind in range(self.descriptor.number_hidden_layers):
                if self.descriptor.init_functions[ind] == 0:
                    w = tf.Variable(np.random.uniform(-0.1, 0.1, size=[self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], self.descriptor.filters[ind][2], last_c]).astype('float32'), name="W"+str(ind))
                else:
                    w = tf.Variable(np.random.normal(0, 0.03, size=[self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], self.descriptor.filters[ind][2], last_c]).astype('float32'), name="W"+str(ind))
                self.List_weights += [tf.Variable(w)]

                last_c = self.descriptor.filters[ind][2]

    def building(self, layer, graph):
        with graph.as_default():
            for ind in range(self.descriptor.number_hidden_layers):
                dyn_input_shape = tf.shape(layer)
                batch_size = dyn_input_shape[0]

                output_shape = tf.stack([batch_size, self.descriptor.output_shapes[ind][1], self.descriptor.output_shapes[ind][2], self.descriptor.output_shapes[ind][3]])

                layer = tf.nn.conv2d_transpose(layer, self.List_weights[ind], output_shape, (1, self.descriptor.strides[ind][0], self.descriptor.strides[ind][1], self.descriptor.strides[ind][2]), padding="VALID")

                if self.descriptor.act_functions[ind] is not None:
                    layer = self.descriptor.act_functions[ind](layer)

                self.List_layers += [layer]

        return layer


def compute_output(input_shape, layer_type, filter_size, stride):

    if layer_type < 3:
        output_shape = (np.array(input_shape[:2]) - np.array(filter_size[:2]) + 1) // np.array(stride[:2])
        return np.array([np.max([output_shape[0], 1]), np.max([output_shape[1], 1]), filter_size[2]])
    else:
        return [(input_shape[0]-1) * stride[0] + filter_size[0], (input_shape[1]-1) * stride[1] + filter_size[1]]
