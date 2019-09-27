import tensorflow as tf
import numpy as np
import copy


def xavier_init(fan_in=None, fan_out=None, shape=None):
    """ Xavier initialization of network weights"""
    if fan_in is None or shape is not None:
        fan_in = shape[0]
        fan_out = shape[1]
    # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    lim = np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-lim, maxval=lim, dtype=tf.float32)


act_functions = [None, tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh]
init_functions = [xavier_init, tf.random_uniform, tf.random_normal]


class ConvDescriptor:
    def __init__(self, input_dim, output_dim, layers, filters, strides, sizes, act_fns, init_fns):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = layers  # List with layer types.

        # Parameters of the conv and deconv layers. Will be -1 when corresponding to a pooling layer

        self.filters = filters  # List of number of filters
        self.act_fns = act_fns  # List of activation functions of weights
        self.init_fns = init_fns  # List of initialization function

        # Parameters common to both conv (and deconv) and pooling layers

        self.strides = strides
        self.sizes = sizes  # Kernel sizes

    def copy_from_other_network(self, other_network):
        self.filters = copy.deepcopy(other_network.filters)
        self.input_dim = other_network.input_dim
        self.output_dim = other_network.output_dim
        self.layers = copy.deepcopy(other_network.layers)
        self.strides = copy.deepcopy(other_network.strides)
        self.sizes = copy.deepcopy(other_network.sizes)
        self.act_fns = copy.deepcopy(other_network.act_fns)
        self.init_fns = copy.deepcopy(other_network.init_fns)

    def network_add_layer(self, layer_pos, lay_type, lay_params):
        """
        Function: network_add_layer()
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        if "pool" in lay_type:
            self.filters.insert(layer_pos, -1)
            self.act_fns.insert(layer_pos, -1)
            self.init_fns.insert(layer_pos, -1)
            self.strides.insert(layer_pos, lay_params[0])
            self.sizes.insert(layer_pos, lay_params[1])
        elif "conv" in lay_type:
            self.strides.insert(layer_pos, lay_params[0])
            self.sizes.insert(layer_pos, lay_params[1])
            self.filters.insert(layer_pos, lay_params[2])
            self.act_fns.insert(layer_pos, lay_params[3])
            self.init_fns.insert(layer_pos, lay_params[4])

    """
    Function: network_remove_layer()
    Adds a layer at a specified position, with a given  number of units, init weight
    function, activation function.
    If the layer is inserted in layer_pos in [0,number_hidden_layers] then all the
    other layers are shifted.
    If the layer is inserted in position number_hidden_layers+1, then it is just appended
    to previous layer and it will output output_dim variables.
    If the position for the layer to be added is not within feasible bounds
    in the current architecture, the function silently returns
    """

    def network_remove_layer(self, layer_pos):

        # We set the number of input and output dimensions for the layer to be
        # added and for the ones in the architecture that will be connected to it

        # We delete the layer in pos layer_pos
        self.filters.pop(layer_pos)
        self.act_fns.pop(layer_pos)
        self.init_fns.pop(layer_pos)
        self.strides.pop(layer_pos)
        self.sizes.pop(layer_pos)

    def network_remove_random_layer(self):
        layer_pos = np.random.randint(len(self.filters))
        self.network_remove_layer(layer_pos)

    def change_activation_fn_in_layer(self, layer_pos, new_act_fn):
        self.act_fns[layer_pos] = new_act_fn

    def change_weight_init_fn_in_layer(self, layer_pos, new_weight_fn):
        self.init_fns[layer_pos] = new_weight_fn

    def change_filters_in_layer(self, layer_pos, new_kernel_size):
        self.sizes[layer_pos] = new_kernel_size

    def change_stride_in_layer(self, layer_pos, new_stride):
        self.strides[layer_pos] = new_stride

    def print_components(self, identifier):
        print(identifier, ' n_conv:', len([x for x in self.filters if not x == -1]))
        print(identifier, ' n_pool:', len([x for x in self.filters if x == -1]))
        print(identifier, ' Init:', self.init_fns)
        print(identifier, ' Act:', self.act_fns)
        print(identifier, ' k_size:', self.sizes)
        print(identifier, ' strides:', self.strides)

    def codify_components(self):

        filters = [str(x) for x in self.filters]
        init_funcs = [str(x) for x in self.init_fns]
        act_funcs = [str(x) for x in self.act_fns]
        sizes = [[str(y) for y in x] for x in self.sizes]
        strides = [str(x) for x in self.strides]
        layers = [str(x) for x in self.layers]
        return str(self.input_dim) + "_" + str(self.output_dim) + "_" + ",".join(filters) + "*" + ",".join(["/".join(szs) for szs in sizes]) + "*" + ",".join(layers) + "*" + ",".join(strides) + "_" + ",".join(init_funcs) + "_" + ",".join(act_funcs)


class NetworkDescriptor:
    def __init__(self, number_hidden_layers=1, input_dim=1, output_dim=1,  list_dims=None, list_init_functions=None, list_act_functions=None, dropout=(), batch_norm=()):
        self.number_hidden_layers = number_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.List_dims = list_dims
        self.List_init_functions = list_init_functions
        self.List_act_functions = list_act_functions
        self.batch_norm = batch_norm
        self.dropout = dropout

    def copy_from_other_network(self, other_network):
        self.number_hidden_layers = other_network.number_hidden_layers
        self.input_dim = other_network.input_dim
        self.output_dim = other_network.output_dim
        self.List_dims = copy.deepcopy(other_network.List_dims)
        self.List_init_functions = copy.deepcopy(other_network.List_init_functions)
        self.List_act_functions = copy.deepcopy(other_network.List_act_functions)

    def network_add_layer(self, layer_pos, lay_dims, init_w_function, init_a_function):
        """
        Function: network_add_layer()
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos >= self.number_hidden_layers:
            return

        # We create the new layer and add it to the network descriptor
        self.List_dims = np.insert(self.List_dims, layer_pos, lay_dims)
        self.List_init_functions = np.insert(self.List_init_functions, layer_pos, init_w_function)
        self.List_act_functions = np.insert(self.List_act_functions, layer_pos, init_a_function)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers + 1

    def network_remove_layer(self, layer_pos):
        """
        Function: network_remove_layer()
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        """

        # If not within feasible bounds, return
        if layer_pos <= 1 or layer_pos > self.number_hidden_layers:
            return

        # We set the number of input and output dimensions for the layer to be
        # added and for the ones in the architecture that will be connected to it

        # We delete the layer in pos layer_pos
        self.List_dims = np.delete(self.List_dims, layer_pos)
        self.List_init_functions = np.delete(self.List_init_functions, layer_pos)
        self.List_act_functions = np.delete(self.List_act_functions, layer_pos)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers - 1

    def network_remove_random_layer(self):
        layer_pos = np.random.randint(self.number_hidden_layers)
        self.network_remove_layer(layer_pos)

    def change_activation_fn_in_layer(self, layer_pos, new_act_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.List_act_functions[layer_pos] = new_act_fn

    def change_weight_init_fn_in_layer(self, layer_pos, new_weight_fn):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        self.List_init_functions[layer_pos] = new_weight_fn

    def change_all_weight_init_fns(self, new_weight_fn):
        # If not within feasible bounds, return
        for layer_pos in range(self.number_hidden_layers):
            self.List_init_functions[layer_pos] = new_weight_fn

    def change_dimensions_in_layer(self, layer_pos, new_dim):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        # If the dimension of the layer is identical to the existing one, return
        self.List_dims[layer_pos] = new_dim

    def change_dimensions_in_random_layer(self, max_layer_size):
        layer_pos = np.random.randint(self.number_hidden_layers)
        new_dim = np.random.randint(max_layer_size)+1
        self.change_dimensions_in_layer(layer_pos, new_dim)

    def print_components(self, identifier):
        print(identifier, ' n_hid:', self.number_hidden_layers)
        print(identifier, ' Dims:', self.List_dims)
        print(identifier, ' Init:', self.List_init_functions)
        print(identifier, ' Act:', self.List_act_functions)

    def codify_components(self, max_hidden_layers, ref_list_init_functions, ref_list_act_functions):

        max_total_layers = max_hidden_layers + 1
        # The first two elements of code are the number of layers and number of loops
        code = [self.number_hidden_layers]

        # We add all the layer dimension and fill with zeros all positions until max_layers
        code = code + self.List_dims + [-1]*(max_total_layers-len(self.List_dims))

        # We add the indices of init_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_f = []
        for init_f in self.List_init_functions:
            aux_f.append(ref_list_init_functions.index(init_f))
        code = code + aux_f + [-1]*(max_total_layers-len(aux_f))

        # We add the indices of act_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_a = []
        for act_f in self.List_act_functions:
            aux_a.append(ref_list_act_functions.index(act_f))
        code = code + aux_a + [-1]*(max_total_layers-len(aux_a))

        return code


class Network:
    def __init__(self, network_descriptor, in_ch=0):
        self.descriptor = network_descriptor
        self.in_ch = in_ch
        self.List_layers = []
        self.List_weights = []
        self.List_bias = []
        self.List_dims = []
        self.List_init_functions = []
        self.List_act_functions = []

    def reset_network(self):
        self.List_layers = []
        self.List_weights = []
        self.List_bias = []
        self.List_dims = []
        self.List_init_functions = []
        self.List_act_functions = []

    def create_hidden_layer(self, in_size, out_size, init_w_function, layer_name):

        w = tf.Variable(init_w_function(shape=[in_size, out_size]), name="W"+layer_name)
        b = tf.Variable(tf.zeros(shape=[out_size]), name="b"+layer_name)

        self.List_weights.append(w)
        self.List_bias.append(b)

    def network_initialization(self, graph):
        with graph.as_default():
            self.create_hidden_layer(self.descriptor.input_dim, self.descriptor.List_dims[0], self.descriptor.List_init_functions[0], str(0))

            for lay in range(1, self.descriptor.number_hidden_layers):
                self.create_hidden_layer(self.descriptor.List_dims[lay-1], self.descriptor.List_dims[lay], self.descriptor.List_init_functions[lay], str(lay))

            self.create_hidden_layer(self.descriptor.List_dims[self.descriptor.number_hidden_layers-1], self.descriptor.output_dim, self.descriptor.List_init_functions[self.descriptor.number_hidden_layers], str(self.descriptor.number_hidden_layers))

    def network_building(self, layer, graph):
        with graph.as_default():

            for lay in range(self.descriptor.number_hidden_layers+1):
                act = self.descriptor.List_act_functions[lay]
                layer = tf.matmul(layer, self.List_weights[lay]) + self.List_bias[lay]

                if lay in self.descriptor.batch_norm:
                    layer = tf.layers.batch_normalization(layer)

                if act is not None and lay < self.descriptor.number_hidden_layers:
                    layer = act(layer)

                if lay in self.descriptor.dropout:
                    layer = tf.layers.dropout(layer)
                self.List_layers.append(layer)

        return layer

    def convolutional_initialization(self):

        last_c = self.in_ch

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

    def convolutional_evaluation(self, layer):

        for ind, lay in enumerate(self.descriptor.layers):
            if lay == 2:  # If the layer is convolutional
                layer = tf.nn.conv2d(layer, self.List_weights[ind], (1, self.descriptor.strides[ind], self.descriptor.strides[ind], 1), padding="VALID")
            elif lay == 0:  # If the layer is average pooling
                layer = tf.nn.avg_pool(layer, (1, self.descriptor.sizes[ind, 0], self.descriptor.sizes[ind, 1], 1), (1, self.descriptor.strides[ind], self.descriptor.strides[ind], 1), "SAME")
            else:
                layer = tf.nn.max_pool(layer, (1, self.descriptor.sizes[ind, 0], self.descriptor.sizes[ind, 1], 1), (1, self.descriptor.strides[ind], self.descriptor.strides[ind], 1), "SAME")
            if self.descriptor.act_fns[ind] > 0:
                layer = act_functions[self.descriptor.act_fns[ind]](layer)

            self.List_layers += [layer]

        return layer

