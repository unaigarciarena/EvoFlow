import numpy as np
import copy
from VALP.classes import Connection, InOut, NetworkComp, ModelComponent
try:
    import pygraphviz
except:
    pass
import tensorflow as tf
import os
from functools import reduce


def xavier_init(fan_in=None, fan_out=None, shape=None, constant=1):
    """ Xavier initialization of network weights"""
    if fan_in is None:
        fan_in = shape[0]
        fan_out = shape[1]
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


init_functions = np.array([xavier_init, tf.random_uniform, tf.random_normal])
act_functions = np.array([None, tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh, tf.nn.softmax])


class MNMDescriptor(object):
    def __init__(self, max_comp, model_inputs, model_outputs, load=None, name=""):
        """
        :param max_comp: Number of components allowed in the model
        :param model_inputs: List of inputs of the model (InOut objects)
        :param model_outputs: List of outputs of the model (InOut objects)
        """
        self.reachable = {}  # comp_id: [list of components that are reached by comp_id]
        self.last_con = 0  # Counter
        self.last_net = 0  # Counter
        self.networks = {}  # id: ModelComponent
        self.connections = {}  # id: Connection
        self.inputs = {}  # id: InOut
        self.active_outputs = []  # List of components that require an input that don't have any
        self.outputs = {}  # id: InOut
        self.comps_below = {}  # Dict of components providing input to the component as key of the dict
        self.name = name
        if load is not None:
            self.load(load)
        else:
            if type(max_comp) is not int:
                raise Exception("The number of components should be an integer")
            if type(model_inputs) is not dict:
                raise Exception("Inputs of the model should be given in a dictionary; id in the key and size in the content")
            if type(model_outputs) is not dict:
                raise Exception("Outputs of the model should be given in a dictionary; id in the key and size in the content")

            self.constructed = False  # Whether the descriptor is a VVC or not
            self.max_comp = max_comp  # Maximum number of networks
            self.inputs = model_inputs  # id: InOut
            self.active_outputs = list(model_outputs.keys())  # List of components that require an input that don't have any
            self.outputs = model_outputs  # id: InOut

            for i in list(model_outputs.keys()) + list(model_inputs.keys()):
                self.reachable[i] = [i]  # All components are reached by themselves

    def copy_from(self, desc):
        self.constructed = desc.constructed
        self.max_comp = desc.max_comp
        self.networks = desc.networks
        self.connections = desc.connections
        self.inputs = desc.inputs
        self.active_outputs = desc.active_outputs
        self.outputs = desc.outputs
        self.reachable = desc.reachable

    def print(self):
        print("Inputs of the model")

        for i in self.inputs:
            print("  " + i + " " + self.inputs[i].print())
        print("##################################")
        print("")
        print("Outputs of the model")
        for i in self.outputs:
            print("  " + i + " " + self.outputs[i].print())
        print("##################################")
        print("")
        print("Networks of the model")
        for i in self.networks:
            print("  " + i + " " + self.networks[i].print())
        print("##################################")
        print("")
        print("Connections of the model")
        for i in self.connections:
            print(self.connections[i].print())

    def print_model_graph(self, name=None, agent=([], [], [])):
        """
        For printing the descriptor in a PDF
        :param name: Name of the PDF file in which the descriptor will be printed
        :param agent: Unused
        :return: --
        """
        dot = pygraphviz.AGraph(directed="True")
        for outp in list(self.outputs.keys()):
            dot.add_node(outp, pos=(outp[1:] + ",10"), color="red", label=outp + ", " + str(self.outputs[outp].taking.size) + "-" + self.outputs[outp].taking.type)
        for inp in list(self.inputs.keys()):
            dot.add_node(inp, pos=(inp[1:] + ",0"), color="blue", label=inp + ", " + str(self.inputs[inp].producing.size) + "-" + self.inputs[inp].producing.type)
        for comp in list(self.networks.keys()):
            dot.add_node(comp, label=comp + "-" + str(type(self.networks[comp].descriptor).__name__)[:-14] + ":" + str(self.networks[comp].taking.size) + "-" + str(self.networks[comp].producing.size))

        for c in self.connections:
            con = self.connections[c]
            if self.conn_in_agent(con, agent[0]):
                dot.add_edge(con.input, con.output, label=str(con.name) + ": " + str(con.info.size) + " " + self.comp_by_ind(con.input).producing.type, color="blue")
            elif self.conn_in_agent(con, agent[1]):
                dot.add_edge(con.input, con.output, label=str(con.name) + ": " + str(con.info.size) + " " + self.comp_by_ind(con.input).producing.type, color="red")
            elif self.conn_in_agent(con, agent[2]):
                dot.add_edge(con.input, con.output, label=str(con.name) + ": " + str(con.info.size) + " " + self.comp_by_ind(con.input).producing.type, color="green")
            else:
                dot.add_edge(con.input, con.output, label=str(con.name) + ": " + str(con.info.size) + " " + self.comp_by_ind(con.input).producing.type, color="black")
        dot.layout('dot')
        if not name:
            name = str(hash(self))
        dot.draw(name + '.pdf')

    @staticmethod
    def conn_in_agent(con, agent):  # Unused
        if len(agent) == 0:
            return False
        for ag in agent:
            if ag.input == con.input and ag.output == con.output:
                return True
        return False

    def get_net_context(self, net):
        """
        Given the id of one network, this function returns the connections attached to it, and the components at the other sides of said connections
        :param net: ID
        :return: Net context
        """
        inputs = []
        outputs = []
        in_cons = []
        out_cons = []

        for con in self.connections:
            if net == self.connections[con].input:
                outputs += [self.connections[con].output]
                in_cons += [con]
            if net == self.connections[con].output:
                inputs += [self.connections[con].input]
                out_cons += [con]

        return inputs, in_cons, out_cons, outputs

    def add_net(self, net, index=None):
        """
        Adds "net" to the descriptor
        :param net: ModelComponent
        :param index: ID of net. It's deduced if not included
        :return: The ID of the network
        """

        if not type(net) is NetworkComp:
            raise Exception("Don't introduce a plain descriptor, introduce a NetworkComp")

        if not index:
            index = "n" + str(self.last_net)
        self.last_net += 1
        self.networks[index] = net
        return index

    def connect(self, index1, index2, name=""):
        """
        Given two components (index1 and index2), this function creates a connection between them and adds it to the list of connections
        :param index1: name of the first component, e.g., "n0"
        :param index2: name of the second component, e.g., "n1"
        :param name: Name of the connection, e.g., "c20". If no name is provided, the first available is chosen
        :return: The name of the connection
        """

        inp = self.comp_by_ind(index1)
        if name == "":
            name = "c" + str(self.last_con + 1)
        con = Connection(index1, index2, InOut(data_type=inp.producing.type, size=np.random.randint(inp.producing.size)), name)
        self.add_connection(con, name)
        return name

    def save(self, path=""):
        """
        Save the descriptor to a file
        :param path: Path where the file is to be stored
        :return: the path with the name of the file
        """
        path = path + "model_" + str(self.name) + ".txt"
        if os.path.isfile(path):
            os.remove(path)
        f = open(path, "w+")
        for ident in self.networks:
            f.write(ident + "_" + self.networks[ident].descriptor.codify_components() + "_" + str(self.networks[ident].taking.size) + "," + self.networks[ident].taking.type + "_" + str(self.networks[ident].producing.size) + "," + self.networks[ident].producing.type + "_" +
                    str(self.networks[ident].depth) + "_" + ",".join(self.reachable[ident]) + "_" + ",".join(self.comps_below[ident]) + "\n")
        f.write("\n")

        for ident in self.inputs:
            f.write(ident + "_" + str(self.inputs[ident].producing.size) + "_" + self.inputs[ident].producing.type + "_" + str(self.inputs[ident].depth) + "\n")
        f.write("\n")

        for ident in self.outputs:
            f.write(ident + "_" + str(self.outputs[ident].taking.size) + "_" + self.outputs[ident].taking.type + "_" + str(self.outputs[ident].depth) + "_" + ",".join(self.comps_below[ident]) + "\n")
        f.write("\n")

        for con in self.connections:
            f.write(self.connections[con].codify() + "\n")
        #f.write("\n")

        f.close()

        return path

    def load(self, name=""):
        """
        Load the descriptor from a file. Counterpart of the save() method above.
        :param name: Name of the file in which the descriptor was previously stored.
        :return: --
        """

        self.constructed = True
        if name == "":
            name = "/home/unai/Escritorio/MultiNetwork/model/model"

        network_descriptors = {"Generic": GenericDescriptor, "Decoder": DecoderDescriptor, "Discrete": DiscreteDescriptor, "Convolution": ConvolutionDescriptor}

        if not os.path.isfile(name):
            print("Error at loading the model")
            return None

        f = open(name, "r+")

        lines = f.readlines()

        i = 0
        while lines[i] != "\n":  # Each component is stored in a line
            ident, n_inp, kind, n_hidden, layers, init, act, cond_rand, taking, producing, depth, reachable, belows = lines[i][:-1].split("_")
            kwargs = {}
            if int(ident[1:]) > self.last_net:
                self.last_net = int(ident[1:])

            self.reachable[ident] = reachable.split(",")
            self.comps_below[ident] = belows.split(",")

            if "onv" in kind:  # Not working right now
                filters, sizes, layers, strides = layers.split("*")
                sizes = sizes.split(",")
                s = np.array([[int(sz) for sz in szs.split("/")] for szs in sizes])
                desc = network_descriptors[kind](int(inp), int(outp), int(n_inp), layers.split(","), filters.split(","), [int(x) for x in strides.split(",")], s, [int(x) for x in act.split(",")], [int(x) for x in init.split(",")], kwargs)
            else:
                if len(kwargs) > 0:  # Not working right now
                    kwargs = kwargs.split("-")
                    kwargs[0] = [int(x) for x in kwargs[0].split(".") if len(x) > 0]
                    kwargs[1] = [int(x) for x in kwargs[1].split(".") if len(x) > 0]
                if len(cond_rand) > 0:
                    cond_rand = cond_rand.split("-")
                    cond_rand[0] = [int(x) for x in cond_rand[0].split(",") if len(x) > 0]
                    cond_rand[1] = [int(x) for x in cond_rand[1].split(",") if len(x) > 0]
                    kwargs["conds"] = cond_rand
                desc = network_descriptors[kind](int(taking.split(",")[0]), int(producing.split(",")[0]), int(n_inp), int(n_hidden), [int(x) for x in layers.split(",") if x != "-1"], init_functions[[int(x) for x in init.split(",") if x != "-1"]],
                                                 act_functions[[int(x) for x in act.split(",") if x != "-1"]], **kwargs)

            # print("ident", ident, "n_inp", n_inp, "kind", kind, "inp", inp, "outp", outp, "layers", layers, "init", init, "act", act, "taking", taking, "producing", producing, "depth", depth, "kwargs", kwargs)
            net = NetworkComp(desc, InOut(size=int(taking.split(",")[0]), data_type=taking.split(",")[1]), InOut(data_type=producing.split(",")[1], size=int(producing.split(",")[0])), int(depth))

            self.add_net(net, ident)
            i += 1

        i += 1

        while lines[i] != "\n":  # Inputs

            ident, size, kind, depth = lines[i].split("_")

            self.inputs[ident] = ModelComponent(None, InOut(size=int(size), data_type=kind), int(depth))
            i += 1

        i += 1

        while lines[i] != "\n":  # Outputs

            ident, size, kind, depth, belows = lines[i].split("_")

            self.outputs[ident] = ModelComponent(InOut(size=int(size), data_type=kind), None, int(depth))
            self.comps_below[ident] = belows.split(",")
            i += 1

        i += 1

        while i < len(lines):  # Connections
            name, inp, outp, kind, size = lines[i].split("_")

            if int(name[1:]) > self.last_con:
                self.last_con = int(name[1:])

            self.connections[name] = Connection(inp, outp, InOut(kind, int(size)), name)
            i += 1
        self.update_below()

    # Getter-like functions

    def comp_ids(self):
        return list(self.networks.keys())

    def net_exists(self, net):
        return net.index in self.networks.keys()

    def comp_by_ind(self, i):
        if "i" in i:
            return self.inputs[i]
        if "o" in i:
            return self.outputs[i]
        return self.networks[i]

    def comp_number(self):
        return len(self.networks)

    def conn_number(self):
        return len(self.connections)

    def random_output(self):
        return np.random.choice(list(self.networks.keys())+list(self.outputs.keys()))

    def random_input(self, output):
        """
        Given a component, this function returns another component which could feed the first one
        :param output: The component for which an input is required, e.g., "n0" or "o0"
        :return: The required input, e.g., "n1" or "i1
        """
        try:
            if "o" in output or ("Network" in type(self.comp_by_ind(output)).__name__ and "Decoder" in type(self.comp_by_ind(output).descriptor).__name__):  # Outputs and decoders require special treatment
                aux = np.random.choice([i for i in list(self.networks.keys()) if self.networks[i].producing == self.comp_by_ind(output).taking])
                return aux
            else:
                comps = {**self.networks, **self.inputs}
                aux = np.random.choice([i for i in list(comps.keys()) if i not in self.reachable[output] and not self.conn_exists(i, output) and self.networks[output].taking.type in comps[i].producing.type])
                return aux
        except:
            return -1

    def pop_con(self, con):
        c = self.connections[con]
        del self.connections[con]
        self.update_below()
        return c

    def conn_exists(self, i0, i1):
        for conn in self.connections:
            if self.connections[conn].input == i0 and self.connections[conn].output == i1:
                return True
        return False

    def get_connection(self, i0, i1):
        for conn in self.connections:
            if self.connections[conn].input == i0 and self.connections[conn].output == i1:
                return self.connections[conn]
        return -1

    def get_depth(self, index):
        return self.comp_by_ind(index).depth

    def random_model_input(self):
        return np.random.choice(list(self.inputs.keys()))

    def add_connection(self, connection, name=""):
        self.last_con += 1
        c_name = ("c" + str(self.last_con)) if name == "" else name
        self.connections[c_name] = connection
        self.update_below()
        return c_name

    def active_indices(self):
        return self.active_outputs

    def delete_active_by_index(self, index):
        self.active_outputs = [i for i in self.active_outputs if i != index]

    def comp_by_input(self, comp):
        """
        Given a component or a dict of components, this function finds the components providing values to the first one (s)
        :param comp: The first component
        :return: The list of input providing components
        """
        ins = []
        for c in self.connections:
            con = self.connections[c]
            if (type(comp) is dict and con.output in comp) or ("str" in type(comp).__name__ and comp == con.output):
                ins += [con.input]

        return sorted(ins)

    def update_below(self):
        """
        This function updates the dict containing the components below each component
        :return: --
        """
        for comp in list(self.networks.keys()) + list(self.outputs.keys()):
            if comp in self.comps_below:
                below = self.comp_by_input(comp)
                self.comps_below[comp] += [bel for bel in below if bel not in self.comps_below[comp]]
            else:
                self.comps_below[comp] = self.comp_by_input(comp)

    def comp_by_output(self, comp):
        """
        Similar to the previous one, but with the output instead of the input.
        :param comp: The component whom outputs we are searching
        :return: Components reciving data from "comp"
        """
        outs = []
        for c in self.connections:
            con = self.connections[c]
            if con.input == comp:
                outs += [con.output]
        return outs

    def nets_that_produce(self, data_type, from_list):
        return [net for net in from_list if self.comp_by_ind(net).producing.type in data_type]

    def nodes(self):
        return list(self.networks.keys()) + list(self.inputs.keys()) + list(self.outputs.keys())


class NetworkDescriptor:

    def __init__(self, number_hidden_layers=1, n_inputs=0, input_dim=1, output_dim=1, init_fs=None, act_fs=None, dropout=(), dropout_probs=(), batch_norm=()):
        """
        This class implements the descriptor of a generic network. Subclasses of this are the ones evolved.
        :param number_hidden_layers: Number of hidden layers in the network
        :param input_dim: Dimension of the input data (can have one or more dimensions)
        :param output_dim: Expected output of the network (similarly, can have one or more dimensions)
        :param init_fs: Weight initialization functions. Can have different ranges, depending on the subclass
        :param act_fs: Activation functions to be applied after each layer
        :param dropout: A 0-1 array of the length number_hidden_layers indicating  whether a dropout "layer" is to be
        applied AFTER the activation function
        :param batch_norm: A 0-1 array of the length number_hidden_layers indicating  whether a batch normalization
        "layer" is to be applied BEFORE the activation function
        """
        self.number_hidden_layers = number_hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_functions = init_fs
        self.act_functions = act_fs
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_probs = dropout_probs
        self.n_inputs = n_inputs

    def network_remove_layer(self, _):  # Defined just in case the user redefines classes and forgets to define this function
        pass

    def change_dimensions_in_layer(self, _, __):  # Defined just in case the user redefines classes and forgets to define this function
        pass

    def remove_random_layer(self):
        layer_pos = np.random.randint(self.number_hidden_layers)
        self.network_remove_layer(layer_pos)

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

    def change_dimensions_in_random_layer(self, max_layer_size):
        layer_pos = np.random.randint(self.number_hidden_layers)
        new_dim = np.random.randint(max_layer_size)+1
        self.change_dimensions_in_layer(layer_pos, new_dim)

    def change_dropout(self):
        rnd = np.random.choice(np.arange(0, self.dropout.shape[0]), size=np.random.randint(0, self.dropout.shape[0]), replace=False)

        self.dropout[rnd] -= 1

        self.dropout[rnd] = self.dropout[rnd]**2

    def change_batch_norm(self):

        rnd = np.random.choice(np.arange(0, self.batch_norm.shape[0]), size=np.random.randint(0, self.batch_norm.shape[0]), replace=False)

        self.batch_norm[rnd] -= 1

        self.batch_norm[rnd] = self.batch_norm[rnd]**2


class MLPDescriptor(NetworkDescriptor):
    def __init__(self, number_hidden_layers=1, input_dim=1, output_dim=1,  dims=None, init_fs=None, act_fs=None, dropout=(), batch_norm=()):
        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, init_fs=init_fs, act_fs=act_fs, dropout=dropout, batch_norm=batch_norm)
        self.dims = dims  # Number of neurons in each layer

    def random_init(self, input_size=None, output_size=None, nlayers=None, max_layer_size=None, _=None, __=None, no_drop=None, no_batch=None):
        # If the incoming/outgoing sizes have more than one dimension compute the size of the flattened sizes
        if input_size is not None:
            if hasattr(input_size, '__iter__'):
                self.input_dim = reduce(lambda x, y: x * y, input_size)
            else:
                self.input_dim = input_size
        if output_size is not None:
            if hasattr(output_size, '__iter__'):
                self.output_dim = reduce(lambda x, y: x * y, output_size)
            else:
                self.output_dim = output_size

        # Random initialization
        if nlayers is None and max_layer_size is not None:
            self.number_hidden_layers = np.random.randint(nlayers) + 1
            self.dims = [np.random.randint(4, max_layer_size) + 1 for _ in range(self.number_hidden_layers)]
            self.init_functions = np.random.choice(init_functions, size=self.number_hidden_layers + 1)
            self.act_functions = np.random.choice(act_functions, size=self.number_hidden_layers + 1)
        if no_batch is not None:
            if no_batch:
                self.batch_norm = np.zeros(self.number_hidden_layers + 1)
            else:
                self.batch_norm = np.random.randint(0, 2, size=self.number_hidden_layers + 1)
        if no_drop is not None:
            if no_drop:
                self.dropout = np.zeros(self.number_hidden_layers + 1)
                self.dropout_probs = np.zeros(self.number_hidden_layers + 1)
            else:
                self.dropout = np.random.randint(0, 2, size=self.number_hidden_layers + 1)
                self.dropout_probs = np.random.rand(self.number_hidden_layers + 1)

    def add_layer(self, layer_pos, lay_dims, init_w_function, init_a_function, dropout, drop_prob, batch_norm):
        """
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        :param layer_pos: Where the layer is added
        :param lay_dims: Dimension of the layer
        :param init_w_function: Weight initialization function of the layer
        :param init_a_function: Activation function of the layer
        :param dropout: Whether dropout is applied after the layer or not
        :param drop_prob: The probability of applying dropout to the neurons
        :param batch_norm: Whether batch normalization is applied to the layer
        :return: --
        """

        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos >= self.number_hidden_layers:
            return

        # We create the new layer and add it to the network descriptor
        self.dims = np.insert(self.dims, layer_pos, lay_dims)
        self.init_functions = np.insert(self.init_functions, layer_pos, init_w_function)
        self.act_functions = np.insert(self.act_functions, layer_pos, init_a_function)

        # Finally the number of hidden layers is updated
        self.number_hidden_layers = self.number_hidden_layers + 1
        if not (isinstance(self.batch_norm, tuple) or self.batch_norm.shape[0] == 0):
            self.batch_norm = np.insert(self.batch_norm, layer_pos, batch_norm)
        if not (isinstance(self.dropout, tuple) or self.dropout.shape[0] == 0):
            self.dropout = np.insert(self.dropout, layer_pos, dropout)
            self.dropout_probs = np.insert(self.dropout_probs, layer_pos, drop_prob)

    def remove_layer(self, layer_pos):
        """
        Adds a layer at a specified position, with a given  number of units, init weight
        function, activation function.
        If the layer is inserted in layer_pos in [0,number_hidden_layers] then all the
        other layers are shifted.
        If the layer is inserted in position number_hidden_layers+1, then it is just appended
        to previous layer and it will output output_dim variables.
        If the position for the layer to be added is not within feasible bounds
        in the current architecture, the function silently returns
        :param layer_pos: Layer to be removed
        :return: --
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

    def change_layer_dimension(self, layer_pos, new_dim=-1):
        # If not within feasible bounds, return
        if layer_pos < 0 or layer_pos > self.number_hidden_layers:
            return
        if new_dim == -1:
            new_dim = np.random.randint(5, 100)
        # If the dimension of the layer is identical to the existing one, return
        self.dims[layer_pos] = new_dim

    def print_components(self, identifier):
        print(identifier, ' n_hid:', self.number_hidden_layers)
        print(identifier, ' Dims:', self.dims)
        print(identifier, ' Init:', self.init_functions)
        print(identifier, ' Act:', self.act_functions)

    def codify_components(self, max_hidden_layers=10, ref_list_init_functions=init_functions, ref_list_act_functions=act_functions):

        max_total_layers = max_hidden_layers + 1
        # The first two elements of code are the number of layers and number of loops
        code = [[self.number_hidden_layers]]

        # We add all the layer dimension and fill with zeros all positions until max_layers
        code = code + [list(self.dims) + [-1]*(max_total_layers-len(self.dims))]

        # We add the indices of init_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_f = []
        for init_f in self.init_functions:
            aux_f.append(list(ref_list_init_functions).index(init_f))
        code = code + [aux_f + [-1]*(max_total_layers-len(aux_f))]

        # We add the indices of act_functions in each layer
        # and fill with zeros all positions until max_layers
        aux_a = []
        for act_f in self.act_functions:
            aux_a.append(list(ref_list_act_functions).index(act_f))
        code = code + [aux_a + [-1]*(max_total_layers-len(aux_a))]

        return code


class ConvDescriptor(NetworkDescriptor):
    def __init__(self, number_hidden_layers=2, input_dim=(28, 28, 3), output_dim=(7, 7, 1), op_type=(0, 1), filters=((3, 3, 2), (3, 3, 2)), strides=((1, 1, 1), (1, 1, 1)), init_fs=(0, 0), act_fs=(0, 0), dropout=(), batch_norm=()):
        """
        Descriptor for convolutional cells. Not working right now
        :param number_hidden_layers: Number
        :param input_dim:
        :param output_dim:
        :param op_type:
        :param filters:
        :param strides:
        :param init_fs:
        :param act_fs:
        :param dropout:
        :param batch_norm:
        """

        super().__init__(number_hidden_layers=number_hidden_layers, input_dim=input_dim, output_dim=output_dim, init_fs=init_fs, act_fs=act_fs, dropout=dropout, batch_norm=batch_norm)
        self.ops = op_type
        self.filters = filters
        self.strides = strides

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
            self.act_functions.insert(layer_pos, -1)
            self.init_functions.insert(layer_pos, -1)
            self.strides.insert(layer_pos, lay_params[0])
        elif "conv" in lay_type:
            self.strides.insert(layer_pos, lay_params[0])
            self.filters.insert(layer_pos, lay_params[1])
            self.act_functions.insert(layer_pos, lay_params[2])
            self.init_functions.insert(layer_pos, lay_params[3])

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
        self.act_functions.pop(layer_pos)
        self.init_functions.pop(layer_pos)
        self.strides.pop(layer_pos)

    def network_remove_random_layer(self):
        layer_pos = np.random.randint(len(self.filters))
        self.network_remove_layer(layer_pos)

    def change_activation_fn_in_layer(self, layer_pos, new_act_fn):
        self.act_functions[layer_pos] = new_act_fn

    def change_weight_init_fn_in_layer(self, layer_pos, new_weight_fn):
        self.init_functions[layer_pos] = new_weight_fn

    def change_filters_in_layer(self, layer_pos, new_kernel_size):
        self.filters[layer_pos] = new_kernel_size

    def change_stride_in_layer(self, layer_pos, new_stride):
        self.strides[layer_pos] = new_stride

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


class ConvDescriptor:
    def __init__(self, n_inputs, input_dim, output_dim, layers, filters, strides, sizes, act_fns, init_fns):
        self.n_inputs = n_inputs
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
        # print("input_dim", self.input_dim)
        # print("output_dim", self.output_dim)
        # print("filters", filters)
        # print("sizes", sizes)
        # print("strides", strides)
        # print("init_funcs", init_funcs)
        # print("act_funcs", act_funcs)
        # print("layers", layers)
        # print("str", str(self.input_dim) + "_" + str(self.output_dim) + "_" + ",".join(filters) + "*" + ",".join(["/".join(szs) for szs in sizes]) + "*" + ",".join(strides) + "_" + ",".join(init_funcs) + "_" + ",".join(act_funcs) + "_" + ",".join(layers))
        return str(self.input_dim) + "_" + str(self.output_dim) + "_" + ",".join(filters) + "*" + ",".join(["/".join(szs) for szs in sizes]) + "*" + ",".join(layers) + "*" + ",".join(strides) + "_" + ",".join(init_funcs) + "_" + ",".join(act_funcs)


class ConvolutionDescriptor(ConvDescriptor):
    def __init__(self, n_inputs, input_dim, output_dim, layers, filters, strides, sizes, act_fns, init_fns):
        super().__init__(n_inputs, input_dim, output_dim, layers, filters, strides, sizes, act_fns, init_fns)

    def codify_components(self):
        return str(self.n_inputs) + "_" + type(self).__name__[:-10] + "_" + self.codify_components() + "_"


class GenericDescriptor(MLPDescriptor):
    def __init__(self, inp, outp, n_inputs, n_hidden, dims, inits, acts):
        acts = np.append(acts[:-1], [None])
        super().__init__(number_hidden_layers=n_hidden, input_dim=inp, output_dim=outp, init_fs=inits, act_fs=acts)
        self.dims = dims

    def codify_components(self, max_hidden_layers=10, ref_list_init_functions=init_functions, ref_list_act_functions=act_functions):
        return str(self.n_inputs) + "_" + type(self).__name__[:-10] + "_" + "_".join([str(",".join(str(y) for y in x)) for x in super().codify_components()]) + "_"


class DiscreteDescriptor(MLPDescriptor):
    def __init__(self, inp, outp, n_inputs, n_hidden, dims, inits, acts):
        acts = np.append(acts[:-1], [tf.nn.softmax])
        super().__init__(number_hidden_layers=n_hidden, input_dim=inp, output_dim=outp, init_fs=inits, act_fs=acts)
        self.dims = dims

    def codify_components(self, max_hidden_layers=10, ref_list_init_functions=init_functions, ref_list_act_functions=act_functions):
        return str(self.n_inputs) + "_" + type(self).__name__[:-10] + "_" + "_".join([str(",".join(str(y) for y in x)) for x in super().codify_components()]) + "_"


class DecoderDescriptor(MLPDescriptor):
    def __init__(self, inp, outp, n_inputs, n_hidden, dims, inits, acts, conds):
        super().__init__(number_hidden_layers=n_hidden, input_dim=inp, output_dim=outp, init_fs=inits, act_fs=acts)
        self.dims = dims

        self.conds = conds[0]
        self.rands = conds[1]

    def codify_components(self, max_hidden_layers=10, ref_list_init_functions=init_functions, ref_list_act_functions=act_functions):
        return str(self.n_inputs) + "_" + type(self).__name__[:-10] + "_" + "_".join([str(",".join(str(y) for y in x)) for x in super().codify_components()]) + "_" + ",".join([str(x) for x in self.conds]) + "-" + ",".join([str(x) for x in self.rands])


class ConvDecoderDescriptor(ConvDescriptor):

    def __init__(self, n_inputs, x_dim, z_dim, layers, filters, strides, act_fns, init_fns, outs, kwargs):
        super().__init__(n_inputs, z_dim, x_dim, layers, filters, strides, outs, act_fns, init_fns)
        self.conds = kwargs[0]
        self.rands = kwargs[1]

    def codify_components(self):

        return str(self.n_inputs) + "_" + type(self).__name__[:-10] + "_" + super().codify_components() + "_" + ".".join([str(x) for x in self.conds]) + "-" + ".".join([str(x) for x in self.rands])
