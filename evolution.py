import tensorflow as tf
import numpy as np
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from Network import MLP, MLPDescriptor, initializations, activations, TCNN, CNN, ConvDescriptor, TConvDescriptor
from auxiliary_functions import batch
from metrics import mse, accuracy_error
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

descs = {"ConvDescriptor": CNN, "MLPDescriptor": MLP, "TConvDescriptor": TCNN}

# This function set the seeds of the tensorflow function
# to make this notebook's output stable across runs


#####################################################################################################

class MyContainer(object):
    # This class does not require the fitness attribute
    # it will be  added later by the creator
    def __init__(self, descriptor_list):
        # Some initialisation with received values
        self.descriptor_list = descriptor_list


class Evolving:
    def __init__(self, loss="XEntropy", desc_list=(MLPDescriptor, ), complex=False, x_trains=None, y_trains=None, x_tests=None, y_tests=None, evaluation="Accuracy_error", n_inputs=((28, 28)), n_outputs=((10)), batch_size=100, population=20, generations=20, iters=10, lrate=0.01, sel=0, n_layers=10, max_layer_size=100, max_filter=4, max_stride=3, seed=0, cxp=0, mtp=1, no_dropout=False, no_batch_norm=False, evol_kwargs={}, sel_kwargs={}, ev_alg=1, hyperparameters={}, add_obj=0):
        """
        This is the main class in charge of evolving model descriptors.
        """

        np.random.seed(seed)
        tf.random.set_random_seed(seed)
        random.seed(seed)

        self.n_inputs = n_inputs                                        # List of lists with the dimensions of the inputs of each network
        self.n_outputs = n_outputs                                      # List of lists with the dimensions of the outputs of each network

        self.network_descriptor = {}                                    # dict {"net_id": net_desc}
        self.nlayers = n_layers                                         # Maximum number of layers
        self.max_lay = max_layer_size                                   # Maximum number of neurons per layer
        self.max_filter = max_filter                                    # Maximum size of filter
        self.max_stride = max_stride                                    # Maximum stride
        self.descriptors = desc_list                                    # Number of MLPs in the model
        self.loss_function = None                                       # Loss function to be used for training the model
        self.evaluation = None                                          # Function for evaluating the model
        self.define_loss_eval(loss, evaluation)                         # Assigning the values of the previous variables

        self.sess = None                                                # Tensorflow Session
        self.batch_size = batch_size                                    # Batch size for training
        self.predictions = {}                                           # dict {"Net_id": net_output}
        self.inp_placeholders = {}                                      # Placeholders for model inputs
        self.out_placeholders = {}                                      # Placeholders for model outputs
        self.lrate = lrate                                              # Learning rate
        self.iters = iters                                              # Number of training epochs
        self.train_inputs = {}                                          # Training data (X)
        self.train_outputs = {}                                         # Training data (y)
        self.test_inputs = {}                                           # Test data (X)
        self.test_outputs = {}                                          # Test data (y)
        self.data_save(x_trains, y_trains, x_tests, y_tests)            # Save data in the previous dicts
        self.complex = (type(loss) is not str) or (type(evaluation) is not str) or complex or len(self.descriptors) > 1 or self.descriptors[0] is not MLPDescriptor or len(hyperparameters) > 0

        self.toolbox = base.Toolbox()
        self.ev_alg = None                                              # DEAP evolutionary algorithm function
        self.cXp = cxp if len(desc_list) > 1 else 0                     # Crossover probability. 0 in the simple case
        self.mtp = mtp if len(desc_list) > 1 else 1                     # Mutation probability. 1 in the simple case
        self.evol_kwargs = {}                                           # Parameters for the main DEAP function
        self.evol_function = ev_alg                                     # main DEAP function index
        self.generations = generations                                  # Number of generations
        self.population_size = population                               # Individuals in a population
        self.ev_hypers = hyperparameters                                # Hyperparameters to be evolved (e.g., optimizer, batch size)
        self.initialize_deap(sel, sel_kwargs, ev_alg, evol_kwargs, no_batch_norm, no_dropout, add_obj)      # Initialize DEAP-related matters
        # Add model saving parameter

    def data_save(self, x_trains, y_trains, x_tests, y_tests):
        """
        Filling the dicts with the training data
        :param x_trains: X data for training
        :param y_trains: y data for training
        :param x_tests: X data for testing
        :param y_tests: y data for testing
        :return: --
        """

        if isinstance(x_trains, dict):
            self.train_inputs = x_trains
            self.train_outputs = y_trains

            self.test_inputs = x_tests
            self.test_outputs = y_tests
        else:
            for i, x in enumerate(x_trains):
                self.train_inputs["i" + str(i)] = x
                self.train_outputs["o" + str(i)] = y_trains[i]

            for i, x in enumerate(x_tests):
                self.test_inputs["i" + str(i)] = x
                self.test_outputs["o" + str(i)] = y_tests[i]

    def define_loss_eval(self, loss, evaluation):
        """
        Define the loss and evaluation function. Writing the (string) names of the predefined functions is accepted.
        :param loss: Loss function. Either string (predefined) or customized by the user.
        :param evaluation: Evaluation function. Either string (predefined) or customized by the user.
        :return:
        """

        losses = {"MSE": tf.losses.mean_squared_error, "XEntropy": tf.losses.softmax_cross_entropy}
        evals = {"MSE": mse, "Accuracy_error": accuracy_error}

        if type(loss) is str:
            self.loss_function = losses[loss]
        else:
            self.loss_function = loss

        if type(evaluation) is str:
            self.evaluation = evals[evaluation]
        else:
            self.evaluation = evaluation

    def initialize_deap(self, sel, sel_kwargs, ev_alg, ev_kwargs, no_batch, no_drop, add_obj):
        """
        Initialize DEAP algorithm
        :param sel: Selection method
        :param sel_kwargs: Hyperparameters for the selection methods, e.g., size of the tournament if that method is selected
        :param ev_alg: DEAP evolutionary algorithm (EA)
        :param ev_kwargs: Hyperparameters for the EA, e.g., mutation or crossover probability.
        :param no_batch: Whether the evolutive process includes batch normalization in the networks or not
        :param no_drop: Whether the evolutive process includes dropout in the networks or not
        :return: --
        """

        deap_algs = [algorithms.eaSimple, algorithms.eaMuPlusLambda, algorithms.eaMuCommaLambda, algorithms.eaGenerateUpdate]

        creator.create("Fitness", base.Fitness, weights=[-1.0]*(len(self.test_outputs) + add_obj))

        creator.create("Individual", MyContainer, fitness=creator.Fitness)

        self.toolbox.register("individual", self.init_individual, creator.Individual, no_batch, no_drop)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.eval_individual)
        self.toolbox.register("mate", cross, creator.Individual)
        self.toolbox.register("mutate", mutations, self.ev_hypers, self.max_lay, no_batch, no_drop)

        sel_methods = [tools.selBest, tools.selTournament, tools.selNSGA2]

        self.toolbox.register("select", sel_methods[sel], **sel_kwargs)

        if len(ev_kwargs) == 0:
            if ev_alg == 0:
                self.evol_kwargs = {"cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}
            if ev_alg == 1:
                self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, "cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}
            if ev_alg == 2:
                self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, "cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}

        self.ev_alg = deap_algs[ev_alg]

    def evolve(self):
        """
        Actual evolution of individuals
        :return: The last generation, a log book (stats) and the hall of fame (the best individuals found)
        """

        pop = self.toolbox.population(n=self.population_size)
        hall_of = tools.HallOfFame(self.population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        result, log_book = self.ev_alg(pop, self.toolbox, **self.evol_kwargs, stats=stats, halloffame=hall_of)

        return result, log_book, hall_of

    def init_individual(self, init_ind, no_batch, no_drop):
        """
        Creation of a single individual
        :param init_ind: DEAP function for transforming a network descriptor, or a list of descriptors + evolvable
        hyperparameters into a DEAP individual
        :return: a DEAP individual
        """

        network_descriptor = {}

        if not self.complex:  # Simple case
            network_descriptor["n0"] = MLPDescriptor()
            network_descriptor["n0"].random_init(self.train_inputs["i0"].shape[1:], self.train_outputs["o0"].shape[1], self.nlayers, self.max_lay, None, None, no_drop, no_batch)
        else:  # Custom case
            for i, descriptor in enumerate(self.descriptors):
                network_descriptor["n" + str(i)] = descriptor()
                network_descriptor["n" + str(i)].random_init(self.n_inputs[i], self.n_outputs[i], self.nlayers, self.max_lay, self.max_stride, self.max_filter, no_drop, no_batch)
        network_descriptor["hypers"] = {}
        if len(self.ev_hypers) > 0:

            for hyper in self.ev_hypers:
                network_descriptor["hypers"][hyper] = np.random.choice(self.ev_hypers[hyper])

        return init_ind(network_descriptor)

    def eval_individual(self, individual):
        """
        Function for evaluating an individual.
        :param individual: DEAP individual
        :return: Fitness value.
        """

        graph = tf.Graph()

        if not self.complex:
            ev = self.single_net_eval(individual, graph)
        else:
            ev = self.eval_multinetwork(individual, graph)
        return ev

    def single_net_eval(self, individual, graph):
        """
        Function for evolving a single individual. No need of the user providing a evaluation function
        :param individual: DEAP individual
        :param graph: Graph where the tensorflow variables were defined
        :return: Fitness value
        """
        net = MLP(individual.descriptor_list["n0"])
        with graph.as_default():
            self.sess = tf.Session()
            for i, key in enumerate(self.train_inputs.keys()):
                inp = self.train_inputs[key]
                self.inp_placeholders[key] = tf.placeholder(tf.float32, shape=[None] + [i for i in inp.shape[1:]])
            self.out_placeholders["o0"] = tf.placeholder(tf.float32, shape=[None] + [i for i in self.train_outputs["o0"].shape[1:]])

            inp = tf.concat([tf.layers.flatten(self.inp_placeholders[i]) for i in self.inp_placeholders.keys()], axis=1)
            net.initialization(graph)
            out = net.building(inp, graph)
            self.predictions["o0"] = out

            lf = self.loss_function(self.out_placeholders["o0"], out)
            opt = tf.train.AdamOptimizer(learning_rate=self.lrate).minimize(lf)
            self.sess.run(tf.global_variables_initializer())
            self.train_single_network(graph, opt, lf)
            ev = self.evaluate_network(graph)
            self.sess.close()
        return ev

    def eval_multinetwork(self, individual, graph):
        """
        Function for evaluating a DEAP individual consisting of more than a single MLP. The user must have implemented the
        training and evaluation functions.
        :param individual: DEAP individual
        :param graph: Tensorflow graph where the tensorflow variables were defined
        :return: Fitness value
        """
        nets = {}
        with graph.as_default():
            self.sess = tf.Session()
            for i in range(len(self.n_inputs)):
                self.inp_placeholders["i" + str(i)] = tf.placeholder(tf.float32, shape=[None] + [j for j in self.n_inputs[i]], name="i" + str(i))
            for i in range(len(self.n_outputs)):
                self.out_placeholders["o" + str(i)] = tf.placeholder(tf.float32, shape=[None] + [j for j in self.n_outputs[i]], name="o" + str(i))

            for index, net in enumerate(individual.descriptor_list.keys()):
                if "hypers" not in net:
                    nets[net] = descs[self.descriptors[index].__name__](individual.descriptor_list[net])
                    nets[net].initialization(graph)

            predictions = self.loss_function(nets, {"in": self.inp_placeholders, "out": self.out_placeholders}, self.sess, graph, self.train_inputs, self.train_outputs, self.batch_size, individual.descriptor_list["hypers"])

            ev = self.evaluation(predictions, self.inp_placeholders, self.sess, graph, self.test_inputs, self.test_outputs, individual.descriptor_list["hypers"])

            self.sess.close()
        return ev

    def train_single_network(self, graph, opt, lf):
        """
        For the simple case of learning a single MLP, training
        :param graph: Graph in which the tensorflow variables were created
        :param opt: Tensorflow optimizer
        :param lf: Loss function being optimized. Only for visualization purposes
        :return: --
        """

        aux_ind = 0
        for i in range(self.iters):
            feed_dict = {}
            for inp in self.train_inputs:
                feed_dict[self.inp_placeholders[inp]] = batch(self.train_inputs[inp], self.batch_size, aux_ind)
            for output in self.train_outputs:
                feed_dict[self.out_placeholders[output]] = batch(self.train_outputs[output], self.batch_size, aux_ind)

            aux_ind = (aux_ind + self.batch_size) % self.train_inputs["i0"].shape[0]
            with graph.as_default():
                _, partial_loss = self.sess.run([opt, lf], feed_dict=feed_dict)

    def evaluate_network(self, graph):
        """
        Function for evaluating a MLP in the simple case
        :param graph: Tensorflow graph where the model is created
        :return: Fitness value
        """

        feed_dict = {}
        for inp in self.train_inputs:
            feed_dict[self.inp_placeholders[inp]] = self.test_inputs[inp]
        with graph.as_default():
            res = self.sess.run(self.predictions, feed_dict=feed_dict)

        return self.evaluation(res["o0"], self.test_outputs["o0"]),


def mutations(ev_hypers, max_lay, batch, drop, individual):
    """
    Mutation operators for individuals. They can affect any network or the hyperparameters.
    :param individual: DEAP individual. Contains a dict where the keys are the components of the model
    :return: Mutated version of the DEAP individual.
    """

    nets = list(individual.descriptor_list.keys())

    nets.remove("hypers")

    network = individual.descriptor_list[np.random.choice(nets)]

    if isinstance(network, MLPDescriptor):
        mutation_types = ["add_layer", "del_layer", "weight_init", "activation", "dimension", "hyper"]
    elif isinstance(network, ConvDescriptor):
        mutation_types = ["add_conv_layer", "del_conv_layer", "weight_init", "activation", "stride_conv", "filter_conv", "hyper"]
    elif isinstance(network, TConvDescriptor):
        mutation_types = ["add_deconv_layer", "del_deconv_layer", "weight_init", "activation", "stride_deconv", "filter_deconv", "hyper"]
    else:
        mutation_types = []

    if not batch:
        mutation_types = ["batch_norm"] + mutation_types
    if not drop:
        mutation_types = ["dropout"] + mutation_types

    type_mutation = np.random.choice(mutation_types if len(ev_hypers) > 0 else mutation_types[:-1])

    if type_mutation == "add_layer":             # We add one layer
        layer_pos = np.random.randint(network.number_hidden_layers)+1
        lay_dims = np.random.randint(max_lay)+1
        init_w_function = initializations[np.random.randint(len(initializations))]
        init_a_function = activations[np.random.randint(len(activations))]
        if not drop:
            dropout = np.random.randint(0, 2)
            drop_prob = np.random.rand()
        else:
            dropout = 0
            drop_prob = 0
        if not batch:
            batch_norm = np.random.randint(0, 2)
        else:
            batch_norm = 0

        network.add_layer(layer_pos, lay_dims, init_w_function, init_a_function, dropout, drop_prob, batch_norm)

    elif type_mutation == "del_layer":              # We remove one layer
        network.remove_random_layer()

    elif type_mutation == "weight_init":             # We change weight initialization function in all layers
        layer_pos = np.random.randint(network.number_hidden_layers)
        init_w_function = initializations[np.random.randint(len(initializations))]
        network.change_weight_init(layer_pos, init_w_function)

    elif type_mutation == "activation":             # We change the activation function in layer
        layer_pos = np.random.randint(network.number_hidden_layers)
        init_a_function = activations[np.random.randint(len(activations))]
        network.change_activation(layer_pos, init_a_function)

    elif type_mutation == "dimension":              # We change the number of neurons in layer
        network.change_layer_dimension(max_lay)

    elif type_mutation == "dropout":
        network.change_dropout()

    elif type_mutation == "batch_norm":
        network.change_batch_norm()

    elif type_mutation == "hyper":                  # We change the value of a hyperparameter for another value
        h = np.random.choice(list(ev_hypers.keys()))  # We select the hyperparameter to be mutated
        # We choose two values, just in case the first one is the one already selected
        new_value = np.random.choice(ev_hypers[h], size=2, replace=False)
        if individual.descriptor_list["hypers"][h] == new_value[0]:
            individual.descriptor_list["hypers"][h] = new_value[1]
        else:
            individual.descriptor_list["hypers"][h] = new_value[0]
    elif type_mutation == "add_conv_layer":
        if network.shapes[-1][0] > 2 and network.shapes[-1][1] > 2:
            network.add_layer(np.random.randint(0, network.number_hidden_layers), np.random.randint(0, 3), [1, np.random.randint(2, 4), np.random.choice(activations[1:]), np.random.choice(initializations[1:])])

    elif type_mutation == "del_conv_layer":
        if network.number_hidden_layers > 1:
            network.remove_random_layer()
    elif type_mutation == "stride_conv":
        layer = np.random.randint(0, network.number_hidden_layers)
        if network.strides[layer][0] == 1 and network.shapes[-1][0] >= 4:
            network.change_stride(layer, 2)
        elif network.strides[layer][0] == 2:
            network.change_stride(layer, 1)
    elif type_mutation == "filter_conv":
        layer = np.random.randint(0, network.number_hidden_layers)
        channels = np.random.randint(0, 65)
        if network.filters[layer][0] == 2 and network.shapes[-1][0] >= 3:
            network.change_filters(layer, 3, channels)
        elif network.filters[layer][0] == 3:
            network.change_filters(layer, 2, channels)
    elif type_mutation == "add_deconv_layer":
        network.add_layer(np.random.randint(0, network.number_hidden_layers), [1, np.random.randint(2, 4), np.random.choice(activations[1:]), np.random.choice(initializations[1:])])
    elif type_mutation == "del_deconv_layer":
        network.remove_random_layer()
    elif type_mutation == "stride_deconv":
        layer = np.random.randint(0, network.number_hidden_layers)
        if network.strides[layer][1] == 2 and network.output_shapes[-1][1] / 2 >= network.output_dim[1]:
            network.change_stride(layer, 1)
        elif network.strides[layer][1] == 1:
            network.change_stride(layer, 2)
    elif type_mutation == "filter_deconv":
        layer = np.random.randint(0, network.number_hidden_layers)

        if network.filters[layer][1] == 3 and network.output_shapes[-1][1] - 6 >= network.output_dim[1]:
            network.change_filters(layer, 2, np.random.randint(0, 65))
        elif network.filters[layer][1] == 2:
            network.change_filters(layer, 3, np.random.randint(0, 65))

    return individual,


def cross(init_ind, ind1, ind2):
    """
    Crossover operator for individuals. Cannot be applied in the simple case, as it randomly interchanges model components
    :param init_ind: DEAP function for initializing dicts (in this case) as DEAP individuals
    :param ind1: 1st individual to be crossed
    :param ind2: 2st individual to be crossed
    :return: Two new DEAP individuals, the crossed versions of the incoming parameters.
    """

    keys = list(ind1.descriptor_list.keys())
    # We randomly select the keys of the components that will be interchanged.
    cx_point = np.random.choice(keys, size=np.random.randint(1, len(keys)) if len(keys) > 2 else 1, replace=False)
    new1 = {}
    new2 = {}

    for key in keys:
        if key in cx_point:
            new1[key] = ind1.descriptor_list[key]
            new2[key] = ind2.descriptor_list[key]
        else:
            new1[key] = ind2.descriptor_list[key]
            new2[key] = ind1.descriptor_list[key]

    return init_ind(new1), init_ind(new2)
