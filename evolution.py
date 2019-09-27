import tensorflow as tf
import numpy as np
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from Network import Network, init_functions, act_functions
from auxiliary_functions import batch, init_mlp_desc
from metrics import mse, accuracy_error
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    def __init__(self, loss, n_nets, x_trains, y_trains, x_tests, y_tests, evaluation, batch_size, population, generations, iters=1000, lrate=0.01, sel=0, n_layers=10, max_layer_size=100, n_inputs=None, n_outputs=None, seed=0, cxp=0, mtp=1, evol_kwargs={}, sel_kwargs={}, ev_alg=1):

        np.random.seed(seed)
        tf.random.set_random_seed(seed)
        random.seed(seed)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.network_descriptor = {}
        self.nlayers = n_layers
        self.max_lay = max_layer_size
        self.n_nets = n_nets
        self.loss_function = None
        self.evaluation = None
        self.define_loss_eval(loss, evaluation)

        self.sess = None
        self.opt = None
        self.batch_size = batch_size
        self.lf = None
        self.predictions = {}
        self.inp_placeholders = {}
        self.out_placeholders = {}
        self.lrate = lrate
        self.iters = iters
        self.train_inputs = {}
        self.train_outputs = {}
        self.test_inputs = {}
        self.test_outputs = {}
        self.data_save(x_trains, y_trains, x_tests, y_tests)
        self.example_num = self.train_inputs["i0"].shape[0]

        self.toolbox = base.Toolbox()
        self.ev_alg = ev_alg
        self.cXp = cxp
        self.mtp = mtp
        self.evol_kwargs = {}
        self.evol_function = ev_alg
        self.generations = generations
        self.population_size = population
        self.initialize_deap(sel, sel_kwargs, evol_kwargs)

    def data_save(self, x_trains, y_trains, x_tests, y_tests):

        for i, x in enumerate(x_trains):
            self.train_inputs["i" + str(i)] = x
            self.train_outputs["o" + str(i)] = y_trains[i]

        for i, x in enumerate(x_tests):
            self.test_inputs["i" + str(i)] = x
            self.test_outputs["o" + str(i)] = y_tests[i]

    def define_loss_eval(self, loss, evaluation):

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

    def initialize_deap(self, sel, sel_kwargs, ev_kwargs):

        deap_algs = [algorithms.eaSimple, algorithms.eaMuPlusLambda, algorithms.eaMuCommaLambda, algorithms.eaGenerateUpdate]

        creator.create("Fitness", base.Fitness, weights=[-1.0]*len(self.test_outputs))

        creator.create("Individual", MyContainer, fitness=creator.Fitness)

        self.toolbox.register("individual", self.init_individual, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.eval_individual)
        # self.toolbox.register("mate", cx_gan)
        self.toolbox.register("mutate", self.mutations)

        sel_methods = [tools.selBest, tools.selTournament, tools.selNSGA2]

        self.toolbox.register("select", sel_methods[sel], **sel_kwargs)

        if len(ev_kwargs) == 0:
            if self.ev_alg == 0:
                self.evol_kwargs = {"cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}
            if self.ev_alg == 1:
                self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, "cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}
            if self.ev_alg == 2:
                self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, "cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}

        self.ev_alg = deap_algs[self.ev_alg]

    def evolve(self):

        pop = self.toolbox.population(n=self.population_size)
        hall_of = tools.HallOfFame(self.population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        result, log_book = self.ev_alg(pop, self.toolbox, **self.evol_kwargs, stats=stats, halloffame=hall_of)

        return result, log_book, hall_of

    def init_individual(self, init_ind):

        network_descriptor = {}

        if self.n_nets == 1:
            network_descriptor["n0"] = init_mlp_desc(self.train_inputs["i0"].shape[1:], self.train_outputs["o0"].shape[1], self.nlayers, self.max_lay)

        else:
            for net in range(self.n_nets):
                network_descriptor["n" + str(net)] = init_mlp_desc(self.n_inputs[net], self.n_outputs[net], self.nlayers, self.max_lay)
        return init_ind(network_descriptor)

    def eval_individual(self, individual):

        graph = tf.Graph()

        if self.n_nets == 1:
            ev = self.single_net_eval(individual, graph)
        else:
            ev = self.eval_multinetwork(individual, graph)
        return ev

    def single_net_eval(self, individual, graph):
        net = Network(individual.descriptor_list["n0"])
        with graph.as_default():
            self.sess = tf.Session()
            for i, key in enumerate(self.train_inputs.keys()):
                inp = self.train_inputs[key]
                self.inp_placeholders[key] = tf.placeholder(tf.float32, shape=[None] + [i for i in inp.shape[1:]])
            self.out_placeholders["o0"] = tf.placeholder(tf.float32, shape=[None] + [i for i in self.train_outputs["o0"].shape[1:]])

            inp = tf.concat([tf.layers.flatten(self.inp_placeholders[i]) for i in self.inp_placeholders.keys()], axis=1)
            net.network_initialization(graph)
            out = net.network_building(inp, graph)
            self.predictions["o0"] = out

            self.lf = self.loss_function(self.out_placeholders["o0"], out)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lrate).minimize(self.lf)
            self.sess.run(tf.global_variables_initializer())
            self.train_single_network(graph)
            ev = self.evaluate_network()
            self.sess.close()
        return ev

    def eval_multinetwork(self, individual, graph):
        nets = {}
        with graph.as_default():
            self.sess = tf.Session()
            for i in range(len(self.n_inputs)):
                self.inp_placeholders["i" + str(i)] = tf.placeholder(tf.float32, shape=[None] + [j for j in self.n_inputs[i]], name="i" + str(i))

            for i in range(len(self.n_outputs)):
                self.out_placeholders["o" + str(i)] = tf.placeholder(tf.float32, shape=[None] + [j for j in self.n_outputs[i]], name="o" + str(i))

            for index, net in enumerate(individual.descriptor_list.keys()):
                nets[net] = Network(individual.descriptor_list[net])
                nets[net].network_initialization(graph)

            predictions = self.loss_function(nets, {"in": self.inp_placeholders, "out": self.out_placeholders}, self.sess, graph, self.train_inputs, self.train_outputs, self.batch_size)

            ev = self.evaluation(predictions, self.inp_placeholders, self.sess, graph, self.test_inputs, self.test_outputs)
        return ev

    def train_single_network(self, graph):

        aux_ind = 0
        for i in range(self.iters):
            feed_dict = {}
            for inp in self.train_inputs:
                feed_dict[self.inp_placeholders[inp]] = batch(self.train_inputs[inp], self.batch_size, aux_ind)
            for output in self.train_outputs:
                feed_dict[self.out_placeholders[output]] = batch(self.train_outputs[output], self.batch_size, aux_ind)

            aux_ind = (aux_ind + self.batch_size) % self.example_num
            with graph.as_default():
                _, partial_loss = self.sess.run([self.opt, self.lf], feed_dict=feed_dict)

    def evaluate_network(self):

        feed_dict = {}
        for inp in self.train_inputs:
            feed_dict[self.inp_placeholders[inp]] = self.test_inputs[inp]

        res = self.sess.run(self.predictions, feed_dict=feed_dict)

        pred = np.argmax(res["o0"], axis=1)
        real = np.argmax(self.test_outputs["o0"], axis=1)

        return self.evaluation(pred, real),

    def mutations(self, individual):

        mutation_types = ["add_layer", "network_loops", "del_layer", "weigt_init", "activation", "dimension", "divergence", "latent", "lrate"]

        network = individual.descriptor_list[np.random.choice(list(individual.descriptor_list.keys()))]

        type_mutation = np.random.choice(mutation_types)

        if type_mutation == "add_layer":             # We add one layer
            layer_pos = np.random.randint(network.number_hidden_layers)+1
            lay_dims = np.random.randint(self.max_lay)+1
            init_w_function = init_functions[np.random.randint(len(init_functions))]
            init_a_function = act_functions[np.random.randint(len(act_functions))]
            network.network_add_layer(layer_pos, lay_dims, init_w_function, init_a_function)

        elif type_mutation == "del_layer":              # We remove one layer
            network.network_remove_random_layer()

        elif type_mutation == "weigt_init":             # We change weight initialization function in all layers
            init_w_function = init_functions[np.random.randint(len(init_functions))]
            network.change_all_weight_init_fns(init_w_function)

        elif type_mutation == "activation":             # We change the activation function in layer
            layer_pos = np.random.randint(network.number_hidden_layers)
            init_a_function = act_functions[np.random.randint(len(act_functions))]
            network.change_activation_fn_in_layer(layer_pos, init_a_function)

        elif type_mutation == "dimension":              # We change the number of neurons in layer
            network.change_dimensions_in_random_layer(self.max_lay)

        return individual,
