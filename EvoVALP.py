"""
This is a use case of EvoFlow

Another example of a multinetwork model, a GAN. In order to give an automatic fitness fuction to each GAN, we use the Inception Score (IS, https://arxiv.org/pdf/1606.03498.pdf)
We use the MobileNet model instead of Inception because it gave better accuracy scores when training it.
"""

import numpy as np
from evolution import Evolving
import argparse
from VALP.ModelDescriptor import MNMDescriptor, recursive_creator, fix_in_out_sizes
from VALP.evolution import diol
from VALP.small_improvements import bypass, del_con, add_con, divide_con, load_model
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from metrics import mse, accuracy_error
from VALP.Model_Building import MNM
from Network import initializations, activations
from PIL import Image
import tensorflow as tf


class MyContainer(object):
    # This class does not require the fitness attribute
    # it will be  added later by the creator
    def __init__(self, descriptor):
        # Some initialisation with received values
        self.descriptor = descriptor


def prepare_image(image, target_width=90, target_height=90, max_zoom=0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    height = image.size[0]
    width = image.size[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height

    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)

    # Next, we can select a random location on the image for this bounding box.
    x0 = np.random.randint(0, width - crop_width) if crop_vertically else 0
    y0 = np.random.randint(0, height - crop_height) if not crop_vertically else 0
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    # Let's crop the image using the random bounding box we built.

    image = image.crop((x0, y0, x1, y1))

    # Now, let's resize the image to the target dimensions.
    image = image.resize((target_width, target_height))
    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from 0.0 to 1.0 (for now):
    return np.array(image) / 255


def inception_score(images):
    height, width = 90, 90

    images = np.array([np.array(Image.fromarray(x, mode="RGB").resize((height, width))) for x in np.reshape(images, (-1, 28, 28, 3))]) / 255.

    with loaded_model[0].as_default():
        predictions = loaded_model[1].predict(images)
    preds = np.argmax(predictions, axis=1)
    aux_preds = np.zeros(10)
    unique, counts = np.unique(preds, return_counts=True)
    for number, appearances in zip(unique, counts):
        aux_preds[number] = appearances
    aux_preds = aux_preds / predictions.shape[0]
    predictions = np.sort(predictions, axis=1)
    predictions = np.mean(predictions, axis=0)

    sam_error = np.sum([aux_preds[w] * np.log(aux_preds[w] / predictions[w]) if aux_preds[w] > 0 else 0 for w in range(predictions.shape[0])])

    return sam_error


class EvoVALP(Evolving):
    def __init__(self, desc=MNMDescriptor, x_trains=None, y_trains=None, x_tests=None, y_tests=None, evaluation={"o0": mse, "o1": accuracy_error, "o2": inception_score},
                 batch_size=100, population=20, generations=20, iters=10, lrate=0.01, sel=0, n_layers=10, max_layer_size=100, max_net=10, max_con=15,
                 seed=0, cxp=0, mtp=1, no_dropout=True, no_batch_norm=True, evol_kwargs={}, sel_kwargs={}, ev_alg=3, hyperparameters={}, add_obj=0):
        """
        This is the main class in charge of evolving model descriptors.
        """

        super().__init__(loss=None, desc_list=[0, 0], complex=False, x_trains=x_trains, y_trains=y_trains, x_tests=x_tests, y_tests=y_tests,
                         evaluation=evaluation, n_inputs=[x_trains[x].shape[1:] for x in x_trains], n_outputs=[y_trains[x].shape[1:] for x in y_trains],
                         batch_size=batch_size, population=population, generations=generations, iters=iters, lrate=lrate, sel=sel, n_layers=n_layers,
                         max_layer_size=max_layer_size, seed=seed, cxp=cxp, mtp=mtp, no_dropout=no_dropout, no_batch_norm=no_batch_norm,
                         evol_kwargs=evol_kwargs, sel_kwargs=sel_kwargs, ev_alg=ev_alg, hyperparameters=hyperparameters, add_obj=add_obj)

        self.descriptor = desc  # Number of MLPs in the model
        self.evaluation = evaluation
        self.cXp = cxp
        self.mtp = mtp
        self.initialize_deap(sel, sel_kwargs, ev_alg, evol_kwargs, no_batch_norm, no_dropout, add_obj)  # Initialize DEAP-related matters
        self.max_net = max_net
        self.max_con = max_con
        self.population = None
        self.no_batch = no_batch_norm
        self.no_drop = no_dropout

        self.accs = []
        self.mses = []
        self.ids = []

        self.acc_args = []
        self.mse_args = []
        self.id_args = []

    def initialize_deap(self, sel, sel_kwargs, ev_alg, ev_kwargs, no_batch, no_drop, add_obj):
        """
        Initialize DEAP algorithm
        :param sel: Selection method
        :param sel_kwargs: Hyperparameters for the selection methods, e.g., size of the tournament if that method is selected
        :param ev_alg: DEAP evolutionary algorithm (EA)
        :param ev_kwargs: Hyperparameters for the EA, e.g., mutation or crossover probability.
        :param no_batch: Whether the evolutive process includes batch normalization in the networks or not
        :param no_drop: Whether the evolutive process includes dropout in the networks or not
        :param add_obj: The number of objectives to be optimized apart from the network performance
        :return: --
        """
        deap_algs = [algorithms.eaSimple, algorithms.eaMuPlusLambda, algorithms.eaMuCommaLambda, algorithms.eaGenerateUpdate]

        creator.create("Fitness", base.Fitness, weights=[-1.0] * (len(self.test_outputs) + add_obj))

        creator.create("Individual", MyContainer, fitness=creator.Fitness)

        self.toolbox.register("evaluate", self.eval_individual)
        self.toolbox.register("generate", self.generate, creator.Individual)
        self.toolbox.register("update", self.update)

        if len(ev_kwargs) == 0:
            if ev_alg == 0:
                self.evol_kwargs = {"cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}
            if ev_alg == 1:
                self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, "cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}
            if ev_alg == 2:
                self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, "cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}
            if ev_alg == 3:
                self.evol_kwargs = {"ngen": self.generations, "verbose": 1}
        self.ev_alg = deap_algs[ev_alg]

    def evolve(self):
        """
        Actual evolution of individuals
        :return: The last generation, a log book (stats) and the hall of fame (the best individuals found)
        """

        hall_of = tools.HallOfFame(self.population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        result, log_book = self.ev_alg(self.toolbox, **self.evol_kwargs, stats=stats, halloffame=hall_of)

        return result, log_book, hall_of

    def init_individual(self, init_ind, no_batch, no_drop):
        """
        Creation of a single individual
        :param init_ind: DEAP function for transforming a VALP descriptor + evolvable hyperparameters into a DEAP individual
        :param no_batch: Boolean, whether networks can apply batch normalization or not
        :param no_drop: Boolean, whether networks can apply dropout or not
        :return: a DEAP individual
        """

        desc = MNMDescriptor(10, inp_dict, outp_dict)
        desc = recursive_creator(desc, 0, 0)
        hypers = {}
        if len(self.ev_hypers) > 0:
            for hyper in self.ev_hypers:
                hypers[hyper] = np.random.choice(self.ev_hypers[hyper])

        return init_ind([desc, hypers])

    def eval_individual(self, individual):
        """
        Function for evaluating an individual.
        :param individual: DEAP individual
        :return: Fitness value.
        """

        desc, hypers = individual.descriptor

        model = MNM(desc, hypers["btch_sz"], self.train_inputs, self.train_outputs, loss_func_weights={"o0": hypers["wo0"], "o1": hypers["wo1"], "o2": hypers["wo2"]})
        model.epoch_train(hypers["btch_sz"], 400, 0)

        a = model.predict(self.test_inputs, [], new=True)[0]
        m2e = np.mean(self.evaluation["o0"](a["o0"], self.test_outputs["o0"]))

        acc = np.mean(self.evaluation["o1"](a["o1"][:, 0], np.argmax(self.test_outputs["o1"], axis=1)))
        i_d = -np.mean(self.evaluation["o2"](a["o2"]))
        tf.reset_default_graph()
        del model

        return acc, m2e, i_d

    def update(self, population):

        self.accs = []
        self.mses = []
        self.ids = []

        for individual in population:
            fit = individual.fitness.values
            self.accs += [fit[0]]
            self.mses += [fit[1]]
            self.ids += [fit[2]]

        self.acc_args = np.argsort(self.accs)
        self.mse_args = np.argsort(self.mses)
        self.id_args = np.argsort(self.ids)

        self.population = population

    def generate(self, init_ind):

        if self.population is not None:

            new_pop = tools.selNSGA2(self.population, self.population_size)

            for i in range(len(new_pop)):
                decision = np.random.rand()
                if decision < self.cXp:
                    c_output = np.argmax([self.acc_args, self.mse_args, self.id_args])
                    #print(c_output)
                elif decision < self.cXp + self.mtp:
                    new_pop[i] = init_ind(mutation(new_pop[i], np.argmin([self.acc_args, self.mse_args, self.id_args]), self.ev_hypers, self.max_net, self.max_con, self.max_lay, self.no_drop, self.no_batch))
        else:
            new_pop = [self.init_individual(init_ind, self.no_batch, self.no_drop) for _ in range(self.population_size)]

        return new_pop


def mutation(individual, safe, ev_hypers, max_net, max_con, max_lay, drop=False, norm=False):
    desc, hypers = individual.descriptor
    assert isinstance(desc, MNMDescriptor)
    decision = np.random.rand()
    if decision < 1/3:
        hypers = hyper_mutation(hypers, ev_hypers)
    elif decision < 2/3:
        structure_mutation(max_net, max_con, desc, safe)
    else:
        net = np.random.choice(list(desc.networks.keys()))
        desc.networks[net].descriptor = network_mutation(max_lay, drop, norm, desc.networks[net].descriptor)

    fix_in_out_sizes(desc)

    return desc, hypers


def hyper_mutation(hypers, ev_hypers):
    h = np.random.choice(list(ev_hypers.keys()))  # We select the hyperparameter to be mutated
    # We choose two values, just in case the first one is the one already selected
    new_value = np.random.choice(ev_hypers[h], size=2, replace=False)
    if hypers[h] == new_value[0]:
        hypers[h] = new_value[1]
    else:
        hypers[h] = new_value[0]

    return hypers


def structure_mutation(max_net, max_con, valp, safe, mutations=["del_conn", "del_comp", "add_conn", "add_comp"]):

    result = -1
    if len(mutations) == 0:
        return result

    assert isinstance(valp, MNMDescriptor)
    if len(valp.connections) >= max_con and ["add_conn"] in mutations:
        mutations = mutations.remove("add_conn")
    if len(valp.networks) >= max_net and ["add_comp"] in mutations:
        mutations += mutations.remove("add_comp")

    mut = np.random.choice(mutations)

    if mut == "del_conn":
        result = del_con(valp, safe)
        if result == -1:
            mutations.remove("del_conn")
            result = structure_mutation(max_net, max_con, valp, safe, mutations)

    if mut == "del_comp":
        result = bypass(valp, safe)
        if result == -1:
            mutations.remove("del_comp")
            result = structure_mutation(max_net, max_con, valp, safe, mutations)

    if mut == "add_comp":
        result = divide_con(valp, safe)
        if result == -1:
            mutations.remove("add_comp")
            result = structure_mutation(max_net, max_con, valp, safe, mutations)

    if mut == "add_conn":
        result = add_con(valp, safe)
        if result == -1:
            mutations.remove("add_conn")
            result = structure_mutation(max_net, max_con, valp, safe, mutations)

    return result


def network_mutation(max_lay, batch, drop, network):
    """
    Mutation operators for individuals. They can affect any network or the hyperparameters.

    :param max_lay: Layer limit for a network
    :param batch: Whether batch normalization can be added to a network
    :param drop: Whether dropoff can be added to a network
    :param network: DEAP individual. Contains a dict where the keys are the components of the model
    :return: Mutated version of the DEAP individual.
    """

    mutation_types = ["add_layer", "del_layer", "weight_init", "activation", "dimension"]

    if not batch:
        mutation_types = ["batch_norm"] + mutation_types
    if not drop:
        mutation_types = ["dropout"] + mutation_types

    type_mutation = np.random.choice(mutation_types)

    if type_mutation == "add_layer":  # We add one layer
        layer_pos = np.random.randint(network.number_hidden_layers) + 1
        lay_dims = np.random.randint(max_lay) + 1
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

    elif type_mutation == "del_layer":  # We remove one layer
        network.remove_random_layer()

    elif type_mutation == "weight_init":  # We change weight initialization function in all layers
        layer_pos = np.random.randint(network.number_hidden_layers)
        init_w_function = initializations[np.random.randint(len(initializations))]
        network.change_weight_init(layer_pos, init_w_function)

    elif type_mutation == "activation":  # We change the activation function in layer
        layer_pos = np.random.randint(network.number_hidden_layers)
        init_a_function = activations[np.random.randint(len(activations))]
        network.change_activation(layer_pos, init_a_function)

    elif type_mutation == "dimension":  # We change the number of neurons in layer
        network.change_layer_dimension(max_lay)

    elif type_mutation == "dropout":
        network.change_dropout()

    elif type_mutation == "batch_norm":
        network.change_batch_norm()

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

    return network


def cross(init_ind, ind1, ind2):
    """
    Crossover operator for individuals. Cannot be applied in the simple case, as it randomly interchanges model components
    :param init_ind: DEAP function for initializing dicts (in this case) as DEAP individuals
    :param ind1: 1st individual to be crossed
    :param ind2: 2st individual to be crossed
    :return: Two new DEAP individuals, the crossed versions of the incoming parameters.
    """

    return init_ind, ind1, ind2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(10001),
                        nargs='+', help='an integer in the range 0..3000')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum,
                        default=max, help='sum the integers (default: find the max)')

    args = parser.parse_args()
    seed = args.integers[0]
    max_comp = args.integers[1]
    pop = args.integers[2]
    generations = args.integers[3]
    epochs = args.integers[4]

    loaded_model = load_model()

    loss_weights, (data_inputs, inp_dict), (data_outputs, outp_dict), (x_train, c_train, y_train, x_test, c_test, y_test) = diol()
    # The GAN evolutive process is a common 2-DNN evolution
    hyps = {"btch_sz": [100, 150, 200], "wo0": [0.1, 0.5, 1], "wo1": [0.1, 0.5, 1], "wo2": [0.1, 0.5, 1]}

    e = EvoVALP(x_trains=data_inputs["Train"], y_trains=data_outputs["Train"], x_tests=data_inputs["Test"], y_tests=data_outputs["Test"], population=pop, generations=generations, hyperparameters=hyps)
    res = e.evolve()

    print(res[0])