from VALP.Networks import Decoder, Generic, Discrete, CNN, ConvDecoder, batch
from VALP.ModelDescriptor import recursive_creator
from VALP.descriptor import MNMDescriptor
from VALP.classes import ModelComponent, InOut
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from functools import reduce
import time
from datetime import timedelta

network_types = {"Decoder": Decoder, "Generic": Generic, "Discrete": Discrete, "Convolution": CNN, "ConvDecoder": ConvDecoder}

opts = [tf.train.AdamOptimizer, tf.train.AdagradOptimizer, tf.train.AdadeltaOptimizer, tf.train.GradientDescentOptimizer, tf.train.RMSPropOptimizer]


class MNM(object):
    """
    Tensorflow level implementation of the VALP
    """
    def __init__(self, descriptor, batch_size, inputs, outputs, loss_func_weights, name="", load=None, init=True, lr=0.01, opt=0, random_seed=None):

        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_random_seed(random_seed)
            random.seed(random_seed)

        self.name = name

        self.descriptor = descriptor
        if load is not None:
            if isinstance(load, str):
                self.descriptor.load(load + "model_" + name + ".txt")
            else:
                self.descriptor.load("model_" + name + ".txt")
        elif not descriptor.constructed:
            self.descriptor = recursive_creator(descriptor, 0, 0)  # Create a new descriptor if it is empty and not loaded

        self.inputs = {}  # ID: placeholder
        self.outputs = {}  # ID: placeholder
        self.components = {}  # ID: Component
        self.predictions = {}  # ID: Data (where the result is placed, what has to be sess.run-ed)
        self.input_data = {}  # ID: Data (numpy data, from which learning is done)
        self.output_data = {}  # ID: DAta (numpy data, from which learning is done)
        self.lr = lr
        self.opt = opt

        self.initialized = []  # List of initialized components (to know what to build, when recursively creating tf DNNs)
        self.sess = tf.Session()  # config=tf.ConfigProto(device_count={'GPU': 0})
        self.optimizer = None  # To be sess.run-ed to train all the objectives of the VALP
        self.optimizer_samp = None  # To be sess.run-ed to train only the sampling output (VAE does multiple training for each data piece)
        self.loss_function = 0  # General loss function
        self.loss_function_sample = 0  # loss function containing only the sampling and KL losses
        self.batch_size = batch_size
        self.example_num = inputs[random.choice(list(inputs.keys()))].shape[0]
        self.loss_weights = {}  # Beta parameter. Implemented as tf variables, in case we want to dinamically modify it
        for i in loss_func_weights:
            self.loss_weights[i] = tf.Variable(loss_func_weights[i], dtype=np.float, trainable=False)

        self.sub_losses = {}  # All the loss functions separated (mainly for debugging)

        for model_input in descriptor.inputs:
            self.add_input(inputs[model_input], model_input)

        for outp in descriptor.outputs:
            self.add_output(outputs[outp], outp)

        if init:  # If the tf variables have to be initialized (most of the times)
            if load is not None:  # If the weights have to be loaded
                if "str" in type(load).__name__:  # specific path
                    self.load(load)
                elif load:  # or default
                    self.load("/")
            else:
                self.initialize(load)  # Random initialization

    def add_input(self, data, inp_id):

        self.inputs[inp_id] = tf.placeholder(tf.float32, [None] + list(data.shape[1:]), name=inp_id)
        self.input_data[inp_id] = data

    def add_output(self, data, outp_id):

        self.outputs[outp_id] = tf.placeholder(tf.float32, [None] + list(data.shape[1:]), name=outp_id)
        self.output_data[outp_id] = data

    def add_component(self, comp, comp_id=None):
        if not comp_id:
            comp_id = str(len(self.components.keys()))
        self.components[comp_id] = comp

    def component_output_by_id(self, ident):

        if ident in self.components:
            return self.components[ident].result
        elif ident in self.inputs:
            return self.inputs[ident]
        elif ident in self.outputs:
            return self.outputs[ident]
        else:
            return None

    def initialize(self, load, load_path="", vars=None):
        """
        This function calls recursive_init, which either initializes or loads the tf weights, and constructs the different loss functions
        :param load: Whether the weights have to be loaded or can be randomly initialized
        :param load_path: Path from which info has to be loaded
        :return: --
        """
        aux_pred = {}

        self.recursive_init(self.descriptor.comp_by_input(self.descriptor.outputs), aux_pred, load, load_path)  # Weight initialization

        self.loss_function = 0
        self.loss_function_sample = 0

        # ### Loss function initialization
        for pred in self.predictions.keys():
            if self.descriptor.outputs[pred].taking.type == "discrete":  # If this fails, it is being initialized twice
                self.sub_losses[pred] = self.loss_weights[pred] * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions[pred], labels=self.outputs[pred]))
                self.loss_function += self.sub_losses[pred]
                self.predictions[pred] = tf.reshape(tf.argmax(self.predictions[pred], axis=1), (-1, 1))  # tf.sigmoid(self.predictions[pred])
            elif self.descriptor.outputs[pred].taking.type == "values":
                self.sub_losses[pred] = self.loss_weights[pred] * tf.reduce_mean(tf.pow(self.predictions[pred] - self.outputs[pred], 2) / reduce(lambda x, y: x*y, self.outputs[pred].shape[1:]).value)  # tf.losses.mean_squared_error(self.predictions[pred], self.outputs[pred])
                self.loss_function += self.sub_losses[pred]
            elif self.descriptor.outputs[pred].taking.type == "samples":
                # print(self.predictions[pred], tf.reshape(self.outputs[pred], (-1, reduce(lambda x, y: x*y, self.outputs[pred].shape[1:]))))
                self.sub_losses[pred] = self.loss_weights[pred] * tf.reduce_mean(tf.pow(tf.layers.flatten(self.predictions[pred]) - tf.layers.flatten(self.outputs[pred]), 2) / reduce(lambda x, y: x*y, self.outputs[pred].shape[1:]).value)
                self.loss_function_sample += self.sub_losses[pred]
                self.loss_function += self.sub_losses[pred]

        for network in self.descriptor.networks:
            if "Decoder" in type(self.descriptor.comp_by_ind(network).descriptor).__name__:
                self.sub_losses[network] = -0.00001 * tf.reduce_mean(1 + self.components[network].z_log_sigma_sq - tf.square(self.components[network].z_mean) - tf.exp(self.components[network].z_log_sigma_sq))
                self.loss_function += self.sub_losses[network]
                self.loss_function_sample += self.sub_losses[network]
        if vars is not None:
            vars = [self.components[net].List_weights + self.components[net].List_bias for net in vars]
            vars = [item for sublist in vars for item in sublist]
        self.optimizer = opts[self.opt](learning_rate=self.lr).minimize(self.loss_function, var_list=vars)
        try:
            self.optimizer_samp = opts[self.opt](learning_rate=self.lr).minimize(self.loss_function_sample, var_list=vars)
        except:
            self.optimizer_samp = None
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def epoch_train(self, batch_size, epochs, sync, display_step=1):
        """
        Train by epoch limit.
        :param batch_size: Self explanatory
        :param epochs: Epoch limit
        :param sync: Times the sampling loss function is trained each epoch (VAEs are trained several times with the same input)
        :param display_step: verbose
        :return: --
        """
        aux_ind = 0
        partial_loss = 0
        for epoch in range(epochs):
            feed_dict = {}

            for inp in self.inputs:
                feed_dict[self.inputs[inp]] = batch(self.input_data[inp], batch_size, aux_ind)
            for output in self.outputs:
                feed_dict[self.outputs[output]] = batch(self.output_data[output], batch_size, aux_ind)

            aux_ind = (aux_ind + batch_size) % self.example_num
            for i in self.components:
                if not np.isfinite(self.sess.run(self.components[i].result, feed_dict=feed_dict)).all():
                    print(epoch, i)
            _, partial_loss = self.sess.run([self.optimizer, self.loss_function], feed_dict=feed_dict)
            if epoch % display_step == 1:
                print(epoch, partial_loss)
        return partial_loss

    def convergence_train(self, batch_size, conv, conv_param, proportion=0.7, epoch_lim=1000, sync=0, display_step=-1):
        """
        Train by convergence. When the loss is was smaller 'n' iterations ago than currently, training is stopped. Also has a epoch limit
        :param batch_size: Self explanatory
        :param conv: Epochs to take into account when deciding whether training has converged or not, 'n'
        :param conv_param: flexibility multiplier to allow larger or shorter convergence criterion
        :param proportion: proportion to divide train and test sets
        :param epoch_lim: Training also has to be limited by an epoch limit
        :param sync: Times the sampling loss function is trained each epoch (VAEs are trained several times with the same input)
        :param display_step: verbose
        :return:
        """
        last_res = np.zeros(conv)
        epoch = 0
        aux_ind = 0
        test_dict = {}
        train_dict = {}
        for inp in self.inputs:

            test_dict[self.inputs[inp]] = self.input_data[inp][int(self.input_data[inp].shape[0]*proportion):]
            train_dict[self.inputs[inp]] = self.input_data[inp][:int(self.input_data[inp].shape[0]*proportion)]
        for output in self.outputs:
            test_dict[self.outputs[output]] = self.output_data[output][int(self.output_data[output].shape[0]*proportion):]
            train_dict[self.outputs[output]] = self.output_data[output][:int(self.output_data[output].shape[0]*proportion)]

        while True:
            feed_dict = {}

            for inp in self.inputs:
                feed_dict[self.inputs[inp]] = batch(train_dict[self.inputs[inp]], batch_size, aux_ind)

            for output in self.outputs:
                feed_dict[self.outputs[output]] = batch(train_dict[self.outputs[output]], batch_size, aux_ind)

            aux_ind = (aux_ind + batch_size) % self.example_num
            for _ in range(sync):
                _, partial_loss = self.sess.run([self.optimizer_samp, self.loss_function_sample], feed_dict=feed_dict)  # Sampling training

            _, partial_loss = self.sess.run([self.optimizer, self.loss_function], feed_dict=feed_dict)  # Normal training

            last_res[1:] = last_res[:-1]
            last_res[0], p1 = self.sess.run([self.loss_function, self.predictions["o1"]], test_dict)

            if epoch % display_step == 0 or True:
                print(epoch, last_res[0])

            epoch += 1
            if (epoch > epoch_lim) or (last_res[-1] < last_res[0]*conv_param and last_res[last_res > 0].shape[0] == conv):
                # If we have more than 'n' epochs, and 'n' epochs ago we had an smaller error, convergence has been reached.
                break
        return last_res[0]

    def predict(self, inputs, intra_preds=(), new=True):
        """
        Given an input, this function returns the VALPs prediction
        :param inputs: Dictionary ID: data (numpy)
        :param intra_preds: List of IDs whose output has to be returned as well. Use to observe the output of a certain network
        :param new: Whether to delete connections from the Decoder or not
        :return:
        """

        feed_dict = {}
        intra = []
        examples = 1
        for intra_pred in intra_preds:
            intra += [self.components[intra_pred].z_log_sigma_sq, self.components[intra_pred].z_mean]

        for inp in inputs.keys():
            feed_dict[self.inputs[inp]] = inputs[inp]
            if isinstance(inputs[inp], dict):
                examples = inputs[inp]["i0"].shape[0]
            else:
                examples = inputs[inp].shape[0]
        if new:
            for net in self.components:
                if "Decoder" in type(self.components[net]).__name__:

                    feed_dict[self.components[net].z_log_sigma_sq] = np.reshape(np.random.normal(0, 1, examples*self.components[net].z_log_sigma_sq.shape[1].value), (examples, self.components[net].z_log_sigma_sq.shape[1].value))
                    feed_dict[self.components[net].z_mean] = np.reshape(np.random.normal(0, 1, examples*self.components[net].z_log_sigma_sq.shape[1].value), (examples, self.components[net].z_log_sigma_sq.shape[1].value))

        return self.sess.run([self.predictions] + intra, feed_dict=feed_dict)

    def recursive_init(self, comps, aux_pred, load, load_path=""):
        """
        This function initializes the tensorflow graph. It initializes the components by levels. They graph initialization is
        performed as a depth-first algorithm. The principal call to the function requires the initialization of the components
        directly connected to the outputs. The program explores the graph in a depth-first way, so that all the components
        on which the outputs depend (all of them) are initialized

        :param comps: Components on which the initialization starts
        :param aux_pred: Unused
        :param load: whether to load network weights or not
        :param load_path: path from which the info is loaded
        :return: The tf graph is initialized
        """

        for comp in comps:  # For each component

            if comp not in self.initialized and "i" not in comp:  # If it is not already in the tf model, or is an input
                self.initialized += [comp]  # Save as initialized
                comps_below = self.descriptor.comp_by_input(comp)  # Get all the components on which comp depends
                self.recursive_init(comps_below, aux_pred, load, load_path)  # Initialize them

                net = self.descriptor.comp_by_ind(comp)  # Get the tf structure of comp

                aux_input = []

                for comp_below in comps_below:  # For each component on which comp depends
                    aux_input += [self.component_output_by_id(comp_below)]
                self.components[comp] = network_types[type(net.descriptor).__name__[:-10]](net.descriptor, comp)  # Initialize the Network (as tf)
                self.components[comp].building(aux_input, load, load_path + self.name + "_" + comp + ".npy")  # Initialize the Network (as tf)

                outs = self.descriptor.comp_by_output(comp)  # Compile all the components that take the production of comp

                for out in outs:
                    if "o" in out:  # If the component is an output, add it to the predictions
                        if out not in self.predictions.keys():  # If it is the first one
                            # if isinstance(self.descriptor.outputs[out].taking.size, tuple):
                            # else:
                            self.predictions[out] = self.components[comp].result[:, :self.descriptor.outputs[out].taking.size if not isinstance(self.descriptor.outputs[out].taking.size, tuple) else reduce(lambda x, y: x*y, self.descriptor.outputs[out].taking.size)]
                        else:  # If not, add it
                            self.predictions[out] = self.predictions[out] + self.components[comp].result[:, :self.descriptor.outputs[out].taking.size if not isinstance(self.descriptor.outputs[out].taking.size, tuple) else reduce(lambda x, y: x*y, self.descriptor.outputs[out].taking.size)]

    def save(self, path="/home/unai/Escritorio/MultiNetwork/model"):
        saver = tf.train.Saver()

        saver.save(self.sess, path)
        self.descriptor.save(path + ".txt")

    def load(self, path="/home/unai/Escritorio/MultiNetwork"):

        self.descriptor.load(path + "model.txt")
        self.initialize(False)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def save_weights(self, path="/home/unai/Escritorio/MultiNetwork"):

        for cmp in self.components:
            self.components[cmp].save_weights(self.sess, path + str(self.name) + "_")

    def load_weights(self):

        self.initialize(True)


def histogram(x):
    # the histogram of the data
    plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    plt.grid(True)

    plt.show()


def reset_graph(random_seed):

    tf.reset_default_graph()
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == "__main__":

    fashion_mnist = tf.keras.datasets.mnist.load_data()

    x_train = np.expand_dims(fashion_mnist[0][0], axis=3)/256
    x_train = np.concatenate((x_train, x_train, x_train), axis=3)
    c_train = fashion_mnist[0][1]
    y_train = np.array([np.histogram(obs, bins=32)[0] for obs in x_train])/784

    x_test = np.expand_dims(fashion_mnist[1][0], axis=3)/256
    x_test = np.concatenate((x_test, x_test, x_test), axis=3)
    c_test = fashion_mnist[1][1]
    y_test = np.array([np.histogram(obs, bins=32)[0] for obs in x_test])/784

    # dataset = datasets.fetch_mldata('MNIST original')

    # data_y = np.reshape(np.sin(data.data[:, 1]) + data.data[:, 2] * data.data[:, 0] - np.cos(data.data[:, 3]), (-1, 1))  # Manual function
    # data_y = np.array([np.histogram(obs, bins=32)[0] for obs in dataset.data])/784
    # dataset.data = dataset.data/256

    # x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(dataset.data, dataset.target, data_y, random_state=1)

    model_inputs = {}

    output_placeholders = {}

    data_inputs = {}
    inp_dict = {}
    # Separated inputs
    """
    for i in range(iris.data.shape[1]):
        data_inputs["i" + str(i)] = np.reshape(iris.data[:, i], (-1, 1))
        inp_dict["i" + str(i)] = ModelComponent(None, InOut(size=1, type="values"))
    """

    # Merged inputs

    data_inputs["i0"] = x_train
    inp_dict["i0"] = ModelComponent(None, InOut(size=reduce(lambda x, y: x*y, x_train.shape[1:]), data_type="features"), -1)

    OHEnc = OneHotEncoder(categories='auto')

    a = OHEnc.fit_transform(np.reshape(c_train, (-1, 1))).toarray()

    data_outputs = {"o0": y_train}

    outp_dict = {"o0": ModelComponent(InOut(size=y_train.shape[1], data_type="values"), None, 0)}

    # Separated one hot encoding
    """
    for i in range(a.shape[1]):
        data_outputs["o" + str(i+1)] = np.reshape(a[:, i], [-1, 1])
        outp_dict["o" + str(i+1)] = ModelComponent(InOut(size=1, type="values"), None)
    """
    # Merged one hot encoding
    data_outputs["o1"] = a
    outp_dict["o1"] = ModelComponent(InOut(size=a.shape[1], data_type="discrete"), None, 0)

    # Samples

    data_outputs["o2"] = x_train
    outp_dict["o2"] = ModelComponent(InOut(size=x_train.shape[1:], data_type="samples"), None, 0)

    btch_sz = 50
    loss_weights = {"o0": 1, "o1": 1, "o2": 1}

    accs = []
    mses = []
    images = []
    conds = []
    total = 0

    for seed in range(0, 500):

        reset_graph(seed)

        model_descriptor = MNMDescriptor(10, inp_dict, outp_dict)

        model_descriptor = recursive_creator(model_descriptor, 0, conv_prob=0)
        model_descriptor.print_model_graph()
        model = MNM(model_descriptor, btch_sz, data_inputs, data_outputs, loss_weights)

        start = time.time()
        print("Seed:", str(seed), "Started at", time.asctime(time.localtime(start)))

        loss = model.epoch_train(btch_sz, 40000, 0)
        # model.save()
        # a, = model.predict({"i0": x_test}, [], new=False)
        a, = model.predict({"i0": x_test}, [], new=True)

        last_run = time.time() - start
        total += last_run

        print("Ended at", time.asctime(time.localtime(time.time())), "Total time spent", timedelta(seconds=last_run))

        print("Total time spent for", seed+1, "runs:", timedelta(seconds=total), "\n")

        mses += [mean_squared_error(a["o0"], y_test)]
        accs += [accuracy_score(a["o1"], c_test)]

        choice1, choice2 = np.random.randint(a["o2"].shape[0], size=2)

        images += [a["o2"][choice1], a["o2"][choice2]]
        conds += [c_test[choice1], c_test[choice2]]

        print(mses)
        print(accs)

        if seed % 10 == 0:
            np.save("accuracies" + str(seed) + ".npy", accs)
            np.save("MSEs" + str(seed) + ".npy", mses)
            np.save("images" + str(seed) + ".npy", images)
            np.save("conds" + str(seed) + ".npy", conds)
            accs = []
            mses = []
            images = []
            conds = []
        model.sess.close()
