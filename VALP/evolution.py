import numpy as np
import argparse
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
import os
from VALP.Model_Building import MNM
from VALP.classes import ModelComponent, InOut
from VALP.ModelDescriptor import recursive_creator
from VALP.descriptor import MNMDescriptor


def add_subpath(agent, desc):  # Not used right now
    inps = [0]
    con = None

    while len(inps) <= 1:
        con = np.random.choice(agent)
        inps = desc.comp_by_input(con.output)
        for inp in inps:
            pass
    inps.remove(con.input)
    inp = np.random.choice(inps)
    con = desc.get_connection(inp, con.output)
    agent += [con]
    get_agent(desc, inp, agent)


def eval_valp(individual):
    """
    Given a descriptor, this function creates a VALP and evaluates it. Used just for development
    :param individual: VALP descriptor
    :return: --
    """
    model = MNM(individual.model_descriptor, batch_size, data_inputs, data_outputs, loss_weights)
    loss = model.epoch_train(batch_size, 40000, 5)
    a, = model.predict({"i0": x_test}, [], new=True)


def data():
    """
    Load Fashion MNIST
    :return: Train and test data for classification and regression problems
    """
    if not os.path.isfile("f_mnist.npy"):
        import tensorflow as tf
        fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
        np.save("f_mnist.npy", fashion_mnist)
    else:
        fashion_mnist = np.load("f_mnist.npy", allow_pickle=True)
    x_tr = np.expand_dims(fashion_mnist[0][0], axis=3)/256
    x_tr = np.concatenate((x_tr, x_tr, x_tr), axis=3)
    c_tr = fashion_mnist[0][1]
    y_tr = np.array([np.histogram(obs, bins=32)[0] for obs in x_tr])/784

    x_te = np.expand_dims(fashion_mnist[1][0], axis=3)/256
    x_te = np.concatenate((x_te, x_te, x_te), axis=3)
    c_te = fashion_mnist[1][1]
    y_te = np.array([np.histogram(obs, bins=32)[0] for obs in x_te])/784

    return x_tr, c_tr, y_tr, x_te, c_te, y_te


def diol():
    """
    Loads Fashion MNIST in dict format, as a VALP uses it
    :return: Fashion MNIST in dict format
    """
    x_tr, c_tr, y_tr, x_te, c_te, y_te = data()
    oh_enc = OneHotEncoder(categories='auto')
    c_tr = oh_enc.fit_transform(np.reshape(c_tr, (-1, 1))).toarray()
    c_te = oh_enc.fit_transform(np.reshape(c_te, (-1, 1))).toarray()

    lw = {"o0": 1, "o1": 1, "o2": 1}

    di = {"Train": {"i0": x_tr}, "Test": {"i0": x_te}}
    id = {"i0": ModelComponent(None, InOut(size=reduce(lambda x, y: x*y, x_tr.shape[1:]), data_type="features"), -1)}

    od = {"o0": ModelComponent(InOut(size=y_tr.shape[1], data_type="values"), None, 0), "o1": ModelComponent(InOut(size=c_tr.shape[1], data_type="discrete"), None, 0), "o2": ModelComponent(InOut(size=reduce(lambda x, y: x * y, x_tr.shape[1:]), data_type="samples"), None, 0)}

    do = {"Train": {"o0": y_tr, "o1": c_tr, "o2": x_tr}, "Test": {"o0": y_te, "o1": c_te, "o2": x_te}}

    return lw, (di, id), (do, od), (x_tr, c_tr, y_tr, x_te, c_te, y_te)


def overkill_conn(desc):  # Not used
    available_nets = list(desc.networks.keys())  # This list contains the networks that can receive a connection (until proven otherwise)

    while len(available_nets) > 0:
        net = np.random.choice(available_nets)
        inp = desc.random_input(net)
        if inp == -1:
            available_nets.remove(net)
            continue
        else:
            desc.connect(inp, net)
            desc.reachable[inp] = desc.reachable[inp] + desc.reachable[net]
            for comp in list(desc.reachable.keys()):
                if inp in desc.reachable[comp]:
                    desc.reachable[comp] += desc.reachable[net]
    return desc


def get_agent(desc, cmp, cons):  # Not used

    for con in cons:
        if con.output == cmp or "i" in cmp:
            return

    inputs = desc.comp_by_input(cmp)
    n = len(inputs)
    ps = np.arange(n, 0, -1)/((n+1)*n)*2
    n = np.random.choice(np.arange(n, 0, -1), p=ps)
    m = np.random.randint(1, min(3, len(inputs)+1))
    inputs = np.random.choice(inputs, size=m, replace=False)
    for inp in inputs:
        cons += [desc.get_connection(inp, cmp)]
        get_agent(desc, inp, cons)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(3000),nargs='+', help='an integer in the range 0..3000')

    args = parser.parse_args()
    batch_size = 50

    loss_weights, (data_inputs, inp_dict), (data_outputs, outp_dict), (x_train, c_train, y_train, x_test, c_test, y_test) = diol()
    md = MNMDescriptor(30, inp_dict, outp_dict)
    descriptor = recursive_creator(md, 0)
    descriptor.save("basic.txt")
    descriptor.load("basic.txt")
    descriptor.print_model_graph("basic")

    #descriptor = overkill_conn(descriptor)
    #descriptor.save("overkill.txt")
    #descriptor.print_model_graph("overkill")
    #descriptor.load("overkill.txt")

    o2_agent = []
    o1_agent = []
    o0_agent = []
    get_agent(descriptor, "o2", o2_agent)
    get_agent(descriptor, "o1", o1_agent)
    get_agent(descriptor, "o0", o0_agent)

    descriptor.print_model_graph("Agents", (o0_agent, o1_agent, o2_agent))
    descriptor.print_model_graph("Plain")
