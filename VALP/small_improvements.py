import numpy as np
import argparse
import os

from VALP.classes import InOut, NetworkComp
from VALP.ModelDescriptor import recursive_creator, fix_in_out_sizes
from VALP.descriptor import MNMDescriptor, DecoderDescriptor, GenericDescriptor, DiscreteDescriptor
from VALP.evolution import diol
from VALP.Networks import convolutional_descriptor, decoder_descriptor, discrete_descriptor, generic_descriptor
from VALP.Model_Building import MNM
from copy import deepcopy
import tensorflow as tf
import random
from sklearn.metrics import accuracy_score, mean_squared_error
import datetime

from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import model_from_json
from scipy import misc
from tqdm import tqdm

# With convolutions

# types_go_to = {"features": ["Gen", "Conv", "Dis"], "values": ["Gen", "Dis", "Dec"], "samples": ["Gen", "Dis"], "discrete": ["Dis", "Gen", "Dec"]}
# types_come_from = {"features": ["Conv"], "values": ["Gen"], "discrete": ["Dis"], "samples": ["Gen", "Dec"]}
# nets_require = {GenericDescriptor: ["values", "discrete", "samples", "features"], DecoderDescriptor: ["values"], DiscreteDescriptor: ["values", "discrete", "samples", "features"], ConvolutionDescriptor: ["features"]}
# nets_produce = {GenericDescriptor: ["values", "samples"], DecoderDescriptor: ["samples"], DiscreteDescriptor: ["discrete"], ConvolutionDescriptor: ["features"]}

# Without convolutions

types_go_to = {"features": ["Gen", "Dis"], "values": ["Gen", "Dis", "Dec"], "samples": ["Gen", "Dis"], "discrete": ["Dis", "Gen", "Dec"]}
types_come_from = {"features": [], "values": ["Gen"], "discrete": ["Dis"], "samples": ["Gen", "Dec"]}
nets_require = {GenericDescriptor: ["values", "discrete", "samples", "features"], DecoderDescriptor: ["values"], DiscreteDescriptor: ["values", "discrete", "samples", "features"]}
nets_produce = {GenericDescriptor: ["values", "samples"], DecoderDescriptor: ["samples"], DiscreteDescriptor: ["discrete"]}

pareto = []

functions = {"Gen": generic_descriptor, "Dec": decoder_descriptor, "Dis": discrete_descriptor, "Conv": convolutional_descriptor}

loaded_model = None
three_objectives = np.array([999, 999, 999])


def reset_graph(seed):

    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def dividing_nets(desc, con):
    """
    Given a VALP descriptor and a connection, this function returns the set of networks that can be introduced in the middle of the connection
    :param desc: VALP descriptor
    :param con: Connection name
    :return: Set of networks that can be introduced in the middle of the connection (can be empty)
    """
    c = desc.pop_con(con)
    inp_comp = desc.comp_by_ind(c.input)
    out_comp = desc.comp_by_ind(c.output)

    # Compute the nets which can be placed in the middle of the connection
    # First according to the input of the connection
    if inp_comp.type == "Model":
        inp = [inp_comp.producing.type]
    else:
        inp = nets_produce[type(inp_comp.descriptor)]

    inp_nets = []
    for kind in inp:
        inp_nets += types_come_from[kind]

    # Then the output of the connection
    if out_comp.type == "Model":
        out = [out_comp.taking.type]
    else:
        out = nets_require[type(out_comp.descriptor)]
    out_nets = []
    for kind in out:
        out_nets += types_go_to[kind]

    nets = set(inp_nets).intersection(set(out_nets))  # Finally, we intersect both sets of possible networks. If a net is present in this final set, we exit the while loop because we have found the connection to be modified

    desc.connections[con] = c

    return nets, inp_comp, c


def divide_con(desc, safe="None", con=""):
    """
    Given a descriptor, this function takes a connection and places a network in the middle
    :param desc: VALP descriptor
    :param safe: An output of the VALP. If specified, this mutator won't affect any connection related to this output
    :param con: Name of the connection to be divided. In case it is not specified, it is randomly chosen respecting the "safe" parameter
    :return: The name of the recently introduced network (or -1 in case it fails)
    """

    nets = []  # This variable will contain the list of networks that fit into the selected connection
    trial = 50
    inp_comp = None  # This variable will contain the type of data produced by the component in the input of the selected connection
    c = None  # This variable will contain the actual connection (not the name) to be divided.

    if con == "":  # If the connection to be modified has not been specified, we have to find one which can be divided
        while len(nets) == 0 and trial > 0:  # While we haven't found a connection which can be divided and have trials left (it shouldn't be a problem)
            trial += 1

            con = np.random.choice([w for w in desc.connections.keys() if safe not in desc.reachable[desc.connections[w].output]])  # Choose a random connection which doesn't affect the safe output
            nets, inp_comp, c = dividing_nets(desc, con)

    else:  # If the connection has been specified, we simply compute the possible networks to be introduced into the connection (the code is similar)
        nets, inp_comp, c = dividing_nets(desc, con)

    if len(nets) == 0:  # If we haven't found a possibility, return with error code
        return -1

    # Select a network type from all the possibilities, and its different characteristics

    net = np.random.choice(list(nets))
    aux = np.random.randint(5, 50)
    d = functions[net](0, aux)
    d.n_inputs = 1

    if "Dec" in net:
        prod = "samples"
    elif "Dis" in net:
        prod = "discrete"
    elif "Conv" in net:
        prod = "features"
    elif "samples" in inp_comp.producing.type:
        prod = "samples"
    else:
        prod = "values"

    new = NetworkComp(d, InOut(data_type=inp_comp.producing.type, size=inp_comp.producing.size), InOut(data_type=prod, size=aux), depth=288)
    new = desc.add_net(new)

    if "Dec" in net:
        desc.networks[new].descriptor.rands += [0]

    # Delete the previous connection and add the new ones (from the input to the new net, and from the new net to the output)

    desc.connect(c.input, new, c.name)
    desc.connect(new, c.output)

    desc.reachable[new] = [new] + desc.reachable[c.output]
    desc.reachable[c.input] += [new]

    for node in list(desc.networks.keys()) + list(desc.inputs.keys()):  # Update the reachable attribute from the previous networks.
        if c.input in desc.reachable[node]:
            desc.reachable[node] += [new]

    return new


def add_con(desc, safe, inp=None, out=None):
    """
    Given a descriptor, this function adds a connection.
    :param desc: VALP descriptor
    :param safe: An output of the VALP. If specified, this mutator won't affect any connection related to this output
    :param inp: A VALP component. If specified, would be the input of the connection
    :param out: A VALP component. If specified, would be the output of the connection
    :return: The name of the recently added connection (or -1 in case it fails)
    """

    able = False
    trial = 50

    while not able and trial > 0:  # Similarly to the previous function, we have a limit of trials, but it shouldn't be a problem

        if inp is None:  # If the input is not predefined
            if out is None:  # If both are undefined, we randomly select an output for the connection, and search for compatible inputs
                out = np.random.choice([w for w in (list(desc.networks.keys()) + list(desc.outputs.keys())) if safe not in desc.reachable[w]])
            if "Network" in type(desc.comp_by_ind(out)).__name__:
                pass  # print(type(desc.comp_by_ind(out).descriptor).__name__, "onv" in type(desc.comp_by_ind(out).descriptor).__name__, [x for x in desc.networks.keys() if "onv" in type(desc.comp_by_ind(x).descriptor).__name__])
            if "Network" in type(desc.comp_by_ind(out)).__name__ and "onv" in type(desc.comp_by_ind(out).descriptor).__name__:
                pos = list(set([x for x in desc.networks.keys() if "onv" in type(desc.comp_by_ind(x).descriptor).__name__] + list(desc.inputs.keys())) - set(desc.reachable[out]))
            elif "o" in out:
                pos = list(set([x for x in desc.networks.keys() if "onv" not in type(desc.comp_by_ind(x).descriptor).__name__]) - set(desc.reachable[out]))
            else:
                pos = list(set(list(desc.networks.keys()) + list(desc.inputs.keys())) - set(desc.reachable[out]))
            if len(pos) > 0:
                able = True
                inp = np.random.choice(pos)
            else:
                out = None

        else:
            if out is None:  # If the input is defined but not the output, we search for compatible outputs
                options = [x for x in list(desc.networks.keys()) + list(desc.outputs.keys()) if inp not in desc.reachable[x] and safe not in desc.reachable[x]]
                if len(options) > 0:
                    able = True
                    out = np.random.choice(options)

            elif inp in desc.reachable[out]:
                raise Exception("U cant make recursive connections yet")

    if not able:  # If there are no compatibilities, return error
        return -1

    # Perform the connection
    c = desc.connect(inp, out)
    desc.reachable[inp] += desc.reachable[out]

    for node in list(desc.networks.keys()) + list(desc.inputs.keys()):  # Update reachables
        if inp in desc.reachable[node]:
            desc.reachable[node] += desc.reachable[out]
    # Update the rest of requirements
    if "n" in desc.connections[c].output:
        desc.networks[desc.connections[c].output].descriptor.n_inputs += 1
        if "ecoder" in type(desc.networks[desc.connections[c].output].descriptor).__name__:
            if np.random.rand() > 0.3:
                desc.networks[desc.connections[c].output].descriptor.rands += [desc.networks[desc.connections[c].output].descriptor.n_inputs]
            else:
                desc.networks[desc.connections[c].output].descriptor.conds += [desc.networks[desc.connections[c].output].descriptor.n_inputs]
    return c


def del_con(desc, safe, con=""):
    """
    Given a descriptor, this function deletes a connection.
    :param desc: VALP descriptor
    :param safe: An output of the VALP. If specified, this mutator won't affect any connection related to this output
    :param con: Name of the connection to be deleted. If unspecified, a random viable one will be chosen
    :return: Name of the deleted connection (or -1 if failed)
    """

    if con == "":  # If the connection is not predefined, choose a deletable one
        keys = list(desc.connections.keys())
        np.random.shuffle(keys)
        con = keys.pop()
        while len(keys) > 0 and not (is_deletable(desc, con) and safe not in desc.reachable[desc.connections[con].output]):
            con = keys.pop()

    if not is_deletable(desc, con):  # If we haven't found a deletable connection
        return -1

    if "n" in desc.connections[con].output:  # Perform the necessary changes to the descriptor of the network in the input of the connection
        desc.networks[desc.connections[con].output].descriptor.n_inputs -= 1
        if "ecoder" in type(desc.networks[desc.connections[con].output].descriptor).__name__:
            if (np.random.rand() < 0.5 and len(desc.networks[desc.connections[con].output].descriptor.rands) > 1) or len(desc.networks[desc.connections[con].output].descriptor.conds) < 1:
                desc.networks[desc.connections[con].output].descriptor.rands.pop()
            else:
                desc.networks[desc.connections[con].output].descriptor.conds.pop()

    # Update the reachable attributes of the VALP
    to_be_deleted = desc.reachable[desc.connections[con].output]
    to_be_updated = []

    for net in desc.networks:  # Identify the affected components
        if desc.connections[con].input in desc.reachable[net]:
            desc.reachable[net] = [w for w in desc.reachable[net] if w not in to_be_deleted] + [net]
            to_be_updated += [net]

    for net in desc.networks:  # Apply the necessary changes to the identified components
        if net in to_be_updated:
            for goal in to_be_deleted:
                if desc.conn_exists(net, goal) or len([w for w in desc.nodes() if w in desc.reachable[net] and goal in desc.reachable[w]]) > 0:
                    aux = desc.reachable[goal]
                    desc.reachable[net] = desc.reachable[net] + aux

    del desc.connections[con]
    return con


def is_deletable(desc, con):
    """
    Check if con is deleteble from desc
    :param desc: VALP descriptor
    :param con: Connection name
    :return: Boolean, whether a connection can be deleted while keeping the VALP working properly
    """
    ins_of_out = []
    outs_of_in = []
    for conn in desc.connections:  # We need to check whether the functionality of the selected connection is redundant or not...
        if conn not in con:
            if desc.connections[conn].input == desc.connections[con].input:  # ...in the component in the input of the connection
                outs_of_in += [conn]
            if desc.connections[conn].output == desc.connections[con].output:  # ...in the component in the output of the connection
                ins_of_out += [conn]

    return (len(ins_of_out) > 0) and (len(outs_of_in) > 0)  # If we have found connections which cover the effect produced by deleting the connection both in the input and output components, its deletable


def bypass(desc, safe, net=""):
    """
    Delete one network from a VALP
    :param desc: VALP descriptor
    :param safe: An output of the VALP. If specified, this mutator won't affect any component related to this output
    :param net: Network to be deleted. If unspecified, the function will search for one
    :return: the name of the deleted network (or -1 if failed to find a bypassable one)
    """
    if net == "":  # If the network to be deleted has not been predefined, search for a deletable one
        keys = list(desc.networks.keys())
        np.random.shuffle(keys)
        net = keys.pop()
        while len(keys) > 0 and not (is_bypassable(desc, net) and safe not in desc.reachable[net]):
            net = keys.pop()

    if not is_bypassable(desc, net):  # If we haven't found deletable networks, error
        return -1

    # Delete all connections related to the network to be bypassed, and connect the other ends of the networks together
    inps, in_conns, out_conns, outs = desc.get_net_context(net)
    cons = in_conns + out_conns  # Checka si esto da error
    for inp in inps:
        for out in outs:
            if len(cons) > 0:
                con_name = cons.pop()
            else:
                con_name = ""
            desc.connect(inp, out, con_name)

    while len(cons) > 0:
        del desc.connections[cons.pop()]

    # Delete network and update reachables
    del desc.networks[net]

    for network in desc.networks:
        desc.reachable[network] = [w for w in desc.reachable[network] if not w == net]
    for inp in desc.inputs:
        desc.reachable[inp] = [w for w in desc.reachable[inp] if not w == net]

    del desc.reachable[net]

    return net


def is_bypassable(desc, net):
    """
    Given a descriptor and a network, decide whether the VALP can continue working without it
    :param desc: VALP descriptor
    :param net: Network name
    :return: Whether the network can be deleted or not
    """
    if desc.comp_by_ind(net).taking.type == desc.comp_by_ind(net).producing.type:  # If the network takes and produces the same data type
        return True
    if "alues" in desc.comp_by_ind(net).taking.type and "iscre" in desc.comp_by_ind(net).producing.type:
        return True
    if "iscre" in desc.comp_by_ind(net).taking.type and "alues" in desc.comp_by_ind(net).producing.type:
        return True
    if "amples" in desc.comp_by_ind(net).taking.type and "alues" in desc.comp_by_ind(net).producing.type:
        return True
    return False


def keras_model(xtr, ytr, xts, yts):
    """
    This function trains a DNN classification model and stores it. The model is used to compute the Inception score.
    :param xtr: Train x
    :param ytr: Train y
    :param xts: Test x
    :param yts: Test y
    :return:
    """

    np.random.seed(0)
    tf.set_random_seed(0)

    height, width = 90, 90
    input_image = Input(shape=(height, width, 3))

    # base_model = InceptionV3(input_tensor=input_image, include_top=False, pooling='avg')  # MobileNet model provides better accuracy in the test set
    base_model = MobileNet(input_tensor=input_image, include_top=False, pooling='avg')
    output = Dropout(0.5)(base_model.output)
    predict = Dense(10, activation='softmax')(output)

    model = Model(inputs=input_image, outputs=predict)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    x_tr = np.array([misc.imresize(x, (height, width)).astype(float) for x in tqdm(iter(xtr))])/255.

    x_tst = np.array([misc.imresize(x, (height, width)).astype(float) for x in tqdm(iter(xts))])/255.

    model.fit(x_tr, ytr, batch_size=64, epochs=50, validation_data=(x_tst, yts))
    preds = model.predict(x_tst)

    np.save("MobilePredictions.npy", preds)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model


def load_model(model_name="Mobile"):
    """
    This function loads a Fashion MNIST classifier. Used for evaluating the sampling capabilities of a model.
    :param model_name: Which model to load ("Mobile" or "Inception"). The original distance uses inception. Mobile is more accurate for Fashion MNIST
    :return: A tuple; (tf_graph, predictor). The prediction must be performed using the corresponding tf_graph
    """
    global loaded_model
    model_paths = {"Mobile": "Mobile-99-94/", "Inception": "Inception-95-91/"}

    json_file = open(model_paths[model_name] + 'model.json', 'r')
    loaded_model = (json_file.read(), model_paths[model_name])
    json_file.close()
    g_1 = tf.Graph()
    with g_1.as_default():
        model = model_from_json(loaded_model[0])
        model.load_weights(loaded_model[1] + "model.h5")
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    loaded_model = (g_1, model)
    return loaded_model


def is_non_dominated(cand):
    """
    Given a solution, compute if it can be added to a PS stored as a MAE. It is used as a chriterion of the Hill Climbing algorithm
    :param cand: Solution
    :return: Whether it is added or not (it is also added if true).
    """

    global pareto

    for elem in pareto:
        if (cand[:3] >= elem).all():  # Unless there is a solution which dominates the candidate, return true
            print("Fail")
            return False, "o288"

    pareto += [cand[:3]]

    print(pareto)
    return True, "o288"


def improve_two_obectives(cand):
    """
        Given a solution, compute if it improves two out of three objectives of the previous solution, stored as a MAE. It is used as a chriterion of the Hill Climbing algorithm
        :param cand: Solution
        :return: Whether it is acceptable or not (it is also used as current solution if true).
        """
    global three_objectives

    if np.sum(cand[:3] < three_objectives) > 1:
        safe = np.argmin(cand[:3]/three_objectives)
        three_objectives = cand[:3]
        return True, "o" + str(safe)
    return False, -1


def evaluate(mnm):
    """
    Given a VALP, evaluate it with train-test and test-test
    :param mnm: VALP model
    :return: 6 values: Three for train-test and test-test
    """

    global loaded_model
    height, width = 90, 90
    # Train-test set
    a = mnm.predict({"i0": x_tt}, new=True)[0]
    acc = accuracy_score(a["o1"], np.argmax(c_tt, axis=1))
    mse = mean_squared_error(a["o0"], y_tt)

    images = np.array([misc.imresize(x, (height, width)).astype(float) for x in iter(np.reshape(a["o2"], (-1, 28, 28, 3)))])/255.  # Transform generations to fit the MobileNet model
    # Compute IS
    with loaded_model[0].as_default():
        predictions = loaded_model[1].predict(images)
    preds = np.argmax(predictions, axis=1)
    aux_preds = np.zeros(10)
    unique, counts = np.unique(preds, return_counts=True)
    for number, appearances in zip(unique, counts):
        aux_preds[number] = appearances
    aux_preds = aux_preds/predictions.shape[0]
    predictions = np.sort(predictions, axis=1)
    predictions = np.mean(predictions, axis=0)

    sam_error = np.sum([aux_preds[w] * np.log(aux_preds[w] / predictions[w]) if aux_preds[w] > 0 else 0 for w in range(predictions.shape[0])])

    test_res = np.array([1-acc, mse, 20-sam_error])

    # Test-test set
    a = mnm.predict({"i0": x_test}, new=True)[0]
    acc = accuracy_score(a["o1"], np.argmax(c_test, axis=1))
    mse = mean_squared_error(a["o0"], y_test)

    images = np.array([misc.imresize(x, (height, width)).astype(float) for x in iter(np.reshape(a["o2"], (-1, 28, 28, 3)))])/255.

    with loaded_model[0].as_default():
        predictions = loaded_model[1].predict(images)
    preds = np.argmax(predictions, axis=1)
    aux_preds = np.zeros(10)
    unique, counts = np.unique(preds, return_counts=True)
    for number, appearances in zip(unique, counts):
        aux_preds[number] = appearances
    aux_preds = aux_preds/predictions.shape[0]
    predictions = np.sort(predictions, axis=1)
    predictions = np.mean(predictions, axis=0)

    sam_error = np.sum([aux_preds[w] * np.log(aux_preds[w] / predictions[w]) if aux_preds[w] > 0 else 0 for w in range(predictions.shape[0])])

    return np.concatenate((np.array([1-acc, mse, 20-sam_error]), test_res))


def hill_climbing(seed, evals_remaining, local):
    """
    Perform Hill Climbing (HC)
    :param seed: Random
    :param evals_remaining: Number of evaluations allowed in total
    :param local: Number of evaluations allowed in this HC run (before restarting until reaching evals_remaining)
    :return: -- Save the data related to the HC search
    """
    global pareto
    global three_objectives
    chriterion = improve_two_obectives  # is_non_dominated

    reset_no = -1
    reset_graph(seed)
    dom = [False, -1]

    data = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, datetime.datetime.now().timestamp()]]  # This will contain the data to be saved
    while evals_remaining > 0:
        three_objectives = np.array([999, 999, 999])
        pareto = []
        reset_no += 1
        trial = 0
        # Create and evaluate first random VALP
        pivot = MNMDescriptor(10, inp_dict, outp_dict)
        pivot = recursive_creator(pivot, 0, 0)
        # pivot.print_model_graph("Pivot")
        g_2 = tf.Graph()
        with g_2.as_default():
            model = MNM(pivot, btch_sz, data_inputs, data_outputs, loss_weights)
            model.convergence_train(btch_sz, min_iter, conv_param, max_iter, {"i0": x_tt, "o1": c_tt, "o0": y_tt, "o2": x_tt}, sync=1)
            model.save_weights(str(evals_remaining))

        pivot_fit = evaluate(model)
        chriterion(pivot_fit)

        pivot.save("descriptors/Seed" + str(seed) + "_Eval" + str(evals_remaining) + "_local" + str(trial) + "_reset" + str(reset_no) + "_acc" + str(pivot_fit[0]) + "_mse" + str(pivot_fit[1]) + "_sam" + str(pivot_fit[2]) + ".txt")

        data = data + [[evals_remaining, trial, reset_no, pivot_fit[0], pivot_fit[1], pivot_fit[2], pivot_fit[3], pivot_fit[4], pivot_fit[5], -1, 1, datetime.datetime.now().timestamp()]]

        # Perform local search
        while trial < local and evals_remaining > 0:

            new = deepcopy(pivot)
            op = np.random.randint(len(ops))  # Operation choosen randomly

            # Perform the change and evaluate again
            res = ops[op](new, dom[1])
            #print(res, ops[op].__name__)
            # new.print_model_graph("Eval" + str(evals_remaining) + str(ops[op].__name__) + " " + str(res) + "_Last" + str(last_impr))
            if res == -1:
                continue
            elif op == 0 and os.path.isfile(res + ".npy"):
                os.remove(res + ".npy")
            log = str(ops[op]) + " " + str(res)
            fix_in_out_sizes(new, loaded=True)
            evals_remaining -= 1

            trial += 1
            try:
                with g_2.as_default():
                    model = MNM(new, btch_sz, data_inputs, data_outputs, loss_weights, init=False)
                    model.load_weights()
                    model.convergence_train(btch_sz, min_iter, conv_param, max_iter, {"i0": x_tt, "o1": c_tt, "o0": y_tt, "o2": x_tt}, sync=1)
                    loss = evaluate(model)
                    dom = chriterion(loss)  # Check whether it should be accepted or not
                    data = data + [[evals_remaining, trial, reset_no, loss[0], loss[1], loss[2], loss[3], loss[4], loss[5], op, int(dom[0]), datetime.datetime.now().timestamp()]]
            except Exception as e:
                #print("huehue", log, e)
                model.save_weights(str(evals_remaining))
                with g_2.as_default():
                    model.sess.close()
                # raise e

            if dom[0]:  # In case it should be accepted,
                model.save_weights(str(evals_remaining))
                pivot = new
                new.save("descriptors/Seed" + str(seed) + "_Eval" + str(evals_remaining) + "_local" + str(trial) + "_reset" + str(reset_no) + "_acc" + str(loss[0]) + "_mse" + str(loss[1]) + "_sam" + str(loss[2]) + ".txt")
                trial = 0

            model.sess.close()

    np.save("Data" + str(seed) + ".npy", data)


if __name__ == "__main__":

    ops = [divide_con, del_con, add_con, bypass]

    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(3000), nargs=1, help='an integer in the range 0..3000')
    args = parser.parse_args()

    loss_weights, (data_inputs, inp_dict), (data_outputs, outp_dict), (x_train, c_train, y_train, x_test, c_test, y_test) = diol()

    btch_sz = 150

    x_tt = x_train[int(x_train.shape[0]/6*5):, :]
    y_tt = y_train[int(y_train.shape[0]/6*5):, :]
    c_tt = c_train[int(c_train.shape[0]/6*5):]

    x_train = x_train[:int(x_train.shape[0]/6*5), :]
    y_train = y_train[:int(y_train.shape[0]/6*5), :]
    c_train = c_train[:int(c_train.shape[0]/6*5)]

    data_inputs["i0"] = x_train
    data_outputs["o0"] = y_train
    data_outputs["o1"] = c_train
    data_outputs["o2"] = x_train

    load_model()

    min_iter = 100
    max_iter = 5000
    conv_param = 1.0
    sd = args.integers[0]
    while sd < args.integers[0]+2:

        hill_climbing(sd, 50, 55)
        sd += 1

    three_objectives = np.array([0.9, 0.05, 18])
    loss_weights, (data_inputs, inp_dict), (data_outputs, outp_dict), (x_train, c_train, y_train, x_test, c_test, y_test) = diol()
    reset_graph(1)
    pvt = MNMDescriptor(10, inp_dict, outp_dict)
    pvt.load("model.txt")
    mdl = MNM(pvt, 150, data_inputs, data_outputs, loss_weights, init=False)
    mdl.load_weights()
    b = improve_two_obectives(np.array([0.1, 0.04, 7]))
    pvt.print_model_graph("huehue0")

    for i in range(1, 500):
        reset_graph(i)
        piv = deepcopy(pvt)
        print(bypass(piv, b[1]))
