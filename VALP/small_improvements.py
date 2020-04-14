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


def divide_con(desc, safe, con=""):

    nets = []
    trial = 50
    inp_comp = None
    c = None
    if con == "":
        while len(nets) == 0 and trial > 0:
            trial += 1

            con = np.random.choice([w for w in desc.connections.keys() if safe not in desc.reachable[desc.connections[w].output]])

            c = desc.pop_con(con)
            inp_comp = desc.comp_by_ind(c.input)
            out_comp = desc.comp_by_ind(c.output)
            if inp_comp.type == "Model":
                inp = [inp_comp.producing.type]
            else:
                inp = nets_produce[type(inp_comp.descriptor)]

            if out_comp.type == "Model":
                out = [out_comp.taking.type]
            else:
                out = nets_require[type(out_comp.descriptor)]

            inp_nets = []
            out_nets = []
            for kind in inp:
                inp_nets += types_come_from[kind]

            for kind in out:
                out_nets += types_go_to[kind]

            nets = set(inp_nets).intersection(set(out_nets))

            desc.connections[con] = c
    else:
        c = desc.pop_con(con)
        inp_comp = desc.comp_by_ind(c.input)
        out_comp = desc.comp_by_ind(c.output)
        if inp_comp.type == "Model":
            inp = [inp_comp.producing.type]
        else:
            inp = nets_produce[type(inp_comp.descriptor)]

        if out_comp.type == "Model":
            out = [out_comp.taking.type]
        else:
            out = nets_require[type(out_comp.descriptor)]

        inp_nets = []
        out_nets = []
        for kind in inp:
            inp_nets += types_come_from[kind]

        for kind in out:
            out_nets += types_go_to[kind]

        nets = set(inp_nets).intersection(set(out_nets))

        desc.connections[con] = c
    if len(nets) == 0:
        return -1

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

    desc.connect(c.input, new, c.name)
    desc.connect(new, c.output)

    desc.reachable[new] = [new] + desc.reachable[c.output]
    desc.reachable[c.input] += [new]

    for node in list(desc.networks.keys()) + list(desc.inputs.keys()):
        if c.input in desc.reachable[node]:
            desc.reachable[node] += [new]

    return new


def add_con(desc, safe, inp=None, out=None):

    able = False
    trial = 50

    while not able and trial > 0:

        if inp is None:
            if out is None:
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
            if out is None:
                options = [x for x in list(desc.networks.keys()) + list(desc.outputs.keys()) if inp not in desc.reachable[x] and safe not in desc.reachable[x]]
                if len(options) > 0:
                    able = True
                    out = np.random.choice(options)

            elif inp in desc.reachable[out]:
                raise Exception("U cant make recursive connections yet")

    if not able:
        return -1

    c = desc.connect(inp, out)
    desc.reachable[inp] += desc.reachable[out]

    for node in list(desc.networks.keys()) + list(desc.inputs.keys()):
        if inp in desc.reachable[node]:
            desc.reachable[node] += desc.reachable[out]
    if "n" in desc.connections[c].output:
        desc.networks[desc.connections[c].output].descriptor.n_inputs += 1
        if "ecoder" in type(desc.networks[desc.connections[c].output].descriptor).__name__:
            if np.random.rand() > 0.3:
                desc.networks[desc.connections[c].output].descriptor.rands += [desc.networks[desc.connections[c].output].descriptor.n_inputs]
            else:
                desc.networks[desc.connections[c].output].descriptor.conds += [desc.networks[desc.connections[c].output].descriptor.n_inputs]
    return c


def del_con(desc, safe, con=""):
    if con == "":
        keys = list(desc.connections.keys())
        np.random.shuffle(keys)
        con = keys.pop()
        while len(keys) > 0 and not (is_deletable(desc, con) and safe not in desc.reachable[desc.connections[con].output]):
            con = keys.pop()

    if not is_deletable(desc, con):
        return -1

    if "n" in desc.connections[con].output:
        desc.networks[desc.connections[con].output].descriptor.n_inputs -= 1
        if "ecoder" in type(desc.networks[desc.connections[con].output].descriptor).__name__:
            if (np.random.rand() < 0.5 and len(desc.networks[desc.connections[con].output].descriptor.rands) > 1) or len(desc.networks[desc.connections[con].output].descriptor.conds) < 1:
                desc.networks[desc.connections[con].output].descriptor.rands.pop()
            else:
                desc.networks[desc.connections[con].output].descriptor.conds.pop()

    to_be_deleted = desc.reachable[desc.connections[con].output]
    to_be_actualized = []

    for net in desc.networks:
        if desc.connections[con].input in desc.reachable[net]:
            desc.reachable[net] = [w for w in desc.reachable[net] if w not in to_be_deleted] + [net]
            to_be_actualized += [net]

    for net in desc.networks:
        if net in to_be_actualized:
            for goal in to_be_deleted:
                if desc.conn_exists(net, goal) or len([w for w in desc.nodes() if w in desc.reachable[net] and goal in desc.reachable[w]]) > 0:
                    aux = desc.reachable[goal]
                    desc.reachable[net] = desc.reachable[net] + aux

    del desc.connections[con]
    return con


def is_deletable(desc, con):
    ins_of_out = []
    outs_of_in = []
    for conn in desc.connections:
        if conn not in con:
            if desc.connections[conn].input == desc.connections[con].input:
                outs_of_in += [conn]
            if desc.connections[conn].output == desc.connections[con].output:
                ins_of_out += [conn]

    return (len(ins_of_out) > 0) and (len(outs_of_in) > 0)


def bypass(desc, safe, net=""):
    if net == "":
        keys = list(desc.networks.keys())
        np.random.shuffle(keys)
        net = keys.pop()
        while len(keys) > 0 and not (is_bypassable(desc, net) and safe not in desc.reachable[net]):
            net = keys.pop()

    if not is_bypassable(desc, net):
        return -1

    inps, cons, outs = desc.get_net_context(net)
    for inp in inps:
        for out in outs:
            if len(cons) > 0:
                con_name = cons.pop()
            else:
                con_name = ""
            desc.connect(inp, out, con_name)

    while len(cons) > 0:
        del desc.connections[cons.pop()]

    del desc.networks[net]

    for network in desc.networks:
        desc.reachable[network] = [w for w in desc.reachable[network] if not w == net]
    for inp in desc.inputs:
        desc.reachable[inp] = [w for w in desc.reachable[inp] if not w == net]

    del desc.reachable[net]

    return net


def is_bypassable(desc, net):
    if desc.comp_by_ind(net).taking.type == desc.comp_by_ind(net).producing.type:
        return True
    if "alues" in desc.comp_by_ind(net).taking.type and "iscre" in desc.comp_by_ind(net).producing.type:
        return True
    if "iscre" in desc.comp_by_ind(net).taking.type and "alues" in desc.comp_by_ind(net).producing.type:
        return True
    if "amples" in desc.comp_by_ind(net).taking.type and "alues" in desc.comp_by_ind(net).producing.type:
        return True
    return False


def keras_model(xtr, ytr, xts, yts):

    np.random.seed(2017)
    tf.set_random_seed(2017)

    height, width = 90, 90
    input_image = Input(shape=(height, width, 3))

    # base_model = InceptionV3(input_tensor=input_image, include_top=False, pooling='avg')
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

    global pareto

    for elem in pareto:
        if (cand[:3] >= elem).all():
            print("Fail")
            return False, "o288"

    pareto += [cand[:3]]

    print(pareto)
    return True, "o288"


def improve_two_obectives(cand):

    global three_objectives

    if np.sum(cand[:3] < three_objectives) > 1:
        safe = np.argmin(cand[:3]/three_objectives)
        three_objectives = cand[:3]
        return True, "o" + str(safe)
    return False, -1


def evaluate(mnm):

    global loaded_model
    height, width = 90, 90

    a = mnm.predict({"i0": x_tt}, new=True)[0]
    acc = accuracy_score(a["o1"], np.argmax(c_tt, axis=1))
    mse = mean_squared_error(a["o0"], y_tt)

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

    test_res = np.array([1-acc, mse, 20-sam_error])

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

    global pareto
    global three_objectives
    chriterion = improve_two_obectives  # is_non_dominated

    reset_no = -1
    reset_graph(seed)
    dom = [False, -1]

    data = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, datetime.datetime.now().timestamp()]]
    while evals_remaining > 0:
        three_objectives = np.array([999, 999, 999])
        pareto = []
        reset_no += 1
        trial = 0
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

        while trial < local and evals_remaining > 0:
            #print(evals_remaining)
            new = deepcopy(pivot)
            op = np.random.randint(len(ops))
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
                    dom = chriterion(loss)
                    data = data + [[evals_remaining, trial, reset_no, loss[0], loss[1], loss[2], loss[3], loss[4], loss[5], op, int(dom[0]), datetime.datetime.now().timestamp()]]
            except Exception as e:
                #print("huehue", log, e)
                model.save_weights(str(evals_remaining))
                with g_2.as_default():
                    model.sess.close()
                # raise e

            if dom[0]:
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
