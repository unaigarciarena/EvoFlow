import numpy as np
from VALP.Networks import generic_descriptor, discrete_descriptor, decoder_descriptor, convolutional_descriptor, conv_dec_descriptor
from VALP.classes import InOut, NetworkComp, ModelComponent
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
import tensorflow as tf
from VALP.descriptor import MNMDescriptor
import copy
import random

"""
The critical phase of the descriptor creation method is the moment when the descriptor needs as many components
(has as many active_outputs) for fulfilling the VVC conditions as it can add (i.e., comp_limit - |comp|)
"""


def complete_model(model, conv_prob):
    """
    This function takes a descriptor in a critical phase and returns a VVC
    :param model: descriptor in a critical phase
    :param conv_prob: Probability of adding a C-Decoder instead of a MLP-Decoder
    :return: VVC
    """

    while len(model.active_outputs) > 0:  # Until there are no active_inputs

        output = model.active_outputs[0]
        inp = model.random_input(output)  # Look if there is a network that can serve the output

        if inp != -1:  # If there is, connect it
            model.connect(inp, output)

            model.delete_active_by_index(output)

            model.reachable[inp] = model.reachable[inp] + model.reachable[output]
            for comp in list(model.reachable.keys()):
                if inp in model.reachable[comp]:
                    model.reachable[comp] += model.reachable[output]
        else:  # In case there is not, create a new one
            aux = np.random.randint(2, 50)
            inp = None
            if model.comp_by_ind(output).taking.type in "discrete":
                d = discrete_descriptor(0, aux)
                inp = NetworkComp(d, InOut(data_type="values", size=0), InOut(data_type="discrete", size=aux), model.get_depth(output)+1)

            elif model.comp_by_ind(output).taking.type in "values":
                d = generic_descriptor(0, aux)
                inp = NetworkComp(d, InOut(data_type="features", size=0), InOut(data_type="values", size=aux), model.get_depth(output)+1)

            elif model.comp_by_ind(output).taking.type in "samples":
                if "o" not in output or np.random.rand() > conv_prob:
                    d = decoder_descriptor(0, aux)
                    inp = NetworkComp(d, InOut(data_type="values", size=0), InOut(data_type="samples", size=aux), model.get_depth(output)+1)
                else:
                    d = conv_dec_descriptor(0, model.comp_by_ind(output).taking.size)
                    inp = NetworkComp(d, InOut(data_type="values", size=0), InOut(data_type="samples", size=aux), model.get_depth(output)+1)

            elif "eatures" in model.comp_by_ind(output).taking.type:  # This should never be used

                d = convolutional_descriptor(0, (aux % 4)+1, model.comp_by_ind(output), model.inputs)
                inp = NetworkComp(d, InOut(data_type="features", size=0), InOut(data_type="features", size=(aux % 4)+1), model.get_depth(output)+1)

            inp = model.add_net(inp)
            model.connect(inp, output)

            if "o" in output:
                model.reachable[inp] = [output, inp]
            else:
                model.reachable[inp] = model.reachable[output] + [output, inp]

            model.delete_active_by_index(output)
            model.active_outputs += [inp]

    return model


def recursive_creator(model, depth, conv_prob):  # Main recursive function
    """
    This function takes an (presumably) empty model and returns a VVC
    :param model: Empty model
    :param depth: At what point the depth count should start. Zero makes the most sense
    :param conv_prob: Probability of adding a C-Decoder instead of a MLP-Decoder
    :return: VVC
    """

    model.constructed = True

    model = recursive_function(model, depth, conv_prob)

    fix_in_out_sizes(model)
    return model


def recursive_function(model, depth, conv_prob):
    """
    Real recursive function. Takes a model in any state, and adds a component (with its corresponding connection)
    or a simple connection.
    :param model: Model in any state (empty, intermediate, critical, complete)
    :param depth: Depth level
    :param conv_prob: Probability of adding a C-Decoder instead of a MLP-Decoder
    :return: VVC
    """

    if (model.max_comp - model.comp_number()) <= len([x for x in model.active_outputs if x in model.outputs or ("Network" in type(model.comp_by_ind(x)).__name__ and "Decoder" in type(model.comp_by_ind(x).descriptor).__name__)]):
        # If the descriptor is on its critical phase,
        return complete_model(model, conv_prob)

    elements = model.networks.copy()

    elements.update(model.outputs)

    a = list(elements.keys())

    random.shuffle(a)

    con_output = np.random.choice(a)  # This element will get a new input in when this function ends
    m = copy.deepcopy(model)  # Create a copy of the model

    index = m.random_input(con_output)  # select an existing component that can provide input to con_output

    if np.random.rand() < 0.5 or index == -1:  # if there is no component capable of giving input to a, or by chance, create a new one

        aux = np.random.randint(2, 50)
        con_input = None
        # Depending on the required type, one or other network is created
        if m.comp_by_ind(con_output).taking.type in "discrete" or (np.random.rand() < 0.5 and "o" not in con_output):
            d = discrete_descriptor(0, aux)
            con_input = NetworkComp(d, InOut(data_type="", size=0), InOut(data_type="discrete", size=aux), m.get_depth(con_output)+1)

        if m.comp_by_ind(con_output).taking.type in "values":
            if con_output not in m.active_outputs and np.random.rand() < conv_prob and "o" not in con_output and "Decoder" not in type(m.comp_by_ind(con_output).descriptor).__name__:
                d = convolutional_descriptor(0, (aux % 4)+1, model.comp_by_ind(con_output), model.inputs)
                con_input = NetworkComp(d, InOut(data_type="features", size=0), InOut(data_type="features", size=(aux % 4)+1), model.get_depth(con_output)+1)
            else:
                d = generic_descriptor(0, aux)
                con_input = NetworkComp(d, InOut(data_type="", size=0), InOut(data_type="values", size=aux), m.get_depth(con_output)+1)

        elif "samples" in m.comp_by_ind(con_output).taking.type:
            if "o" not in con_output or np.random.rand() > conv_prob:
                if np.random.rand() > 0.5:
                    d = decoder_descriptor(0, aux)
                    con_input = NetworkComp(d, InOut(data_type="values", size=0), InOut(data_type="samples", size=aux), m.get_depth(con_output)+1)
                else:
                    d = generic_descriptor(0, aux)
                    con_input = NetworkComp(d, InOut(data_type="samples", size=0), InOut(data_type="samples", size=aux), m.get_depth(con_output)+1)
            else:
                d = conv_dec_descriptor(0, model.comp_by_ind(con_output).taking.size)
                con_input = NetworkComp(d, InOut(data_type="values", size=0), InOut(data_type="samples", size=aux), m.get_depth(con_output)+1)

        elif "eatures" in model.comp_by_ind(con_output).taking.type:
            d = convolutional_descriptor(0, (aux % 4)+1, model.comp_by_ind(con_output), model.inputs)
            con_input = NetworkComp(d, InOut(data_type="features", size=0), InOut(data_type="features", size=(aux % 4)+1), model.get_depth(con_output)+1)

        # Update the model descriptor with the new component
        index = m.add_net(con_input)

        if "o" in con_output:
            m.reachable[index] = [con_output, index]
        else:
            m.reachable[index] = m.reachable[con_output] + [con_output, index]

        m.networks[index] = con_input

        m.active_outputs += [index]

    else:  # Use an existing component as the input for the new connection
        m.reachable[index] = m.reachable[index] + m.reachable[con_output]
        for comp in list(m.reachable.keys()):
            if index in m.reachable[comp]:
                m.reachable[comp] += m.reachable[con_output]

    m.connect(index, con_output)

    m.delete_active_by_index(con_output)

    a = recursive_function(m, depth+1, conv_prob)

    return a


def fix_in_out_sizes(model, loaded=False):
    """
    This function makes the necessary changes in a VVC so that all connections match their requirements; e.g., there are no connections
    ending in outputs which provide data with a smaller size than required.
    :param model: VVC
    :param loaded: Whether the model has been loaded or newly constructed. Models that have been "fixed" and saved before require less testing
    :return: VVC with appropriate connections
    """

    for out in model.outputs:  # Change the output of the networks in the last layer to the maximum required size
        for comp in model.comp_by_input(out):
            if ("Model" not in type(model.comp_by_ind(comp)).__name__) and ("ConvDecoder" not in type(model.comp_by_ind(comp).descriptor).__name__):

                model.comp_by_ind(comp).update_output(model.comp_by_ind(out).taking.size)
                # print(out, comp, model.comp_by_ind(out).taking.size, vars(model.comp_by_ind(comp).descriptor.network), vars(model.comp_by_ind(comp).descriptor), vars(model.comp_by_ind(comp)))

    for inp in model.networks:  # Increase the input size of the networks to fit all the incomes they have
        for comp in model.comp_by_input(inp):
            model.comp_by_ind(inp).increase_input(model.comp_by_ind(comp).producing.size)

    if not loaded:  # The result of applying the following instructions is saved with the model. So, if the model is loaded and not new,
        # these instructions must not be run

        for c in model.connections:  # Change the input/output size of the networks when necessary
            con = model.connections[c]
            if "o" not in con.output:
                con.info.size = model.comp_by_ind(con.input).producing.size

                model.comp_by_ind(con.output).descriptor.n_inputs += 1
            else:
                con.info.size = model.comp_by_ind(con.output).taking.size

        for n in model.networks:  # Decide what connections will be deleted from a decoder when sampling

            if "Decoder" in type(model.networks[n].descriptor).__name__:
                for i in range(model.networks[n].descriptor.n_inputs):
                    if np.random.rand() < 0.3 and len(model.networks[n].descriptor.rands) > 0:
                        model.networks[n].descriptor.conds += [i]
                    else:
                        model.networks[n].descriptor.rands += [i]


def cell_descriptor(model, inherit_prob, dupl_prob):  # Not used. Probably will never :c

    # ########################## Additive construction ######################### #
    inputs = {"n0": np.array(list(model.inputs.keys()))}
    for i in model.outputs.keys():
        inputs[i] = ["n0"]
    outputs = {"n0": np.array(list(model.outputs.keys()))}
    for i in model.inputs.keys():
        outputs[i] = ["n0"]
    in_aux = 0
    out_aux = np.random.randint(1, 10)
    d = NetworkComp(generic_descriptor(in_aux, out_aux), InOut("Values", in_aux), InOut("Values", out_aux), 0)
    model.add_net(d, "n0")

    for i in range(1, model.max_comp):

        new_net = "n" + str(i)

        ins = np.random.choice(inputs["n0"], size=np.random.randint(1, max(2, int(len(inputs["n0"])*inherit_prob))), replace=False)  # Select what inputs are passed to the new network
        for j in ins:  # This loop deletes some of the inputs from the network (some are duplicated others are just passed)
            outputs[j] = np.append(outputs[j], [new_net])
            if np.random.random() < 1-dupl_prob and len(inputs["n0"]) > 1:
                inputs["n0"] = inputs["n0"][inputs["n0"] != j]
                outputs[j] = outputs[j][outputs[j] != "n0"]

        outs = np.random.choice(outputs["n0"], size=np.random.randint(1, max(2, int(len(outputs["n0"])*inherit_prob))), replace=False)  # Select what outputs are passed to the new network
        for j in outs:  # This loop deletes some of the outputs from the network (some are duplicated others are just passed)
            inputs[j] = np.append(inputs[j], [new_net])
            if np.random.random() < 1-dupl_prob and len(outputs["n0"]) > 1:
                outputs["n0"] = outputs["n0"][outputs["n0"] != j]
                inputs[j] = inputs[j][inputs[j] != "n0"]

        action = np.random.random()  # Define whether the new net is placed before, after or in parallel to n0

        if action < 1/3:  # After
            ins = np.append(ins, ["n0"])
            outputs["n0"] = np.append(outputs["n0"], new_net)

        elif action < 2/3:  # Before
            outs = np.append(outs, ["n0"])
            inputs["n0"] = np.append(inputs["n0"], new_net)

        inputs[new_net] = ins
        outputs[new_net] = outs
        aux_out = np.random.randint(1, 10)
        d = NetworkComp(generic_descriptor(0, aux_out), InOut("Values", 0), InOut("Values", aux_out), 0)

        model.add_net(d, new_net)

    # ############################# Network type decision ############################ #

    # ############################# Descriptive transformation ####################### #

    fix_in_out_sizes(model)

    for comp in inputs:
        for inp in inputs[comp]:
            model.connect(inp, comp)


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

    data_outputs = {"o0": y_train}

    outp_dict = {"o0": ModelComponent(InOut(size=y_train.shape[1], data_type="values"), None, 0)}

    # Separated one hot encoding
    """
    for i in range(a.shape[1]):
        data_outputs["o" + str(i+1)] = np.reshape(a[:, i], [-1, 1])
        outp_dict["o" + str(i+1)] = ModelComponent(InOut(size=1, type="values"), None)
    """
    # Merged one hot encoding
    data_outputs["o1"] = OHEnc.fit_transform(np.reshape(c_train, (-1, 1))).toarray()
    outp_dict["o1"] = ModelComponent(InOut(size=data_outputs["o1"].shape[1], data_type="discrete"), None, 0)

    # Samples

    data_outputs["o2"] = x_train
    outp_dict["o2"] = ModelComponent(InOut(size=x_train.shape[1:], data_type="samples"), None, 0)

    for seed in range(500):
        print(seed)

        np.random.seed(seed)
        random.seed(seed)

        model_descriptor = MNMDescriptor(10, inp_dict, outp_dict)

        model_descriptor = recursive_creator(model_descriptor, 0, conv_prob=0)

        print(model_descriptor)
