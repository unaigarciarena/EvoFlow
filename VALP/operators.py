# import os
from VALP.descriptor import MNMDescriptor
from VALP.ModelDescriptor import recursive_creator
from VALP.evolution import diol
from VALP.Model_Building import MNM
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
# from VALP.intelligent_local import inception_score, load_model
import copy
from shutil import copyfile


def network_clone(desc, net):  # Acuérdate del fix in out al final de las mutaciones
    """
    Given a descriptor, this function adds a connection.
    :param desc: VALP descriptor
    :param net: network to be cloned
    :return: The name of the recently added network (or -1 in case it fails)
    """
    assert isinstance(desc, MNMDescriptor)
    desc.print_model_graph("pre")
    clone = copy.deepcopy(desc.networks[net])
    clone_name = desc.add_net(clone)
    _, in_cons, out_cons, _ = desc.get_net_context(net)
    new_conns = {"Ins": [], "Outs": []}
    for c in in_cons:
        new_c = copy.deepcopy(desc.connections[c])
        new_c.input = clone_name
        c_name = desc.add_connection(new_c)
        new_c.name = c_name
        new_conns["Outs"] += [c_name]
    for c in out_cons:
        new_c = copy.deepcopy(desc.connections[c])
        new_c.output = clone_name
        c_name = desc.add_connection(new_c)
        new_c.name = c_name
        new_conns["Ins"] += [c_name]
    return desc, clone_name, new_conns


def network_clone_morphism(desc, net):
    desc, new_net, new_conns = network_clone(desc, net)
    receivers = [desc.connections[con].output for con in new_conns["Outs"]]
    copyfile(desc.name + "_" + net + ".npy", desc.name + "_" + new_net + ".npy")
    for receiver in receivers:
        if receiver[0] == "n":
            start = 0
            end = 0
            for r in desc.comps_below[receiver]:
                if net == r:
                    end = start + desc.networks[r].producing.size
                    break
                else:
                    start += desc.networks[r].producing.size
            if end == 0:
                raise
            weights = np.load(desc.name + "_" + receiver + ".npy", allow_pickle=True)
            weights[0][0][start:end] = weights[0][0][start:end]/2
            weights[0][0] = np.vstack((weights[0][0], weights[0][0][start:end]))
            np.save(desc.name + "_" + receiver + ".npy", weights)
    return desc, new_net, new_conns


def test_clone_morphism(d_m):

    d_m, _, _ = network_clone_morphism(d_m, "n1")
    model_m = MNM(d_m, 150, data_inputs["Train"], data_outputs["Train"], loss_weights, init=False)
    model_m.load_weights("1")
    a_m = model_m.predict({"i0": x_test}, new=True)[0]
    acc_m = accuracy_score(a_m["o1"], np.argmax(c_test, axis=1))
    mse_m = mean_squared_error(a_m["o0"], y_test)
    print(acc_m, mse_m)
    # print(model.sess.run(model.components["n3"].List_layers[0][:2], feed_dict={model.inputs["i0"]: x_test}))
    # load_model()
    # sam_error = inception_score(a["o2"][:100])


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


if __name__ == "__main__":
    loss_weights, (data_inputs, inp_dict), (data_outputs, outp_dict), (x_train, c_train, y_train, x_test, c_test, y_test) = diol()
    d = MNMDescriptor(5, inp_dict, outp_dict, name="1")
    d = recursive_creator(d, 0, 0, seed=0)
    d.print_model_graph("huehue1")
    model = MNM(d, 150, data_inputs["Train"], data_outputs["Train"], loss_weights, init=False)
    #model.load_weights("1")
    #model.save_weights("1")
    a = model.predict({"i0": x_test}, new=True)[0]
    acc = accuracy_score(a["o1"], np.argmax(c_test, axis=1))
    mse = mean_squared_error(a["o0"], y_test)
    print(acc, mse)
    test_clone_morphism(d)
