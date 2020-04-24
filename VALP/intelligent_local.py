import tensorflow as tf
import numpy as np
from VALP.descriptor import MNMDescriptor
from VALP.ModelDescriptor import recursive_creator, fix_in_out_sizes
from VALP.Model_Building import MNM
from VALP.evolution import diol
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc_err
from PIL import Image
import argparse
import random
from VALP.small_improvements import del_con, add_con, bypass, divide_con, is_deletable, is_bypassable


def ranking(orig_data):
    """
    Given an array, this function computes the ranking of each value relative each other
    :param orig_data: Values of which the ranking has to be computed
    :return: The ranking
    """
    data = np.copy(orig_data)
    values = np.sort(data)
    rank = np.zeros(data.shape)
    r = 0
    for i in range(values.shape[0]):
        for j in range(data.shape[0]):
            if data[j] == values[i]:
                rank[j] = r
                data[j] = 9223372036854775807  # MaxInt
                break
        if i < values.shape[0]-1 and values[i] < values[i+1]:
            r = i + 1
    return rank


def load_model(model_name="Mobile"):
    """
    This function loads a Fashion MNIST classifier. Used for evaluating the sampling capabilities of a model.
    :param model_name: Which model to load ("Mobile" or "Inception"). The original distance uses inception. Mobile is more accurate for Fashion MNIST
    :return: A tuple; (tf_graph, predictor). The prediction must be performed using the corresponding tf_graph
    """
    from keras import backend as k

    from keras.models import model_from_json

    global loaded_model
    model_paths = {"Mobile": "Mobile-99-94/", "Inception": "Inception-95-91/"}

    json_file = open(model_paths[model_name] + 'model.json', 'r')

    g_1 = tf.Graph()

    with g_1.as_default():
        config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})  # We use only CPU to avoid problems with the VALP trainings. Also, predicting is pretty cheap, so not much performance is lost.
        session = tf.Session(config=config)
        k.set_session(session)
        model = model_from_json(json_file.read())
        model.load_weights(model_paths[model_name] + "model.h5")
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    json_file.close()
    loaded_model = (g_1, model)

    return loaded_model


def inception_score(images):
    """
    Computes the inception score of a set of artificial generations. The trained classification model needs to be stored in "loaded_model", as does the previous function.
    :param images: List of images.
    :return: Inception score of the generations.
    """
    height, width = 90, 90

    images = np.array([np.array(Image.fromarray(x, mode="RGB").resize((height, width))) for x in np.reshape(images, (-1, 28, 28, 3))]) / 255.  # Transform images to a suitable form

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


def train_init():
    """
    This function trains random VALPs. It is used for generating random initial VALPs to which mutation operators can be applied.
    :return: -- (The structure, hyperparameters, weights, and performance of the VALP are saved in files including the seed (first one) used to generate them)
    """
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    random.seed(seed)

    name = str(seed)
    desc = MNMDescriptor(10, inp_dict, outp_dict, name=name)
    desc = recursive_creator(desc, 0, 0, seed)
    hypers = {}
    for hyper in hyps:
        hypers[hyper] = np.random.choice(hyps[hyper])

    model = MNM(desc, hypers["btch_sz"], data_inputs["Train"], data_outputs["Train"], loss_func_weights={"o0": hypers["wo0"], "o1": hypers["wo1"], "o2": hypers["wo2"]}, name=name, lr=hypers["lr"], opt=hypers["opt"], random_seed=seed)

    model.convergence_train(hypers["btch_sz"], iter_lim/5, conv_param, proportion, iter_lim, display_step=50)

    # ####### Save model characteristics.

    model.descriptor.save(path="")
    model.save_weights(path="")

    results = evaluate_model(model)

    np.save("hypers" + str(seed) + ".npy", hypers)

    np.save("orig_results" + str(seed) + ".npy", results)


def reload():
    """
    This function reloads an already trained and saved VALP (according to the specified first seed). The relevance of the different networks in the VALP is measured too.
    :return: The list of nets present in the VALP, values associated to their importance in the model (the lower value the more important), the ranked version of these values, the model descriptor, and the hyperparameters.
    """
    name = str(seed)

    hypers = np.load("hypers" + str(seed) + ".npy", allow_pickle=True).item()

    assert isinstance(hypers, dict)

    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    random.seed(seed)

    orig_res = np.load("orig_results" + str(seed) + ".npy")
    desc = MNMDescriptor(10, inp_dict, outp_dict, name=name)
    desc.load("model_" + str(seed) + ".txt")
    model = MNM(desc, hypers["btch_sz"], data_inputs["Train"], data_outputs["Train"], loss_func_weights={"o0": hypers["wo0"], "o1": hypers["wo1"], "o2": hypers["wo2"]}, name=name, load=False, init=False, random_seed=seed)

    model.initialize(load=True, load_path="")

    nets, probs, ranks = network_relevance(model, orig_res)  # List of network names, their associated probabilities (to be mutated) and their rankings.

    del model

    return nets, probs, ranks, desc, hypers


def increasing_mutations(nets, probs, desc):
    """
    Perform mutations which increase the complexity of the model.
    :param nets: A list of the networks within the VALP.
    :param probs: The probabilities of the previous networks being mutated
    :param desc: The VALP descriptor in which "nets" are placed
    :return: Variables affected by the mutation (the ones that should be trained), the result of the mutation, the component selected for mutation, (the mutation is alse performed in the "desc" desctriptor)
    """
    inverse = []  # After the "while", this variable will contain the list of networks in reverse order of "candidateness" to be mutated.

    # Decide the place in which the mutation is performed

    while len(nets) != 0:  # While there are networks which have not been added to the inverse list
        net = nets[np.argmax(probs)]  # Determine the "most useless" network still available
        cands = [net] + desc.comp_by_input(net) + desc.comp_by_output(net)  # Candidates to be added to the inverse list (the most useless one and the ones around it, because they dont need more modeling poer, as the "most useless network appears to be not needed)
        inverse += [x for x in cands if (x not in inverse) and ("i" not in x) and ("o" not in x)]  # Add to inverse list if they are networks and are not wet in

        # Update original lists.
        probs = [probs[i] for i in range(len(nets)) if nets[i] not in inverse]
        nets = [nets[i] for i in range(len(nets)) if nets[i] not in inverse]

    for comp in reversed(inverse):  # Try mutation near the networks according to the previous arrangement (could happen that some places cannot fit mutations).

        reaching_outs = list(set([x for x in desc.reachable[comp] if "o" in x]))  # Outputs affected by the mutation
        _, conns, _ = desc.get_net_context(comp)  # Connections near the selected network (which could be affected by the mutation)

        for mutation in np.random.permutation(["add_con", "divide_con"]):  # Try both mutations in case the first one does not work
            res, trainables = mutate(mutation, desc, comp, conns)
            if res != -1:
                return trainables, res, mutation, comp, reaching_outs  # If the mutation is successful, return, else try the second mutation or the next network.


def reducing_mutations(nets, probs, desc):
    """
    Apply one of the mutations which do not increase complexity
    :param nets: List of network names in a VALP
    :param probs: Probability of applying the mutations to each network
    :param desc: VALP descriptor on which "nets" are
    :return: Variables affected by the mutation (the ones that should be trained), the result of the mutation, the component selected for mutation, (the mutation is alse performed in the "desc" desctriptor)
    """

    if (np.isnan(probs)).any():  # If probabilites could not be computed or mutation has to be randomly applied, apply random probabilities
        print("NaN prob")
        probs = np.array([1/probs.shape[0]]*probs.shape[0])
    if rnd == 1:
        probs = np.array([1 / probs.shape[0]] * probs.shape[0])

    comp = np.random.choice(nets, p=probs)  # Choose network to which area the mutation is going to be applied

    reaching_outs = list(set([x for x in desc.reachable[comp] if "o" in x]))  # Outputs affected by the mutation
    _, conns, _ = desc.get_net_context(comp)
    mutations = [con for con in conns if is_deletable(desc, con)]  # Add deletable connections to the mutation pool
    mutations += ["reinit"]

    if is_bypassable(desc, comp):
        mutations += ["bypass"]
    mutation = np.random.choice(mutations)  # Choose mutation

    res, trainables = mutate(mutation, desc, comp, conns)

    return trainables, res, mutation, comp, reaching_outs


def mutate(mutation, desc, comp, conns):
    """
    Given a mutation, a VALP descriptor, and the network/connection to which the mutation has to be applied, this function applies it
    :param mutation: Mutation to be applied
    :param desc: VALP descriptor
    :param comp: Network
    :param conns: Connections
    :return: The result of the mutation (-1 if unsuccessful), variables affected by the mutation (to be trained)
    """

    # Compute variables (weights and biases) affected by the mutation
    trainables = [net for net in desc.reachable[comp] if "n" in net]
    trainables += [x for x in desc.comp_by_input(comp) if "i" not in x]
    trainables = list(set(trainables))

    if mutation[0] == "c":  # Delete connection
        res = del_con(desc, safe="None", con=mutation)
    elif mutation == "bypass":  # Bypass network
        res = bypass(desc, safe="None", net=comp)
        trainables.remove(comp)
    elif mutation == "add_con":  # Add connection
        if np.random.rand() > 0.5:  # Try using the selected component as output
            res = add_con(desc, safe="None", out=comp)
            if res == -1:  # If it didn't work, try as input
                res = add_con(desc, safe="None", inp=comp)
        else:
            res = add_con(desc, safe="None", inp=comp)  # Try using the selected component as input
            if res == -1:  # If it didn't work, try as output
                res = add_con(desc, safe="None", out=comp)
    elif mutation == "divide_con":  # Divide connection
        res = divide_con(desc, safe="None", con=np.random.choice(conns))
    elif mutation == "reinit":  # Reinitialize selected network
        desc.networks[comp].descriptor.random_init(nlayers=10, max_layer_size=100)
        res = comp
    else:
        res = -1

    fix_in_out_sizes(desc)  # Fix any problem that could have happened during mutation

    return res, trainables


def modify(nets, probs, ranks, desc, hypers):
    """
    Main function for applying a modification to a VALP. It also evaluates the VALP and saves the results
    :param nets: List of nets in a VALP
    :param probs: Probability of modifying the previous networks
    :param ranks: Rankings of the probability values
    :param desc: VALP descroptor
    :param hypers: VALP hyperparameters
    :return: --
    """

    name = str(seed)

    np.random.seed(seed2)
    tf.random.set_random_seed(seed2)
    random.seed(seed2)

    print(ranks)
    if not rnd:  # If randomness is not applied
        if (ranks.sum(axis=1) == 0).any():  # If there are any network in the bottom three in importance in all objectives
            probs = (ranks.sum(axis=1) == 0) * probs  # Only accept a network as modifiable if they rank between 3 least important networks in all three objectives
            probs = probs / np.sum(probs)  # Update probabilities once the networks more important than bottom three have been taken away
            trainables, res, mutation, comp, reaching_outs = reducing_mutations(nets, probs, desc)
        else:
            trainables, res, mutation, comp, reaching_outs = increasing_mutations(nets, probs, desc)
    else:  # Random application
        comp = np.random.choice(nets)
        _, conns, _ = desc.get_net_context(comp)
        reaching_outs = list(set([x for x in desc.reachable[comp] if "o" in x]))  # Outputs affected by the mutation
        mutations = [con for con in conns if is_deletable(desc, con)]

        mutations += ["add_con", "divide_con", "reinit"]

        if is_bypassable(desc, comp):
            mutations += ["bypass"]

        mutation = np.random.choice(mutations)
        res, trainables = mutate(mutation, desc, comp, conns)

    model = MNM(desc, hypers["btch_sz"], data_inputs["Train"], data_outputs["Train"], loss_func_weights={"o0": hypers["wo0"], "o1": hypers["wo1"], "o2": hypers["wo2"]}, name=name, load=None, init=False, random_seed=seed2, lr=0.0001)

    model.initialize(load=True, load_path="", vars=trainables)

    model.convergence_train(hypers["btch_sz"], iter_lim/100, conv_param, proportion, iter_lim/20, display_step=50)

    results = evaluate_model(model)

    del model

    if rnd == 1:
        n = "resultsrandom"
    else:
        n = "results"

    np.save(n + str(seed) + "_" + str(seed2) + ".npy", np.concatenate((results, [res, mutation, comp], reaching_outs)))


def evaluate_model(valp):
    """
    Given a VALP, evaluate in all three objectives. The test data has to be preloaded
    :param valp: VALP model
    :return:
    """

    a = valp.predict(data_inputs["Test"], [], new=True)[0]

    m2e = np.mean(mse(a["o0"], data_outputs["Test"]["o0"]))
    acc = 1 - acc_err(a["o1"][:, 0], np.argmax(data_outputs["Test"]["o1"], axis=1))
    i_d = -np.mean(inception_score(a["o2"][:100]))

    return np.array([m2e, acc, i_d])


def network_relevance(valp, orig_res):
    """
    Compute the relevance of the networks in a VALP
    :param valp: VALP model
    :param orig_res: Results of the VALP in all objectives
    :return: The networks in the VALP, selection probabilities, and rankings (0 or 1 if the networks are enough important or not)
    """
    assert isinstance(valp, MNM)

    comps = list(valp.components.keys())

    results = np.zeros((len(comps), 3))

    for i, n in enumerate(comps):

        ws, bs = valp.sess.run([valp.components[n].List_weights, valp.components[n].List_bias])  # Save trained weights
        rws = [np.random.normal(0, 0.1, x.shape) for x in ws]  # Get random weights and biases to test importance of networks
        rbs = [np.random.normal(0, 0.1, x.shape) for x in bs]

        # Change to random values
        feed_dict_w = {p: v for (p, v) in zip(valp.components[n].w_phs, rws)}
        feed_dict_b = {p: v for (p, v) in zip(valp.components[n].b_phs, rbs)}
        valp.sess.run(valp.components[n].w_assigns, feed_dict_w)
        valp.sess.run(valp.components[n].b_assigns, feed_dict_b)

        # Evaluate
        results[i] = evaluate_model(valp)/orig_res
        # Restore original values
        feed_dict_w = {p: v for (p, v) in zip(valp.components[n].w_phs, ws)}
        feed_dict_b = {p: v for (p, v) in zip(valp.components[n].b_phs, bs)}
        valp.sess.run(valp.components[n].w_assigns, feed_dict_w)
        valp.sess.run(valp.components[n].b_assigns, feed_dict_b)

    rank = np.concatenate([ranking(results[:, i]) for i in range(results.shape[1])]).reshape(results.shape, order="F")
    # From here on, the criterion is still raw
    rank[rank <= lim] = 0
    rank[rank > lim] = 1
    rank[results < 1.03] = 0

    results -= (np.min(results, axis=0)*0.9)  # To avoid [0-1] normalization, which could give problems
    results /= np.max(results, axis=0)

    results = 1 / (results[:, 0] * results[:, 1] * results[:, 2])

    results = results/np.sum(results)

    return comps, results, rank


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(3000), nargs=5, help='an integer in the range 0..3000')
    args = parser.parse_args()

    seed = args.integers[0]
    second_seed = args.integers[1]
    total = args.integers[2]
    division = args.integers[3]
    rnd = args.integers[4]
    lim = args.integers[5]
    conv_param = 1
    proportion = 0.9
    iter_lim = 10000
    hyps = {"btch_sz": [150], "wo0": [1], "wo1": [1], "wo2": [1], "opt": [0], "lr": [0.001]}

    loss_weights, (data_inputs, inp_dict), (data_outputs, outp_dict), (x_train, c_train, y_train, x_test, c_test, y_test) = diol()
    loaded_model = load_model()
    # train_init()

    second_seeds = np.arange(0, total)
    for seed2 in second_seeds[second_seed:total:division]:
        print("Seed2", seed2)
        networks, probabilities, rankings, descriptor, hyperparameters = reload()
        modify(networks, probabilities, rankings, descriptor, hyperparameters)
