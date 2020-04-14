import tensorflow as tf
import numpy as np
from VALP.descriptor import MNMDescriptor
from VALP.ModelDescriptor import recursive_creator
from VALP.Model_Building import MNM
from VALP.evolution import diol
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc_err
from PIL import Image
import argparse
import random
from VALP.small_improvements import del_con, add_con, bypass, divide_con, is_deletable, is_bypassable


def load_model(model_name="Mobile"):
    from keras import backend as K

    from keras.models import model_from_json

    global loaded_model
    model_paths = {"Mobile": "Mobile-99-94/", "Inception": "Inception-95-91/"}

    json_file = open(model_paths[model_name] + 'model.json', 'r')

    g_1 = tf.Graph()

    with g_1.as_default():
        config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
        session = tf.Session(config=config)
        K.set_session(session)
        model = model_from_json(json_file.read())
        model.load_weights(model_paths[model_name] + "model.h5")
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    json_file.close()
    loaded_model = (g_1, model)

    return loaded_model


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


def train_init():

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

    model.convergence_train(hypers["btch_sz"], 2000, 1, 0.9, 10000, display_step=50)

    model.descriptor.save(path="")
    model.save_weights(path="")

    results = evaluate_model(model)

    np.save("hypers" + str(seed) + ".npy", hypers)

    np.save("orig_results" + str(seed) + ".npy", results)


def reload():
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

    nets, probs = network_relevancy(model, orig_res)

    del model

    return nets, probs, desc, hypers


def modify(nets, probs, desc, hypers):

    name = str(seed)

    np.random.seed(seed2)
    tf.random.set_random_seed(seed2)
    random.seed(seed2)
    if (np.isnan(probs)).any() or rnd == 1:
        print("NaN prob")
        probs = np.array([1/probs.shape[0]]*probs.shape[0])

    #comp = np.random.choice(nets, p=probs)
    comp = nets[np.argmax(probs)]
    _, conns, _ = desc.get_net_context(comp)

    mutations = [con for con in conns if is_deletable(desc, con)]

    mutations += ["add_con", "divide_con", "reinit"]

    if is_bypassable(desc, comp):
        mutations += ["bypass"]

    mutation = np.random.choice(mutations)

    if mutation[0] == "c":
        res = del_con(desc, safe="None", con=mutation)
    elif mutation == "bypass":
        res = bypass(desc, safe="None", net=comp)
    elif mutation == "add_con":
        if np.random.rand() > 0.5:
            res = add_con(desc, safe="None", out=comp)
        else:
            res = add_con(desc, safe="None", inp=comp)
            if res == -1:
                res = add_con(desc, safe="None", out=comp)
    elif mutation == "divide_con":
        res = divide_con(desc, safe="None", con=np.random.choice(conns))
    elif mutation == "reinit":
        res = desc.networks[comp].descriptor.random_init(nlayers=10, max_layer_size=100)
    else:
        res = "avido 1 prolema"

    print(res, mutation)

    model = MNM(desc, hypers["btch_sz"], data_inputs["Train"], data_outputs["Train"], loss_func_weights={"o0": hypers["wo0"], "o1": hypers["wo1"], "o2": hypers["wo2"]}, name=name, load=False, init=False, random_seed=seed2)

    model.initialize(load=True, load_path="")

    model.convergence_train(hypers["btch_sz"], 600, 1, 0.9, 4000, display_step=50)

    results = evaluate_model(model)

    del model

    if rnd == 1:
        n = "resultsrandom"
    else:
        n = "results"

    np.save(n + str(seed) + "_" + str(seed2) + ".npy", np.concatenate((results, [res, mutation], list(set([x for x in desc.reachable[comp] if "o" in x])))))


def evaluate_model(valp):

    a = valp.predict(data_inputs["Test"], [], new=True)[0]

    m2e = np.mean(mse(a["o0"], data_outputs["Test"]["o0"]))
    acc = 1 - acc_err(a["o1"][:, 0], np.argmax(data_outputs["Test"]["o1"], axis=1))
    i_d = -np.mean(inception_score(a["o2"][:100]))

    return np.array([m2e, acc, i_d])


def network_relevancy(valp, orig_res):

    assert isinstance(valp, MNM)

    comps = list(valp.components.keys())

    results = np.zeros((len(comps), 3))

    for i, n in enumerate(comps):

        ws, bs = valp.sess.run([valp.components[n].List_weights, valp.components[n].List_bias])
        rws = [np.random.normal(0, 0.1, x.shape) for x in ws]
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

    results -= (np.min(results, axis=0)*0.9)
    results /= np.max(results, axis=0)
    results = 1 / (results[:, 0] * results[:, 1] * results[:, 2])
    return comps, results/np.sum(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(3000), nargs=5, help='an integer in the range 0..3000')
    args = parser.parse_args()

    seed = args.integers[0]
    second_seed = args.integers[1]
    total = args.integers[2]
    division = args.integers[3]
    rnd = args.integers[4]

    hyps = {"btch_sz": [150], "wo0": [1], "wo1": [1], "wo2": [1], "opt": [0], "lr": [0.001]}

    loss_weights, (data_inputs, inp_dict), (data_outputs, outp_dict), (x_train, c_train, y_train, x_test, c_test, y_test) = diol()
    loaded_model = load_model()
    #train_init()

    second_seeds = np.arange(0, total)
    for seed2 in second_seeds[second_seed:total:division]:
        print("Seed2", seed2)
        networks, probabilities, descriptor, hyperparameters = reload()
        modify(networks, probabilities, descriptor, hyperparameters)
