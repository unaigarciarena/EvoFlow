"""
This is a use case of EvoFlow

In this instance, we handle a classification problem, which is to be solved by two DNNs combined in a sequential layout.
"""
from data import load_fashion
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from evolution import Evolving, accuracy_error, batch
from Network import MLPDescriptor, MLP
from metrics import ret_evals

evals = []

optimizers = [tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.AdamOptimizer]

"""
This is not a straightforward task as we need to "place" the models in the sequential order.
For this, we need to:
1- Tell the model the designed arrangement.
2- Define the training process.
3- Implement a fitness function to test the models.
"""


def train_wann(nets, placeholders, sess, graph, train_inputs, train_outputs, batch_size, hypers):
    """
    This function takes care of arranging the model and training it. It is used by the evolutionary internally,
    and always is provided with the same parameters
    :param nets: Dictionary with the Networks ("n0", "n1", ..., "nm", in the same order as they have been requested in the *desc_list* parameter)
    :param placeholders: Dictionary with input ("in"->"i0", "i1", ..., im) placeholders ("out"->"o0", "o1", ..., "om") for each network
    :param sess: tf session to be used when training
    :param graph: tf graph to be used when training
    :param train_inputs: Data to be used for training
    :param train_outputs: Data to be used for training
    :param batch_size: Batch_size to be used when training. It is not mandatory to use it
    :param hypers: Optional hyperparameters being evolved in case they were defined for evolution (in this case we also evolve optimizer selection and learning rate)
    :return: A dictionary with the tf layer which makes the predictions
    """

    assignations = []
    predictions = {}

    parameters = np.sum([x.shape[0].value * x.shape[1].value for x in nets["n0"].List_weights])
    string = hypers["start"]
    while len(string) < parameters:
        string = string.replace("0", hypers["p1"])
        string = string.replace("1", hypers["p2"])
    aux = 0
    ls = []
    for layer in nets["n0"].List_weights:
        lay = []
        for i in range(layer.shape[0].value):
            lay += [[int(i) for i in string[aux:aux+layer.shape[1].value]]]
            aux += layer.shape[1].value

        lay = np.array(lay)
        lay = np.where(lay == 0, hypers["weight1"], lay)
        lay = np.where(lay == 1, hypers["weight2"], lay)
        ls += [lay]

    with graph.as_default():
        # The following four lines define the model layout:
        sess.run(tf.global_variables_initializer())

        out = nets["n0"].building(tf.layers.flatten(placeholders["in"]["i0"]), graph)
        for i, layer in enumerate(nets["n0"].List_weights):  # Este for asigna el valor a todos los pesos
            assignations += [tf.assign(layer, ls[i])]
        sess.run(assignations)

        predictions["n0"] = out

    return predictions


def eval_wann(preds, placeholders, sess, graph, inputs, outputs, hypers):
    """
    Here we compute the fitness of the model. It is used by the evolutionary internally and always is provided with the same parameters
    :param preds: Dictionary created in the arranging and training function
    :param placeholders: (Only) Input placeholders: ("i0", "i1", ..., "im").
    :param sess: tf session to perform inference
    :param graph: tf graph in which inference is performed
    :param inputs: Data inputs for the model
    :param outputs: Data outputs for the metric
    :param _: hyperparameters, because we are evolving the optimizer selection and learning rate, they are unused when testing
    :return: fitness of the model (as a tuple)
    """
    global evals
    with graph.as_default():

        res = sess.run(tf.nn.softmax(preds["n0"]), feed_dict={placeholders["i0"]: inputs["i0"]})
        sess.close()
    res = np.argmax(res, axis=1)
    res = 1 - np.sum(np.argmax(outputs["o0"], axis=1) == res) / res.shape[0]

    if len(evals) % 10000 == 0:
        np.save("temp_evals.npy", np.array(evals))
    evals += [res]

    return res,


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion()
    x_train = x_train/255
    x_test = x_test/255
    y_train = np.array([0 if x <= 4 else 1 for x in y_train])
    y_test = np.array([0 if x <= 4 else 1 for x in y_test])
    OHEnc = OneHotEncoder(categories='auto')

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()

    e = Evolving(loss=train_wann, desc_list=[MLPDescriptor], x_trains=[x_train], y_trains=[y_train], x_tests=[x_test], y_tests=[y_test],
                 evaluation=eval_wann, batch_size=150, population=500, generations=10000, n_inputs=[[28, 28]], n_outputs=[[2]], cxp=0, mtp=1,
                 no_batch_norm=True, no_dropout=True, hyperparameters={"weight1": np.arange(-2, 2, 0.5), "weight2": np.arange(-2, 2, 0.5), "start": ["0", "1"], "p1": ["01", "10"], "p2": ["001", "010", "011", "101", "110", "100"]})  # Los pesos, que también evolucionan
    a = e.evolve()
    np.save("simple_res_rand.npy", np.array(ret_evals()))
    print(a[-1])
