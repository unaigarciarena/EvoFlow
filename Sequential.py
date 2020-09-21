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


optimizers = [tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.AdamOptimizer]

"""
This is not a straightforward task as we need to "place" the models in the sequential order.
For this, we need to:
1- Tell the model the designed arrangement.
2- Define the training process.
3- Implement a fitness function to test the models.
"""


def train_sequential(nets, placeholders, sess, graph, train_inputs, train_outputs, batch_size, hypers):
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

    aux_ind = 0
    predictions = {}
    with graph.as_default():
        # The following four lines define the model layout:
        out = nets["n0"].building(tf.layers.flatten(placeholders["in"]["i0"]), graph, None)
        predictions["n0"] = out  # We construct n0 over its input placeholder, "in"-> "i0"
        out = nets["n1"].building(predictions["n0"], graph, None)
        predictions["n1"] = out  # We construct n1 over n0 because they are supposed to be sequential

        # Define the loss function and optimizer with the output of n1, which is the final output of the model
        lf = tf.losses.softmax_cross_entropy(placeholders["out"]["o1"], predictions["n1"])
        opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"]).minimize(lf)

        # The rest is typical training
        sess.run(tf.global_variables_initializer())

        for i in range(10):
            # As the input of n1 is the output of n0, these two placeholders need no feeding
            _, loss = sess.run([opt, lf], feed_dict={placeholders["in"]["i0"]: batch(train_inputs["i0"], batch_size, aux_ind), placeholders["out"]["o1"]: batch(train_outputs["o0"], batch_size, aux_ind)})

            aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]

    return predictions


def eval_sequential(preds, placeholders, sess, graph, inputs, outputs, _):
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
    with graph.as_default():
        res = sess.run(tf.nn.softmax(preds["n1"]), feed_dict={placeholders["i0"]: inputs["i0"]})
        sess.close()

        return accuracy_error(res, outputs["o0"]),


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion()

    OHEnc = OneHotEncoder(categories='auto')

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    # When calling the function, we indicate the training function, what we want to evolve (two MLPs), input and output data for training and
    # testing, fitness function, batch size, population size, number of generations, input and output dimensions of the networks, crossover and
    # mutation probability, the hyperparameters being evolved (name and possibilities), and whether batch normalization and dropout should be
    # present in evolution
    e = Evolving(loss=train_sequential, desc_list=[MLPDescriptor, MLPDescriptor], x_trains=[x_train], y_trains=[y_train], x_tests=[x_test],
                 y_tests=[y_test], evaluation=eval_sequential, batch_size=150, population=10, generations=10, n_inputs=[[28, 28], [10]],
                 n_outputs=[[10], [10]], cxp=0.5, mtp=0.5, hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]},
                 no_batch_norm=True, no_dropout=True)
    a = e.evolve()
    print(a[-1])
