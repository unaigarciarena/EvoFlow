"""
This is a use case of EvoFlow

We face here a multiobjective problem. We create two MLP, one of which is intended for classifyin MNIST and the other one for
Fashion-MNIST. They don't interact in any moment in the model.
"""
from data import load_fashion, load_mnist
import tensorflow as tf
from metrics import accuracy_error
from evolution import Evolving, batch
from Network import MLPDescriptor
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def train(nets, placeholders, sess, graph, train_inputs, train_outputs, batch_size, _):
    aux_ind = 0
    predictions = {}

    with graph.as_default():
        # Both networks are created separately, using their own placeholders. They are not involved in any way
        out = nets["n1"].building(tf.layers.flatten(placeholders["in"]["i1"]), graph)
        predictions["n1"] = out
        out = nets["n0"].building(tf.layers.flatten(placeholders["in"]["i0"]), graph)
        predictions["n0"] = out

        loss = tf.losses.softmax_cross_entropy(placeholders["out"]["o1"], predictions["n1"]) + tf.losses.softmax_cross_entropy(placeholders["out"]["o0"], predictions["n0"])

        solver = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, var_list=[nets["n1"].List_weights, nets["n1"].List_bias, nets["n0"].List_weights, nets["n0"].List_bias])
        sess.run(tf.global_variables_initializer())

        for it in range(10):
            aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]
            # Here all placeholders need to be feeded, unlike in the previous examples, as both networks work on their own
            _ = sess.run([solver], feed_dict={placeholders["in"]["i0"]: batch(train_inputs["i0"], batch_size, aux_ind), placeholders["in"]["i1"]: batch(train_inputs["i1"], batch_size, aux_ind), placeholders["out"]["o0"]: batch(train_outputs["o0"], batch_size, aux_ind), placeholders["out"]["o1"]: batch(train_outputs["o1"], batch_size, aux_ind)})
        return predictions


def eval(preds, placeholders, sess, graph, inputs, outputs, _):

    with graph.as_default():
        # Compute predictions for both problems
        res = sess.run([tf.nn.softmax(preds["n0"]), tf.nn.softmax(preds["n1"])], feed_dict={placeholders["i0"]: inputs["i0"], placeholders["i1"]: inputs["i1"]})
        sess.close()
    # Return both accuracies
    return accuracy_error(res[0], outputs["o0"]), accuracy_error(res[1], outputs["o1"])


if __name__ == "__main__":

    fashion_x_train, fashion_y_train, fashion_x_test, fashion_y_test = load_fashion()
    mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test = load_mnist()

    OHEnc = OneHotEncoder(categories='auto')

    fashion_y_train = OHEnc.fit_transform(np.reshape(fashion_y_train, (-1, 1))).toarray()

    fashion_y_test = OHEnc.fit_transform(np.reshape(fashion_y_test, (-1, 1))).toarray()

    mnist_y_train = OHEnc.fit_transform(np.reshape(mnist_y_train, (-1, 1))).toarray()

    mnist_y_test = OHEnc.fit_transform(np.reshape(mnist_y_test, (-1, 1))).toarray()

    # In this case, we provide two data inputs and outputs
    e = Evolving(loss=train, desc_list=[MLPDescriptor, MLPDescriptor], x_trains=[fashion_x_train, mnist_x_train], y_trains=[fashion_y_train, mnist_y_train], x_tests=[fashion_x_test, mnist_x_test], y_tests=[fashion_y_test, mnist_y_test], evaluation=eval, batch_size=150, population=10, generations=10, n_inputs=[[28, 28], [28, 28]], n_outputs=[[10], [10]], sel=2)
    res = e.evolve()

    print(res[0])
