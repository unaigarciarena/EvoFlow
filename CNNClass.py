"""
This is a use case of EvoFlow

In this instance, we handle a classification problem, which is to be solved by two DNNs combined in a sequential layout.
The problem is the same as the one solved in Sequential.py, only that here a CNN is evolved as the first component of the model.
"""
from data import load_fashion
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from evolution import Evolving, accuracy_error, batch
from Network import MLPDescriptor, ConvDescriptor

optimizers = [tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.AdamOptimizer]


def train_cnn(nets, placeholders, sess, graph, train_inputs, train_outputs, batch_size, hypers):
    aux_ind = 0
    predictions = {}
    with graph.as_default():
        out = nets["n0"].building(placeholders["in"]["i0"], graph, None)  # We need to flatten the output of the CNN before feeding it to the MLP
        out = tf.layers.dense(tf.layers.flatten(out), 20)
        predictions["n0"] = out
        out = nets["n1"].building(predictions["n0"], graph, None)
        predictions["n1"] = out

        lf = tf.losses.softmax_cross_entropy(placeholders["out"]["o1"], predictions["n1"])

        opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"]).minimize(lf)
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            # As the input of n1 is the output of n0, these two placeholders need no feeding
            _, loss = sess.run([opt, lf], feed_dict={placeholders["in"]["i0"]: batch(train_inputs["i0"], batch_size, aux_ind), placeholders["out"]["o1"]: batch(train_outputs["o0"], batch_size, aux_ind)})

            aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]

    return predictions


def eval_cnn(preds, placeholders, sess, graph, inputs, outputs, _):
    with graph.as_default():
        res = sess.run(tf.nn.softmax(preds["n1"]), feed_dict={placeholders["i0"]: inputs["i0"]})
        sess.close()

        return accuracy_error(res, outputs["o0"]),


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion()
    # We fake a 3 channel dataset by copying the grayscale channel three times.
    x_train = np.expand_dims(x_train, axis=3)/255
    x_train = np.concatenate((x_train, x_train, x_train), axis=3)

    x_test = np.expand_dims(x_test, axis=3)/255
    x_test = np.concatenate((x_test, x_test, x_test), axis=3)

    OHEnc = OneHotEncoder(categories='auto')

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    # Here we indicate that we want a CNN as the first network of the model
    e = Evolving(loss=train_cnn, desc_list=[ConvDescriptor, MLPDescriptor], x_trains=[x_train], y_trains=[y_train], x_tests=[x_test], y_tests=[y_test],
                 evaluation=eval_cnn, batch_size=150, population=5, generations=10, n_inputs=[[28, 28, 3], [20]], n_outputs=[[20], [10]], cxp=0.5,
                 mtp=0.5, hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, no_batch_norm=False, no_dropout=False)
    a = e.evolve()

    print(a[-1])
