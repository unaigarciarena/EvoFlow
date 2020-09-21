"""
This is a use case of EvoFlow

We face here an unsupervised problem. We try to reduce the dimensionality of data by using a convolutional autoencoder. For that we
define a CNN that encodes data to a reduced dimension, and a transposed CNN (TCNN) for returning it to its original form.
"""
from data import load_fashion
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from evolution import Evolving, batch
from Network import ConvDescriptor, TConvDescriptor
from sklearn.metrics import mean_squared_error


optimizers = [tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.AdamOptimizer]


def train_cnn_ae(nets, placeholders, sess, graph, train_inputs, _, batch_size, hypers):
    aux_ind = 0
    predictions = {}
    with graph.as_default():
        out = nets["n0"].building(placeholders["in"]["i0"], graph, _)  # We take the ouptut of the CNN
        out = tf.layers.flatten(out)                                   # and flatten it
        out = tf.layers.dense(out, 49)                                 # before transforming it to the desired dimension

        predictions["n0"] = tf.reshape(out, (-1, 7, 7, 1))             # Then we reshape it so that the TCNN can take it

        out = nets["n1"].building(predictions["n0"], graph, _)         # Take the piece of data we're interested in (for reconstruction)
        predictions["n1"] = out[:, :28, :28, :3]                       # as the TCNN could provide more than that
        # Common training
        lf = tf.losses.mean_squared_error(placeholders["in"]["i0"], predictions["n1"])

        opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"]).minimize(lf)
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            # As the input of n1 is the output of n0, these two placeholders need no feeding
            __, loss = sess.run([opt, lf], feed_dict={placeholders["in"]["i0"]: batch(train_inputs["i0"], batch_size, aux_ind)})
            if np.isnan(loss):
                break
            aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]

    return predictions


def eval_cnn_ae(preds, placeholders, sess, graph, inputs, _, __):
    with graph.as_default():
        res = sess.run(preds["n1"], feed_dict={placeholders["i0"]: inputs["i0"]})
        sess.close()
        if np.isnan(res).any():
            return 288,
        else:
            return mean_squared_error(np.reshape(res, (-1)), np.reshape(inputs["i0"], (-1))),


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion()

    x_train = np.expand_dims(x_train, axis=3)/255
    x_train = np.concatenate((x_train, x_train, x_train), axis=3)

    x_test = np.expand_dims(x_test, axis=3)/255
    x_test = np.concatenate((x_test, x_test, x_test), axis=3)

    OHEnc = OneHotEncoder(categories='auto')

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    # Here we define a convolutional-transposed convolutional network combination
    e = Evolving(loss=train_cnn_ae, desc_list=[ConvDescriptor, TConvDescriptor], x_trains=[x_train], y_trains=[y_train], x_tests=[x_test], y_tests=[y_test], evaluation=eval_cnn_ae, batch_size=150, population=2, generations=10, n_inputs=[[28, 28, 3], [7, 7, 1]], n_outputs=[[49], [28, 28, 3]], cxp=0, mtp=1, hyperparameters = {"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2]}, no_batch_norm=False, no_dropout=False)
    a = e.evolve()

    print(a[-1])
