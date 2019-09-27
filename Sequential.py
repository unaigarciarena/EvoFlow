from data import load_fashion
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from evolution import Evolving, accuracy_error, batch


def train_sequential(nets, placeholders, sess, graph, train_inputs, train_outputs, batch_size):
    aux_ind = 0
    predictions = {}

    with graph.as_default():
        out = nets["n0"].network_building(tf.layers.flatten(placeholders["in"]["i0"]), graph)
        predictions["n0"] = out
        out = nets["n1"].network_building(predictions["n0"], graph)
        predictions["n1"] = out

        lf = tf.losses.softmax_cross_entropy(placeholders["out"]["o1"], predictions["n1"])
        opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(lf)
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            with graph.as_default():
                _, loss = sess.run([opt, lf], feed_dict={placeholders["in"]["i0"]: batch(train_inputs["i0"], batch_size, aux_ind), placeholders["out"]["o1"]: batch(train_outputs["o0"], batch_size, aux_ind)})

            aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]

    return predictions


def eval_sequential(preds, placeholders, sess, graph, inputs, outputs):
    with graph.as_default():
        res = sess.run(tf.nn.softmax(preds["n1"]), feed_dict={placeholders["i0"]: inputs["i0"]})
        pred = np.argmax(res, axis=1)
        real = np.argmax(outputs["o0"], axis=1)
        sess.close()

        return accuracy_error(pred, real),


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion()

    OHEnc = OneHotEncoder(categories='auto')

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    OHEnc = OneHotEncoder(categories='auto')

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    e = Evolving(train_sequential, 2, [x_train], [y_train], [x_test], [y_test], eval_sequential, 150, 3, 3, n_inputs=[[28, 28], [10]], n_outputs=[[10], [10]])
    a = e.evolve()

    print(a)
