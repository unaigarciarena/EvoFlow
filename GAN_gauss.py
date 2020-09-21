"""
This is a use case of EvoFlow

Another example of a multinetwork model, a GAN. In order to give an automatic fitness fuction to each GAN, we use the Inception Score (IS, https://arxiv.org/pdf/1606.03498.pdf)
We use the MobileNet model instead of Inception because it gave better accuracy scores when training it.
"""
import tensorflow as tf
import numpy as np
from evolution import Evolving, batch
from Network import MLPDescriptor
from gaussians import mmd, plt_center_assignation, create_data
import argparse
import matplotlib.pyplot as plt

best_mmd = 28888888
eval_tot = 0


def gan_train(nets, placeholders, sess, graph, train_inputs, _, batch_size, __):
    aux_ind = 0
    predictions = {}

    with graph.as_default():
        # We define the special GAN structure
        out = nets["n1"].building(placeholders["in"]["i1"], graph, _)
        predictions["gen"] = tf.nn.sigmoid(out)
        out = nets["n0"].building(placeholders["in"]["i0"], graph, _)
        predictions["realDisc"] = tf.nn.sigmoid(out)
        out = nets["n0"].building(predictions["gen"], graph, _)
        predictions["fakeDisc"] = tf.nn.sigmoid(out)

        # Loss function and optimizer
        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions["realDisc"], labels=tf.ones_like(predictions["realDisc"])) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions["fakeDisc"], labels=tf.zeros_like(predictions["fakeDisc"])))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions["fakeDisc"], labels=tf.ones_like(predictions["fakeDisc"])))

        g_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(g_loss, var_list=[nets["n1"].List_weights, nets["n1"].List_bias])
        d_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss, var_list=[nets["n0"].List_weights, nets["n0"].List_bias])
        sess.run(tf.global_variables_initializer())

        for it in range(epochs):  # Train the model


            x_mb = batch(train_inputs["i0"], batch_size, aux_ind)

            aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]
            z_mb = np.random.uniform(size=(batch_size, z_size))
            _, b = sess.run([g_solver, g_loss], feed_dict={placeholders["in"]["i1"]: z_mb})
            z_mb = np.random.uniform(size=(batch_size, z_size))
            _, b = sess.run([g_solver, g_loss], feed_dict={placeholders["in"]["i1"]: z_mb})
            z_mb = np.random.uniform(size=(batch_size, z_size))
            _, b = sess.run([g_solver, g_loss], feed_dict={placeholders["in"]["i1"]: z_mb})
            _, a = sess.run([d_solver, d_loss], feed_dict={placeholders["in"]["i0"]: x_mb, placeholders["in"]["i1"]: z_mb})

            #if it % 50 == 0: print(a, b)
            #samples = sess.run(predictions["gen"], feed_dict={placeholders["in"]["i1"]: np.random.uniform(size=(1000, z_size))})
            #plt.plot(x_mb[:, 0], x_mb[:, 1], "o")
            #plt.show()
        return predictions


def gan_eval(preds, placeholders, sess, graph, _, outputs, __):

    global best_mmd
    global eval_tot
    with graph.as_default():
        samples = sess.run(preds["gen"], feed_dict={placeholders["i1"]: np.random.uniform(size=(n_samples, z_size))})  # We generate data

    # Make it usable for MoblieNet

    mmd_value, centers = mmd(candidate=samples, target=outputs["o0"])

    if mmd_value < best_mmd:
        best_mmd = mmd_value
        plt.plot(outputs["o0"][:, 0], outputs["o0"][:, 1], "o")
        plt.plot(samples[:, 0], samples[:, 1], "o")
        plt.savefig("Evoflow_" + str(n_gauss) + "_" + str(seed) + "_" + str(eval_tot) + "_" + str(np.round(mmd_value, decimals=3)) + ".pdf")
        plt.clf()
        np.save("Samples_" + str(n_gauss) + "_" + str(seed) + "_" + str(mmd_value), samples)
    eval_tot += 1

    return mmd_value,


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(10001),
                        nargs='+', help='an integer in the range 0..3000')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum,
                        default=max, help='sum the integers (default: find the max)')

    args = parser.parse_args()
    seed = args.integers[0]
    n_gauss = args.integers[1]
    n_samples = args.integers[2]
    population = args.integers[3]
    generations = args.integers[4]
    epochs = args.integers[5]
    z_size = args.integers[6]
    x_train = create_data(n_gauss, n_samples)
    x_train = x_train - np.min(x_train, axis=0)
    x_train = x_train / np.max(x_train, axis=0)

    x_test = create_data(n_gauss, n_samples)
    x_test = x_test - np.min(x_test, axis=0)
    x_test = x_test / np.max(x_test, axis=0)
    # The GAN evolutive process is a common 2-DNN evolution
    e = Evolving(loss=gan_train, desc_list=[MLPDescriptor, MLPDescriptor], x_trains=[x_train], y_trains=[x_train], x_tests=[x_test], y_tests=[x_test], evaluation=gan_eval, batch_size=50, population=population, generations=generations, n_inputs=[[2], [z_size]], n_outputs=[[1], [2]], cxp=0.5, mtp=0.5, no_dropout=True, no_batch_norm=True)
    res = e.evolve()

    print(res[0])
