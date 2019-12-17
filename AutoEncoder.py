"""
This is a use case of EvoFlow

In this instance, we require a simple, single-DNN layer classifier for which we specify the predefined loss and fitness function.
"""
from data import load_fashion
import tensorflow as tf
from Network import MLPDescriptor
import numpy as np
from evolution import Evolving


def hand_made_tf_mse(target, prediction):
    return tf.reduce_mean(tf.squared_difference(target, prediction))


def hand_made_np_mse(target, prediction):
    return np.mean(np.square(target-prediction))


if __name__ == "__main__":

    x_train, _, x_test, _ = load_fashion()

    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))

    e = Evolving(loss=hand_made_tf_mse, desc_list=[MLPDescriptor], x_trains=[x_train], y_trains=[x_train], x_tests=[x_test], y_tests=[x_test], evaluation=hand_made_np_mse, n_inputs=[[784]], n_outputs=[[784]], batch_size=150, population=5, generations=20, iters=1000, n_layers=10, max_layer_size=100, seed=0)
    a = e.evolve()

    print(a)
