"""
This is a use case of EvoFlow

In this instance, we require a simple, single-DNN layer classifier for which we specify the predefined loss and fitness function.
"""
from data import load_fashion
from sklearn.preprocessing import OneHotEncoder
from Network import MLPDescriptor
import numpy as np
from evolution import Evolving

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion()

    OHEnc = OneHotEncoder(categories='auto')

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()

    e = Evolving(loss="XEntropy", desc_list=[MLPDescriptor], x_trains=[x_train], y_trains=[y_train], x_tests=[x_test], y_tests=[y_test], evaluation="Accuracy_error", n_inputs=[[28, 28]], n_outputs=[[10]], batch_size=150, population=20, generations=20, iters=100, n_layers=10, max_layer_size=100, seed=0)
    a = e.evolve()

    print(a)
