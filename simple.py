from data import load_fashion
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from evolution import Evolving

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_fashion()

    OHEnc = OneHotEncoder(categories='auto')

    y_train = OHEnc.fit_transform(np.reshape(y_train, (-1, 1))).toarray()

    OHEnc = OneHotEncoder(categories='auto')

    y_test = OHEnc.fit_transform(np.reshape(y_test, (-1, 1))).toarray()
    e = Evolving("XEntropy", 1, [x_train], [y_train], [x_test], [y_test], "Accuracy_error", 150, 20, 20)
    a = e.evolve()

    print(a)
