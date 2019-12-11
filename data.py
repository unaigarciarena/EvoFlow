import tensorflow as tf

def load_fashion():

    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

    x_train = fashion_mnist[0][0]
    y_train = fashion_mnist[0][1]

    x_test = fashion_mnist[1][0]
    y_test = fashion_mnist[1][1]

    return x_train, y_train, x_test, y_test

def load_mnist():

    mnist = tf.keras.datasets.mnist.load_data()

    x_train = mnist[0][0]
    y_train = mnist[0][1]

    x_test = mnist[1][0]
    y_test = mnist[1][1]

    return x_train, y_train, x_test, y_test
