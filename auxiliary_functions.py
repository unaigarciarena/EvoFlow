import keras
import numpy as np
from keras.initializers import RandomUniform, RandomNormal, glorot_uniform
from keras.layers import Dense, Activation, BatchNormalization, Dropout, InputLayer

from Network import MLPDescriptor, ConvDescriptor


def batch(x, size, i):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param size: Size of the batch desired
    :param i: Index of the last solution used in the last epoch
    :return: The index of the last solution in the batch (to be provided to this same
             function in the next epoch, the solutions in the actual batch, and their
             respective fitness scores
    """

    if i + size > x.shape[0]:  # In case there are not enough solutions before the end of the array

        index = i + size-x.shape[0]  # Select all the individuals until the end and restart
        return np.concatenate((x[i:, :], x[:index, :]))
    else:  # Easy case
        index = i+size
        return x[i:index, :]


def init_conv_desc(input_dim, output_dim, max_lay, max_size):

    layers = np.random.choice([0, 1, 2], size=np.random.randint(np.max((max_lay, 2))))  # Number of layers = 2

    filters = [np.random.randint(1, 5) if layers[i] == 2 else -1 for i in range(layers.shape[0])]
    init = [np.random.randint(0, 2) if layers[i] == 2 else -1 for i in range(layers.shape[0])]
    act = [np.random.randint(0, 2) if layers[i] == 2 else -1 for i in range(layers.shape[0])]

    sizes = np.concatenate((np.random.randint(2, 4, size=(layers.shape[0], 2)), np.random.randint(1, 3, size=(layers.shape[0], 1))), axis=1)
    strides = np.random.randint(1, 2, size=layers.shape[0])
    descriptor = ConvDescriptor(input_dim, output_dim, layers, filters, strides, sizes, act, init)

    return descriptor


def to_keras(individual, num_clases):
    """
    Builds a Sequential model from a MLPDescriptor

    Parameters
    ----------
    individual: MLPDescriptor
        Evolved network descriptor
    num_clases: int

    Returns
    -------
    keras.Sequential
        Keras sequential model
    """

    def add_layer(model, layer, individual, output=False):
        """
        Adds layer to the model

        Parameters
        ----------
        model: keras.Sequential
            Updating model
        layer: int
            Layer index
        individual: MLPDescriptor
            Evolved network descriptor
        output: bool
            Whether the layer is output layer or not

        Returns
        -------
        keras.Sequential
            Updated model
        """
        # Define Keras initialization function
        init = individual.init_functions[layer]
        if init.__name__ == 'random_uniform':
            init = RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        elif init.__name__ == 'random_normal':
            init = RandomNormal(mean=0.0, stddev=0.05, seed=None)
        else:
            init = glorot_uniform()

        if output:
            # Add output dense layer
            model.add(Dense(num_clases, kernel_initializer=init))
        else:
            # Add dense layer
            model.add(Dense(individual.dims[layer], kernel_initializer=init))

        # Add activation layer
        model.add(Activation(individual.act_functions[layer]))

        # Add dropout layer
        if individual.dropout[layer]:
            model.add(Dropout(individual.dropout_probs[layer]))
        if individual.batch_norm[layer]:
            model.add(BatchNormalization())

        return model

    # Create model
    model = keras.Sequential()
    # Add input layer
    model.add(InputLayer(input_shape=(individual.input_dim,)))

    # Add layers
    for layer in range(individual.number_hidden_layers):
        model = add_layer(model, layer, individual, output=False)

    # Add output layer
    model = add_layer(model, individual.number_hidden_layers, individual, output=True)

    return model
