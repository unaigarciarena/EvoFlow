import numpy as np
from functools import reduce
from Network import NetworkDescriptor, init_functions, act_functions


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


def init_mlp_desc(input_size, output_size, nlayers, max_layer_size):

    if hasattr(input_size, '__iter__'):
        input_size = reduce(lambda x, y: x*y, input_size)
    if hasattr(output_size, '__iter__'):
        output_size = reduce(lambda x, y: x*y, output_size)

    n_hidden = np.random.randint(nlayers)+1

    dim_list = [np.random.randint(max_layer_size)+1 for _ in range(n_hidden)]

    initializers = np.random.choice(init_functions, size=n_hidden+1)
    activations = np.random.choice(act_functions, size=n_hidden+1)

    batch_norm = np.random.choice(range(0, n_hidden), size=np.random.randint(0, n_hidden, size=1), replace=False)
    dropout = np.random.choice(range(0, n_hidden), size=np.random.randint(0, n_hidden, size=1), replace=False)

    descriptor = NetworkDescriptor(n_hidden, input_size, output_size, dim_list, initializers, activations, dropout, batch_norm)

    return descriptor


def init_conv_descriptor(input_dim, output_dim):

    layers = np.random.choice([0, 1, 2], size=2)  # Number of layers = 2

    filters = [np.random.randint(1, 5) if layers[i] == 2 else -1 for i in range(layers.shape[0])]
    init = [np.random.randint(0, 2) if layers[i] == 2 else -1 for i in range(layers.shape[0])]
    act = [np.random.randint(0, 2) if layers[i] == 2 else -1 for i in range(layers.shape[0])]

    sizes = np.concatenate((np.random.randint(2, 4, size=(layers.shape[0], 2)), np.random.randint(1, 3, size=(layers.shape[0], 1))), axis=1)
    strides = np.random.randint(1, 2, size=layers.shape[0])
    mydescriptor = ConvolutionDescriptor(input_dim, output_dim, 0)

    mydescriptor.initialization(layers, filters, strides, sizes, act, init, 0)

    return mydescriptor
