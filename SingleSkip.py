"""
This is a use case of EvoFlow

In this instance, we handle a classification problem, which is to be solved by two DNNs combined in a sequential layout.
The problem is the same as the one solved in Sequential.py, only that here a CNN is evolved as the first component of the model.
"""
from data import load_fashion
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import evolution
from Network import MLPDescriptor, ConvDescriptor, CNN


class SkipCNN(CNN):
    def initialization(self, graph, hypers):
        """
        This function creates all the necessary filters for the CNN
        :param graph: Graph in which the variables are created (and convolutional operations are performed)
        :param hypers: Example of how to implement a skip connection
        :return:
        """
        skip = (self.descriptor.number_hidden_layers % (hypers["skip"] - 2)) + 2
        last_c = self.descriptor.input_dim[-1]
        with graph.as_default():
            for ind in range(self.descriptor.number_hidden_layers):
                if skip == ind:
                    last_c += ref_c
                if self.descriptor.layers[ind] == 2:  # If the layer is convolutional
                    if self.descriptor.init_functions[ind] == 0:
                        w = tf.Variable(np.random.uniform(-0.1, 0.1, size=[self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], last_c, self.descriptor.filters[ind][2]]).astype('float32'), name="W"+str(ind))
                    else:
                        w = tf.Variable(np.random.normal(0, 0.03, size=[self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], last_c, self.descriptor.filters[ind][2]]).astype('float32'), name="W"+str(ind))
                    self.List_weights += [w]
                    last_c = self.descriptor.filters[ind][2]

                else:  # In case the layer is pooling, no need of weights
                    self.List_weights += [tf.Variable(-1)]
                if ind == 0:
                    ref_c = last_c

    def building(self, layer, graph, skip):
        """
        Using the filters defined in the initialization function, create the CNN
        :param layer: Input of the network
        :param graph: Graph in which variables were defined
        :param skip: Example of how to implement a skip connection
        :return: Output of the network
        """
        skip = (self.descriptor.number_hidden_layers % (skip-2)) + 2
        with graph.as_default():
            for ind in range(self.descriptor.number_hidden_layers):
                if skip == ind:
                    layer = tf.pad(layer, tf.constant([[0, 0], [0, self.List_layers[0].shape[1].value-layer.shape[1].value], [0, self.List_layers[0].shape[2].value-layer.shape[2].value], [0, 0]]))
                    layer = tf.concat((layer, self.List_layers[0]), axis=3)
                if self.descriptor.layers[ind] == 2:  # If the layer is convolutional
                    layer = tf.nn.conv2d(layer, self.List_weights[ind], (1, self.descriptor.strides[ind][0], self.descriptor.strides[ind][1], self.descriptor.strides[ind][2]), padding=[[0, 0], [0, 0], [0, 0], [0, 0]])
                elif self.descriptor.layers[ind] == 0:  # If the layer is average pooling
                    layer = tf.nn.avg_pool(layer, (1, self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], 1), (1, self.descriptor.strides[ind][0], self.descriptor.strides[ind][1], 1), padding="VALID")
                else:
                    layer = tf.nn.max_pool(layer, (1, self.descriptor.filters[ind][0], self.descriptor.filters[ind][1], 1), (1, self.descriptor.strides[ind][0], self.descriptor.strides[ind][1], 1), padding="VALID")

                if self.descriptor.act_functions[ind] is not None:  # If we have activation function
                    layer = self.descriptor.act_functions[ind](layer)
                # batch normalization and dropout not implemented (maybe pooling operations should be part of convolutional layers instead of layers by themselves)
                self.List_layers += [layer]

        return layer


evolution.descs["ConvDescriptor"] = SkipCNN

optimizers = [tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.AdamOptimizer]


def train_cnn(nets, placeholders, sess, graph, train_inputs, train_outputs, batch_size, hypers):
    aux_ind = 0
    predictions = {}
    with graph.as_default():
        out = nets["n0"].building(placeholders["in"]["i0"], graph, hypers["skip"])  # We need to flatten the output of the CNN before feeding it to the MLP
        out = tf.layers.dense(tf.layers.flatten(out), 20)
        predictions["n0"] = out
        out = nets["n1"].building(predictions["n0"], graph)
        predictions["n1"] = out

        lf = tf.losses.softmax_cross_entropy(placeholders["out"]["o1"], predictions["n1"])

        opt = optimizers[hypers["optimizer"]](learning_rate=hypers["lrate"]).minimize(lf)
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            # As the input of n1 is the output of n0, these two placeholders need no feeding
            _, loss = sess.run([opt, lf], feed_dict={placeholders["in"]["i0"]: evolution.batch(train_inputs["i0"], batch_size, aux_ind), placeholders["out"]["o1"]: evolution.batch(train_outputs["o0"], batch_size, aux_ind)})

            aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]

    return predictions


def eval_cnn(preds, placeholders, sess, graph, inputs, outputs, _):
    with graph.as_default():
        res = sess.run(tf.nn.softmax(preds["n1"]), feed_dict={placeholders["i0"]: inputs["i0"]})
        sess.close()

        return evolution.accuracy_error(res, outputs["o0"]),


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
    e = evolution.Evolving(loss=train_cnn, desc_list=[ConvDescriptor, MLPDescriptor], x_trains=[x_train], y_trains=[y_train], x_tests=[x_test], y_tests=[y_test],
                 evaluation=eval_cnn, batch_size=150, population=5, generations=10, n_inputs=[[28, 28, 3], [20]], n_outputs=[[20], [10]], cxp=0.5,
                 mtp=0.5, hyperparameters={"lrate": [0.1, 0.5, 1], "optimizer": [0, 1, 2], "skip": range(3, 10)}, no_batch_norm=False, no_dropout=False)
    a = e.evolve()

    print(a[-1])
