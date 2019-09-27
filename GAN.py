from data import load_fashion
import tensorflow as tf
import numpy as np
from keras.models import model_from_json
from evolution import Evolving, batch
from scipy import misc


def gan_train(nets, placeholders, sess, graph, train_inputs, _, batch_size):
    aux_ind = 0
    predictions = {}

    with graph.as_default():
        out = nets["n1"].network_building(placeholders["in"]["i1"], graph)
        predictions["gen"] = out
        out = nets["n0"].network_building(tf.layers.flatten(placeholders["in"]["i0"]), graph)
        predictions["realDisc"] = out
        out = nets["n0"].network_building(predictions["gen"], graph)
        predictions["fakeDisc"] = out

        d_loss = -tf.reduce_mean(predictions["realDisc"]) + tf.reduce_mean(predictions["fakeDisc"])
        g_loss = -tf.reduce_mean(predictions["fakeDisc"])

        g_solver = tf.train.AdamOptimizer(learning_rate=0.01).minimize(g_loss, var_list=[nets["n1"].List_weights, nets["n1"].List_bias])
        d_solver = tf.train.AdamOptimizer(learning_rate=0.01).minimize(d_loss, var_list=[nets["n0"].List_weights, nets["n0"].List_bias])
        sess.run(tf.global_variables_initializer())

        for it in range(1000):
            # Learning loop for discriminator

            z_mb = np.random.normal(size=(150, 10))

            x_mb = batch(train_inputs["i0"], batch_size, aux_ind)
            aux_ind = (aux_ind + batch_size) % train_inputs["i0"].shape[0]

            _ = sess.run([d_solver], feed_dict={placeholders["in"]["i0"]: x_mb, placeholders["in"]["i1"]: z_mb})
            _ = sess.run([g_solver], feed_dict={placeholders["in"]["i1"]: z_mb})

        return predictions


def load_model(model_name="Mobile"):

    model_paths = {"Mobile": "Mobile-99-94/", "Inception": "Inception-95-91/"}
    json_file = open(model_paths[model_name] + 'model.json', 'r')
    g_1 = tf.Graph()
    with g_1.as_default():
        class_model = model_from_json(json_file.read())
        class_model.load_weights(model_paths[model_name] + "model.h5")
        class_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    json_file.close()
    return g_1, class_model


def gan_eval(preds, placeholders, sess, graph, _, output):

    height, width = 90, 90
    with graph.as_default():
        samples = sess.run(preds["gen"], feed_dict={placeholders["i1"]: np.random.normal(size=(150, 10))})

    images = np.array([misc.imresize(x, (height, width)).astype(float) for x in iter(np.reshape(samples, (-1, 28, 28, 3)))])/255.

    with mobile_graph.as_default():
        predictions = model.predict(images)
    preds = np.argmax(predictions, axis=1)
    aux_preds = np.zeros(10)
    unique, counts = np.unique(preds, return_counts=True)
    for number, appearances in zip(unique, counts):
        aux_preds[number] = appearances
    aux_preds = aux_preds/predictions.shape[0]
    predictions = np.sort(predictions, axis=1)
    predictions = np.mean(predictions, axis=0)

    return -np.sum([aux_preds[w] * np.log(aux_preds[w] / predictions[w]) if aux_preds[w] > 0 else 0 for w in range(predictions.shape[0])]),


if __name__ == "__main__":

    mobile_graph, model = load_model()

    x_train, _, x_test, _ = load_fashion()

    e = Evolving(gan_train, 2, [x_train], [x_train], [x_test], [x_test], gan_eval, 150, 100, 50, n_inputs=[[28, 28], [10]], n_outputs=[[1], [784]])
    res = e.evolve()

    print(res)
