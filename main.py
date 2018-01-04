#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import os
import inspect
from model import MLP
from config import Config


def get_accuracy(test_data, model):
    y_pred = model.predict(test_data[0])
    y_true = np.argmax(test_data[1], axis=-1)
    n_matches = (y_pred == y_true)
    return np.mean(n_matches)

def main(_):
    from tensorflow.examples.tutorials.mnist import input_data
    tf.reset_default_graph()
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    config = Config()
    
    sess = tf.Session()
    mlp = MLP(config, sess)
    mlp.fit(mnist.train.images, mnist.train.labels)
    
    print("[*] Finished Training")
    print("Test accuracy: {}".format(get_accuracy([mnist.test.images,
                                          mnist.test.labels], mlp)))
    return


if __name__ == "__main__":
    tf.app.run()
