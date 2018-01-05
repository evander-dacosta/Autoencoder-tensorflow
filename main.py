#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import inspect
from model import Autoencoder
from config import Config


def plot_reconstruction(model, test_data, index):
    image = test_data[index:index+1]
    reconstruction = model.predict(image).reshape((28, 28))
    image = image.reshape((28, 28))
    plt.figure()
    plt.subplot(211)
    plt.imshow(image)
    plt.subplot(212)
    plt.imshow(reconstruction)
    plt.show()


def main(_):
    from tensorflow.examples.tutorials.mnist import input_data
    tf.reset_default_graph()
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    config = Config()
    
    sess = tf.Session()
    model = Autoencoder(config, sess)
    model.fit(mnist.train.images)
    
    print("[*] Finished Training")
    return model, mnist


if __name__ == "__main__":
    tf.app.run()
