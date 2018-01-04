#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:33:18 2018

@author: evanderdcosta
"""

import tensorflow as tf
import os
import inspect
from model import MLP
from config import Config


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    tf.reset_default_graph()
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    config = Config()
    
    sess = tf.Session()
    mlp = MLP(config, sess)
    mlp.fit(mnist.train.images, mnist.train.labels)