#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:11:54 2018

@author: evanderdcosta
"""

import tensorflow as tf
from utils import *

def Dense(x, output_size, activation_fn=tf.nn.relu, name='linear'):
    shape = x.get_shape().as_list()
    
    if(isinstance(activation_fn, str)):
        if(activation_fn in activation_fns.keys()):
            activation_fn = activation_fns[activation_fn]
        else:
            raise ValueError('Unknown activation '
                             'function: {}'.format(activation_fn))
        
    with tf.variable_scope(name):
        w = tf.get_variable('weight_matrix', [shape[1], output_size], 
                tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        b = tf.get_variable('bias', [output_size,], 
                            initializer=tf.constant_initializer(0.))
        out = tf.nn.bias_add(tf.matmul(x, w), b)
        
        if(activation_fn != None):
            return activation_fn(out), w, b
        else:
            return out, w, b