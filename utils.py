#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:12:44 2018

@author: evanderdcosta
"""
import tensorflow as tf

activation_fns = {'sigmoid': tf.nn.sigmoid,
                  'softmax': tf.nn.softmax,
                  'relu': tf.nn.relu,
                  'tanh': tf.nn.tanh,
                  'linear': lambda x: x
                  }