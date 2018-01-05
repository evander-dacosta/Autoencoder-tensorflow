#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:10:19 2018

@author: evanderdcosta
"""

import tensorflow as tf

class Config:
    """
    The parameters set in this config are used to:
        1) Initialise the model
        2) Save and load the model
    
    Therefore, be very careful about which parameters you put in here.
    Don't put object types as the checkpoints won't save properly
    Feel free to use Python's native types, though.
    
    Use different configs for different experiments. This way
    you can save the results of different experiments as checkpoints 
    and load them whenever.
    """
    name = 'mnist_mlp_1'
    input_shape = [None, 784]
    hidden_shape = 32
    hidden_activation='relu'
    output_activation = 'linear'
    batchsize = 32
    
    tied_weights = True
    corruption = 0.4
    n_epochs = 20