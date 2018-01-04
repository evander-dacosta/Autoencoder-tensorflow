#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:10:37 2018

@author: evanderdcosta
"""
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from ops import Dense
from base_model import BaseModel

class MLP(BaseModel):
    def __init__(self, config, sess):
        self.sess = sess
        self.config = config
        
        self.input_shape, self.output_shape, self.layer_defs, \
        self.output_activation = config.input_shape, config.output_shape, \
                                      config.layers, config.output_activation
        
        #self.cost_fn = config.cost_function
        self.cost_fn = tf.nn.softmax_cross_entropy_with_logits 
             

        super(MLP, self).__init__(config, sess)
        self.build()
             
    def build(self):
        self.w = {}
        self.layers = []
        
        with tf.variable_scope('mlp'):
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=self.input_shape,
                                        name='input')
            self.target = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.output_shape],
                                         name='output')
            for i, layer in enumerate(self.layer_defs):
                if(i == 0):
                    x = self.input
                else:
                    x = self.layers[-1]
                    
                l, self.w['layer_{}_w'.format(i)], self.w['layer_{}_b'.format(i)] = \
                    Dense(x, output_size=layer.get('n'),
                          activation_fn=layer.get('activation'),
                          name='mlp_layer_{}'.format(i+1))
                self.layers.append(l)
                
            self.output, self.w['output_w'], self.w['output_b'] = \
                       Dense(self.layers[-1], self.output_shape,
                             activation_fn=self.output_activation)
            
            
        with tf.variable_scope('optimiser'):
            self.optimiser = tf.train.AdamOptimizer()
            self.loss = tf.reduce_mean(self.cost_fn(labels=self.target, 
                                                    logits=self.output))
            self.min_op = self.optimiser.minimize(self.loss)
            
        
        # Create scalar and histogram summaries
        with tf.variable_scope('model_summary'):
            scalar_summary_tags = ['mean_loss']
            self.summary_placeholders = {}
            self.summary_ops = {}
            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None,
                                                     name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.scalar('{}'.format(tag), 
                                self.summary_placeholders[tag])
            
            histogram_summary_tags = []
            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None,
                                                     name=tag.replace(' ', '_'))
                self.summary_ops[tag] = tf.summary.histogram('{}'.format(tag), 
                                self.summary_placeholders[tag])
                
            self.writer = tf.summary.FileWriter('./logs/{}'.format(self.model_dir),
                                                self.sess.graph)
            
        self.sess.run(tf.global_variables_initializer())
        
    def add_summary(self, tag_dict, step):
        feed_dict = [(self.summary_placeholders[tag], tag_dict[tag]) for tag in tag_dict.keys()]
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()],
                                dict(feed_dict))
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, step)
             
    def fit(self, x, y, n_epochs=10):
        batchsize = self.config.batchsize
        n_iter = int(len(x) / float(batchsize)) + 1
        
        
        for epoch in range(n_epochs):
            losses = []
            for i in tqdm(range(n_iter)):
                x_train = x[i*batchsize : (i+1)*batchsize]
                y_train = y[i*batchsize : (i+1)*batchsize]
                loss, _ = self.sess.run([self.loss, self.min_op], feed_dict={
                                                        self.input:x_train,
                                                        self.target:y_train
                                                        })
                losses.append(loss)
            print("Epoch {}, loss {}".format(epoch, np.mean(losses)))
            summary = {
                         'mean_loss' : np.mean(losses)
                      }
            self.add_summary(summary, epoch)
        self.save_model()
            
    
    def predict(self, x):
        out = tf.argmax(self.output, axis=-1)
        return self.sess.run(out, feed_dict={self.input: x})