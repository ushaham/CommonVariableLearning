'''
Created on Sep 8, 2016

@author: urishaham
'''
import tensorflow as tf
import numpy as np
import Utils

class Siamese(object):
    def __init__(self, n_input1, n_input2, layerSizes, transfer_function=tf.nn.softplus, l2_penalty=0, dropout_prob=1,
                 global_step=None, optimizer = tf.train.AdamOptimizer(), lossType = 'sigmoid'):
        self.n_input1 = n_input1,
        self.n_input2 = n_input2,
        self.layerSizes = layerSizes
        self.transfer = transfer_function
        self.l2_penalty = l2_penalty
        self.global_step = global_step
        self.lossType = lossType
        #

        self.weights = self._initialize_weights()
        self.keepProb = tf.placeholder(tf.float32)


        # model
        self.x_1 = tf.placeholder(tf.float32, [None, self.n_input1[0]]) # input layer
        self.h1_1 = self.transfer(tf.add(tf.matmul(self.x_1, self.weights['w1_1']), self.weights['b1_1'])) # first hidden layer (nonlinear)
        self.h1_1_drop = tf.nn.dropout(self.h1_1, self.keepProb)
        self.h2_1 =  self.transfer(tf.add(tf.matmul(self.h1_1_drop, self.weights['w2_1']), self.weights['b2_1'])) # second hidden layer (nonlinear)
        self.h3_1 =  self.transfer(tf.add(tf.matmul(self.h2_1, self.weights['w3_1']), self.weights['b3_1'])) # second hidden layer (nonlinear)
        self.z1 = self.transfer(tf.add(tf.matmul(self.h3_1, self.weights['w4_1']), self.weights['b4_1'])) # output layer linear
        
        self.x_2 = tf.placeholder(tf.float32, [None, self.n_input2[0]]) # input layer
        self.h1_2 = self.transfer(tf.add(tf.matmul(self.x_2, self.weights['w1_2']), self.weights['b1_2'])) # first hidden layer (nonlinear)
        self.h1_2_drop = tf.nn.dropout(self.h1_2, self.keepProb)
        self.h2_2 =  self.transfer(tf.add(tf.matmul(self.h1_2_drop, self.weights['w2_2']), self.weights['b2_2'])) # second hidden layer (nonlinear)
        self.h3_2 =  self.transfer(tf.add(tf.matmul(self.h2_2, self.weights['w3_2']), self.weights['b3_2'])) # second hidden layer (nonlinear)        
        self.z2 = self.transfer(tf.add(tf.matmul(self.h3_2, self.weights['w4_2']), self.weights['b4_2'])) # output layer linear
        
        self.targets = tf.placeholder(tf.float32, [None]) # input layer

        # cost
        self.z_diff = tf.subtract(self.z1, self.z2)
        self.z_diff_sq = tf.square(self.z_diff)
        self.z_diff_norm_sq = tf.reduce_sum(self.z_diff_sq, 1)
        
        if self.lossType == 'mse':
            self.sigDiff = tf.nn.sigmoid(self.z_diff_norm_sq)
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.sigDiff, self.targets), 2.0))
            
        if self.lossType == 'cross_entropy':
            self.sigDiff = tf.nn.sigmoid(self.z_diff_norm_sq)
            self.cost = 0.5 * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.z_diff_norm_sq, self.targets))    
        
        if self.lossType == 'DrLIM':   
            # DrLIM loss
            targets_Mod = 2*self.targets - 1
            m = tf.constant(1, dtype=tf.float32) # margin
            zero = tf.constant(0, dtype=tf.float32)
            one = tf.constant(1, dtype=tf.float32)
            hingeTerm = tf.square(tf.maximum(zero, m-self.z_diff_norm_sq))
            self.cost = 0.5*(tf.reduce_sum((one-targets_Mod)*self.z_diff_norm_sq) + tf.reduce_sum(targets_Mod * hingeTerm))
             
        
        self.weightPenaltyCost = 0.
        
        if self.l2_penalty>0:
            self.weightPenaltyCost = self.l2_penalty*(tf.nn.l2_loss(self.weights['w1_1']) + tf.nn.l2_loss(self.weights['w2_1']) +
                                                tf.nn.l2_loss(self.weights['w3_1']) + tf.nn.l2_loss(self.weights['w4_1']) +
                                                tf.nn.l2_loss(self.weights['w1_2']) + tf.nn.l2_loss(self.weights['w2_2']) +
                                                tf.nn.l2_loss(self.weights['w3_2']) + tf.nn.l2_loss(self.weights['w4_2'])
                                                )     
        self.cost = self.cost + self.weightPenaltyCost
        
        
        
        self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1_1'] = tf.Variable(Utils.xavier_init(self.n_input1[0], self.layerSizes[0]))
        all_weights['b1_1'] = tf.Variable(tf.zeros([self.layerSizes[0]], dtype=tf.float32))
        all_weights['w2_1'] = tf.Variable(Utils.xavier_init(self.layerSizes[0], self.layerSizes[1]))
        all_weights['b2_1'] = tf.Variable(tf.zeros([self.layerSizes[1]], dtype=tf.float32))
        all_weights['w3_1'] = tf.Variable(Utils.xavier_init(self.layerSizes[1], self.layerSizes[2]))
        all_weights['b3_1'] = tf.Variable(tf.zeros([self.layerSizes[2]], dtype=tf.float32))
        all_weights['w4_1'] = tf.Variable(Utils.xavier_init(self.layerSizes[2], self.layerSizes[3]))
        all_weights['b4_1'] = tf.Variable(tf.zeros([self.layerSizes[3]], dtype=tf.float32))
        
        all_weights['w1_2'] = tf.Variable(Utils.xavier_init(self.n_input2[0], self.layerSizes[0]))
        all_weights['b1_2'] = tf.Variable(tf.zeros([self.layerSizes[0]], dtype=tf.float32))
        all_weights['w2_2'] = tf.Variable(Utils.xavier_init(self.layerSizes[0], self.layerSizes[1]))
        all_weights['b2_2'] = tf.Variable(tf.zeros([self.layerSizes[1]], dtype=tf.float32))
        all_weights['w3_2'] = tf.Variable(Utils.xavier_init(self.layerSizes[1], self.layerSizes[2]))
        all_weights['b3_2'] = tf.Variable(tf.zeros([self.layerSizes[2]], dtype=tf.float32))
        all_weights['w4_2'] = tf.Variable(Utils.xavier_init(self.layerSizes[2], self.layerSizes[3]))
        all_weights['b4_2'] = tf.Variable(tf.zeros([self.layerSizes[3]], dtype=tf.float32))
        return all_weights


    
    def partial_fit(self, X_1, X_2, targets, keepProb):
        cost, _ = self.sess.run((self.cost, self.optimizer), 
                                feed_dict={self.x_1: X_1, self.x_2: X_2, self.targets: targets, self.keepProb: keepProb})
        return cost

    def calc_cost(self, X_1, X_2, targets):
        return self.sess.run(self.cost, feed_dict = {self.x_1: X_1, self.x_2: X_2, self.targets: targets, self.keepProb: 1.0}) 

    def getWeights(self):
        return self.sess.run(self.weights)
    
    def getWeightPenaltyCost(self):
        return self.sess.run(self.weightPenaltyCost)
    
    def getCodes(self, X_1, X_2):
        return self.sess.run([self.z1, self.z2], feed_dict = {self.x_1: X_1, self.x_2: X_2, self.keepProb: 1.0})
    
    def getSigDiff(self, X_1, X_2):
        return self.sess.run(self.sigDiff, feed_dict = {self.x_1: X_1, self.x_2: X_2, self.keepProb: 1.0})  
    
    def getZ_diff_norm_sq(self, X_1, X_2):
        return self.sess.run(self.z_diff_norm_sq, feed_dict = {self.x_1: X_1, self.x_2: X_2, self.keepProb: 1.0})  
    
