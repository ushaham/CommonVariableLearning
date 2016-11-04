'''
Created on Jul 12, 2016

@author: urishaham
'''

import numpy as np
from numpy import genfromtxt
import sklearn.preprocessing as prep
import SiameseFileIO as io
import os.path



def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(S1, S2, batch_size):
    n,d = S1.shape
    batch_x_1 = np.zeros((batch_size, d))
    batch_x_2 = np.zeros((batch_size, d))
    batch_targets = np.zeros(batch_size)
    for i in range(batch_size):
        label = np.random.randint(2)
        if label == 0: # positive pair
            batch_targets[i] = .5
            ind =  np.random.randint(n)
            batch_x_1[i] = S1[ind]
            batch_x_2[i] = S2[ind]
        if label == 1: # positive pair
            batch_targets[i] = 1.
            ind1 =  np.random.randint(n)
            ind2 =  np.random.randint(n)
            batch_x_1[i] = S1[ind1]
            batch_x_2[i] = S2[ind2]    
    return batch_x_1,  batch_x_2, batch_targets



def readDummyData(n=1000):
    import DummyDatasetGenerator as ddg
    #Load the data:
    tmpcls=ddg.TestDataClass(60,45,30,45,1,2,1.5,2.5)
    d=90
    S1 = np.zeros([n,d])
    S2 = np.zeros([n,d])
    x = np.zeros(n)
    for i in range(n):
        s1_i,s2_i,x_i,_,y_i,z_i = tmpcls.gettrue()
        S1[i] = s1_i
        S2[i] = s2_i
        x[i] = x_i
    p = np.random.permutation(n) 
    S1 = S1[p]
    S2 = S2[p]
    x = x[p]   
    n_train = int(np.floor(n*.8))
    S1_train = S1[:n_train]
    S2_train = S2[:n_train]
    x_train = x[:n_train]
    S1_test = S1[n_train:]
    S2_test = S2[n_train:]
    x_test= x[n_train:]
    return S1_train, S2_train, x_train, S1_test, S2_test, x_test