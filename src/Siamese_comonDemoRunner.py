'''
Created on Sep 8, 2016

@author: urishaham
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import SiameseDataHandlerDemo as dh
import Siamese_commonDemo as Siamese
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as met
import Diffusion as df

##############################################################
#######################  generate data #######################
##############################################################

dataset = 'dummy'
        
# read observations from sensor 1(S1) and sensor (2) and their common variable (x).
# The common variable is only used for evaluation of the results, not for training the Siamese net.        
if dataset == 'dummy':
    S1_train, S2_train, x_train, S1_test, S2_test, x_test = dh.readDummyData(5000)

# normalize data
m1 = np.mean(S1_train)
std1 = np.std(S1_train)
S1_train = (S1_train-m1)/std1
S1_test = (S1_test-m1)/std1

m2 = np.mean(S2_train)
std2 = np.std(S2_train)
S2_train = (S2_train-m2)/std2
S2_test = (S2_test-m2)/std2

# cut a subset of the training data for validation
n = S1_train.shape[0]
n_val = int(np.floor(0.1*n))
valSet_1 =  S1_train[:n_val]
valSet_2 =  S2_train[:n_val]
S1_train = S1_train[n_val:]
S2_train = S2_train[n_val:]
if dataset == 'dummy':
    x_train = x_train[n_val:]
    x_train = x_train[n_val:]

##########################################################

input_dim_1 = S1_train.shape[1]
input_dim_2 = S2_train.shape[1]
n_trainSamples = n-n_val
n_testSamples = S1_test.shape[0]



##########################################################
####################### config net #######################
##########################################################


if dataset == 'dummy':
    training_epochs = 200
    batch_size = 128
    display_step = 1
    layerSizes = [25,25,25,25]
    l2_penalty = 0.000
    starter_learning_rate = 1e-3
    decayRate = 0.95
    decaySteps = 200
    dropout_prob = 1 # keep probability
    lossType = 'contrastive' # either 'contrastive' or 'mse' or 'cross_entropy'
    
    
# define learning rate decay schedule    
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               decaySteps, decayRate, staircase=True)    

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) # use for dummy

#########################################################
####################### train net #######################
###########################################################

# initialize net
siamese = Siamese.Siamese(n_input1 = input_dim_1, n_input2 = input_dim_2, 
                  layerSizes = layerSizes,
                  transfer_function = tf.nn.relu,
                  l2_penalty = l2_penalty,
                  dropout_prob = dropout_prob,
                  global_step = global_step,
                  optimizer = optimizer, 
                  lossType = lossType)

# data structures to save training and validation cost at the end of each epoch
trainingCosts = np.zeros(training_epochs)
validationCosts = np.zeros(training_epochs)

global_steps_counter = 0
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_trainSamples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        global_steps_counter = global_steps_counter+1
        # prepare training batch (of positive and negative pairs)
        batch_x_1,  batch_x_2, batch_targets,= dh.get_random_block_from_data(S1_train, S2_train, batch_size)
        # Fit training using batch data
        cost = siamese.partial_fit(batch_x_1, batch_x_2, batch_targets, dropout_prob)
        # update average loss
        avg_cost += cost /  (total_batch * batch_size)
    # prepare a validation batch (of positive and negative pairs) 
    S_val_1, S_val_2, valTargets = dh.get_random_block_from_data(valSet_1, valSet_2, n_val)
    val_cost = siamese.calc_cost(S_val_1, S_val_2, valTargets)
    val_cost = val_cost/n_val
    trainingCosts[epoch] = avg_cost
    validationCosts[epoch] = val_cost
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch:", '%04d' % (epoch + 1), \
            "training loss=", "{:.9f},".format(avg_cost), "validation loss=", "{:.9f}".format(val_cost))


#################################################################
#######################  evaluate results #######################
#################################################################
X_train_1, X_train_2, trainTargets = dh.get_random_block_from_data(S1_train, S2_train, n_trainSamples)
X_test_1, X_test_2, testTargets = dh.get_random_block_from_data(S1_test, S2_test, n_testSamples)


print ("avg training loss: " + str(siamese.calc_cost(X_train_1, X_train_2, trainTargets)/n_trainSamples))
print ("avg test loss: " + str(siamese.calc_cost(X_test_1, X_test_2, testTargets)/n_testSamples))

# compute "classification" training accuracy
if lossType == 'mse' or lossType == 'cross_entropy':
    zTrain_sig = siamese.getSigDiff(X_train_1, X_train_2)
    y_pred = np.zeros(n_trainSamples)
    y_pred[zTrain_sig<.75]=0
    y_pred[zTrain_sig>.75]=1
    
if lossType == 'contrastive':
    z_diff_norm_sq_train = siamese.getZ_diff_norm_sq(X_train_1, X_train_2)
    y_pred = np.zeros(n_trainSamples)
    y_pred[z_diff_norm_sq_train<1.0]=0 # 1 is the margin
    y_pred[z_diff_norm_sq_train>1.0]=1
    
trainTargets_0 = trainTargets
trainTargets_0[trainTargets==.5]=0
trainTargets_0[trainTargets==1.]=1
    
        
print('train accuracy:')
print(np.mean(y_pred == trainTargets_0))
print('training confusion matrix: ')
print(met.confusion_matrix(trainTargets_0, y_pred))

# compute "classification" test accuracy
if lossType == 'mse' or lossType == 'cross_entropy':
    zTest_sig = siamese.getSigDiff(X_test_1, X_test_2)
    y_pred = np.zeros(n_testSamples)
    y_pred[zTest_sig<.75]=0
    y_pred[zTest_sig>.75]=1
    
if lossType == 'contrastive':
    z_diff_norm_sq_test = siamese.getZ_diff_norm_sq(X_test_1, X_test_2)
    y_pred = np.zeros(n_testSamples)
    y_pred[z_diff_norm_sq_test<1.0]=0
    y_pred[z_diff_norm_sq_test>1.0]=1
    
testTargets_0 = testTargets
testTargets_0[testTargets==.5]=0
testTargets_0[testTargets==1.]=1

print('test accuracy:')
print(np.mean(y_pred == testTargets_0))
print('test confusion matrix: ')
print(met.confusion_matrix(testTargets_0, y_pred))


plt.plot(range(training_epochs), trainingCosts, 'r--', range(training_epochs), validationCosts, 'b--')
plt.title('loss during training')
plt.xlabel('epoch')
plt.xlabel('average loss')
plt.legend(['training', 'validation'])
plt.show()
  
#################################################################
####################### diffusion  and SVD ######################
#################################################################

# net outputs:
Z1, Z2 = siamese.getCodes(S1_test, S2_test)

# diffusion
E1,v1 = df.Diffusion(Z1, k=20, nEigenVals=12)
E2,v2 = df.Diffusion(Z2, k=20, nEigenVals=12)

# svd
U1, s1, _ = np.linalg.svd(Z1)
U2, s2, _ = np.linalg.svd(Z2)

# plot 3 leading coordinates of diffusion / svd embedding of the test data, 
# colored by the value of the common variable
fig, (a1)  = plt.subplots(1,1, subplot_kw={'projection':'3d'})    
a1.scatter(E1[:,0], E1[:,1], E1[:,2], c=x_test, cmap='gist_ncar')
plt.title('diffusion embedding of sensor #1 code')
fig, (a2)  = plt.subplots(1,1, subplot_kw={'projection':'3d'})    
a2.scatter(E2[:,0], E2[:,1], E2[:,2], c=x_test, cmap='gist_ncar')
plt.title('diffusion embedding of sensor #2 code')
fig, (a3)  = plt.subplots(1,1, subplot_kw={'projection':'3d'})    
a3.scatter(U1[:,0], U1[:,1], U1[:,2], c=x_test, cmap='gist_ncar') 
plt.title('SVD embedding of sensor #1 code')   
fig, (a4)  = plt.subplots(1,1, subplot_kw={'projection':'3d'})    
a4.scatter(U2[:,0], U2[:,1], U2[:,2], c=x_test, cmap='gist_ncar')
plt.title('SVD embedding of sensor #2 code')
    